package httputil

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// silentProxy is a raw TCP byte-pump sitting in front of an httptest HTTPS
// server. While armed, it stops forwarding bytes in either direction on
// already-accepted connections — without closing them — emulating a
// middlebox or egress proxy that silently swallows frames on an otherwise
// still-open connection, which is exactly what the HTTP/2 ping health check
// is meant to catch. TLS passes through untouched since the proxy never
// terminates it, so the client ends up talking end-to-end to the real
// server's own certificate.
type silentProxy struct {
	listener net.Listener
	target   string
	silent   atomic.Bool
}

func newSilentProxy(t *testing.T, target string) *silentProxy {
	t.Helper()
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	require.NoError(t, err)
	p := &silentProxy{listener: ln, target: target}
	go p.acceptLoop()
	t.Cleanup(func() { _ = ln.Close() })
	return p
}

func (p *silentProxy) addr() string { return p.listener.Addr().String() }

func (p *silentProxy) goSilent() { p.silent.Store(true) }

func (p *silentProxy) acceptLoop() {
	for {
		clientConn, err := p.listener.Accept()
		if err != nil {
			return
		}
		go p.handle(clientConn)
	}
}

func (p *silentProxy) handle(clientConn net.Conn) {
	serverConn, err := net.Dial("tcp4", p.target)
	if err != nil {
		_ = clientConn.Close()
		return
	}
	go p.pump(serverConn, clientConn)
	go p.pump(clientConn, serverConn)
}

// pump copies src -> dst one read at a time, rechecking p.silent before
// every read so it stops picking up new bytes the instant the proxy goes
// silent. It never closes or resets either connection — only stops moving
// bytes — matching a middlebox that eats frames rather than tearing down
// the TCP session.
func (p *silentProxy) pump(dst, src net.Conn) {
	buf := make([]byte, 32*1024)
	for {
		if p.silent.Load() {
			time.Sleep(5 * time.Millisecond)
			continue
		}
		_ = src.SetReadDeadline(time.Now().Add(10 * time.Millisecond))
		n, err := src.Read(buf)
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return
			}
		}
		if err != nil {
			if ne, ok := err.(net.Error); ok && ne.Timeout() {
				continue // just a wakeup to recheck p.silent
			}
			return
		}
	}
}

func newH2TestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(handler)
	srv.EnableHTTP2 = true
	srv.StartTLS()
	t.Cleanup(srv.Close)
	return srv
}

// clientThrough builds an http.Client using our own newTransport (so it
// carries the HTTP/2 ping wiring under test), redirected at the TCP level to
// dial proxyAddr instead of the request's real target — letting the test
// keep using the real server's own URL/cert for TLS verification while every
// byte actually flows through the (possibly silenced) proxy.
func clientThrough(t *testing.T, proxyAddr string, serverCert *x509.Certificate, cfg *HTTPClientConfig) *http.Client {
	t.Helper()
	pool := x509.NewCertPool()
	pool.AddCert(serverCert)

	transport := newTransport(cfg)
	transport.TLSClientConfig = &tls.Config{RootCAs: pool}
	// Setting TLSClientConfig/DialContext at all normally suppresses
	// Transport's own opportunistic HTTP/2 setup (see onceSetNextProtoDefaults
	// in net/http/transport.go) — production newTransport never touches
	// either field, so this only matters for this test's proxy redirection.
	transport.ForceAttemptHTTP2 = true
	transport.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, network, proxyAddr)
	}

	return &http.Client{Transport: transport}
}

func TestHTTP2Ping_DeadConnectionIsClosedAndSurfacesAsConnectionLost(t *testing.T) {
	srv := newH2TestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	proxy := newSilentProxy(t, strings.TrimPrefix(srv.URL, "https://"))
	client := clientThrough(t, proxy.addr(), srv.Certificate(), &HTTPClientConfig{
		Timeout:              5 * time.Second,
		HTTP2IdlePingTimeout: 300 * time.Millisecond,
		HTTP2PingTimeout:     200 * time.Millisecond,
	})

	// Warm up: establish and pool one real HTTP/2 connection through the
	// (still forwarding) proxy.
	resp, err := client.Get(srv.URL)
	require.NoError(t, err)
	assert.Equal(t, "HTTP/2.0", resp.Proto, "test is only meaningful over a real h2 connection")
	_ = resp.Body.Close()

	// Go dark: the proxy keeps both TCP legs open but stops forwarding any
	// further bytes — exactly like an egress box that silently eats frames
	// on an otherwise-open connection, rather than closing it outright.
	proxy.goSilent()
	// Let the pump goroutines' next poll (10ms granularity) actually pick up
	// the silence before firing the probe request — otherwise it can race a
	// pump mid-Read and slip through on already-in-flight bytes.
	time.Sleep(100 * time.Millisecond)

	start := time.Now()
	_, err = client.Get(srv.URL)
	elapsed := time.Since(start)

	require.Error(t, err, "a request on a connection gone dark must fail once the ping health check gives up on it")
	assert.Contains(t, err.Error(), "client connection lost")
	// Detection bound: HTTP2IdlePingTimeout + HTTP2PingTimeout (300ms+200ms
	// here), with slack for goroutine scheduling — proves the dead
	// connection is actually evicted instead of hanging until
	// ResponseHeaderTimeout/request_timeout.
	assert.Less(t, elapsed, 2*time.Second, "dead connection should be detected well inside the configured ping budget")
}

func TestHTTP2Ping_SlowButAliveStreamSurvivesMultiplePingCycles(t *testing.T) {
	const sendPing = 150 * time.Millisecond
	const pingTimeout = 100 * time.Millisecond

	srv := newH2TestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		// No bytes flow on this connection while the provider "thinks" —
		// long enough to span several idle-ping cycles before responding,
		// proving a successfully-acked ping doesn't itself kill the stream
		// and that each ack keeps re-arming the idle timer.
		time.Sleep(6 * sendPing)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	proxy := newSilentProxy(t, strings.TrimPrefix(srv.URL, "https://"))
	client := clientThrough(t, proxy.addr(), srv.Certificate(), &HTTPClientConfig{
		Timeout:              10 * time.Second,
		HTTP2IdlePingTimeout: sendPing,
		HTTP2PingTimeout:     pingTimeout,
	})

	resp, err := client.Get(srv.URL)
	require.NoError(t, err, "a live connection must survive repeated successful ping/ack cycles while the upstream is merely slow, not dead")
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.Equal(t, "ok", string(body))
}
