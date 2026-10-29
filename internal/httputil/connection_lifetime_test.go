package httputil

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testLifetime = 10 * time.Minute

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// testLifetimeTransport is a lifetimeTransport of testLifetime on a fake clock.
func testLifetimeTransport(build func() *http.Transport) (*lifetimeTransport, *fakeClock) {
	clock := &fakeClock{now: time.Unix(1_700_000_000, 0)}
	t := newLifetimeTransport(testLifetime, build)
	t.now = clock.Now
	return t, clock
}

// connServer is an origin that knows its connections: which remote address
// served each request, and which connections are closed.
type connServer struct {
	*httptest.Server
	mu      sync.Mutex
	closed  map[string]bool
	release chan struct{} // /slow writes its headers, then waits for this
}

func newConnServer(t *testing.T, unstarted *httptest.Server) *connServer {
	t.Helper()
	s := &connServer{Server: unstarted, closed: map[string]bool{}, release: make(chan struct{})}
	s.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Remote", r.RemoteAddr)
		if r.URL.Path == "/slow" {
			w.WriteHeader(http.StatusOK)
			w.(http.Flusher).Flush()
			<-s.release
		}
		_, _ = io.WriteString(w, "ok")
	})
	s.Config.ConnState = func(c net.Conn, state http.ConnState) {
		if state == http.StateClosed || state == http.StateHijacked {
			s.mu.Lock()
			s.closed[c.RemoteAddr().String()] = true
			s.mu.Unlock()
		}
	}
	return s
}

func (s *connServer) isClosed(remote string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed[remote]
}

// get sends a request and returns the remote address the server saw.
func get(t *testing.T, client *http.Client, url string) string {
	t.Helper()
	resp, err := client.Get(url) //nolint:noctx // test
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	return resp.Header.Get("X-Remote")
}

func plainBuild() *http.Transport { return newTransport(nil) }

func TestLifetimeTransport_ReusedWithinItsLifetimeThenRetired(t *testing.T) {
	srv := newConnServer(t, httptest.NewUnstartedServer(nil))
	srv.Start()
	defer srv.Close()
	lt, clock := testLifetimeTransport(plainBuild)
	client := &http.Client{Transport: lt}
	defer lt.CloseIdleConnections()

	old := get(t, client, srv.URL)
	clock.Advance(testLifetime - time.Second)
	assert.Equal(t, old, get(t, client, srv.URL), "younger than its lifetime: reused")

	clock.Advance(time.Second)
	fresh := get(t, client, srv.URL)
	assert.NotEqual(t, old, fresh, "a request at the lifetime dials")
	assert.Eventually(t, func() bool { return srv.isClosed(old) }, 2*time.Second, 5*time.Millisecond, "the idle old connection closes")
	assert.False(t, srv.isClosed(fresh))
	assert.Equal(t, fresh, get(t, client, srv.URL), "the new generation's connection is reused")
}

func TestLifetimeTransport_RequestInFlightFinishesThenItsConnectionCloses(t *testing.T) {
	srv := newConnServer(t, httptest.NewUnstartedServer(nil))
	srv.Start()
	defer srv.Close()
	lt, clock := testLifetimeTransport(plainBuild)
	client := &http.Client{Transport: lt}
	defer lt.CloseIdleConnections()

	slow, err := client.Get(srv.URL + "/slow") //nolint:noctx // test
	require.NoError(t, err)
	old := slow.Header.Get("X-Remote")

	clock.Advance(testLifetime)
	fresh := get(t, client, srv.URL)
	assert.NotEqual(t, old, fresh)
	assert.False(t, srv.isClosed(old), "a retired connection keeps its request")

	close(srv.release)
	body, err := io.ReadAll(slow.Body)
	require.NoError(t, err, "the request on the retired connection completes")
	assert.Equal(t, "ok", string(body))
	require.NoError(t, slow.Body.Close())
	assert.Eventually(t, func() bool { return srv.isClosed(old) }, 2*time.Second, 5*time.Millisecond)
	assert.False(t, srv.isClosed(fresh))
}

func TestLifetimeTransport_NoRequestOnAConnectionOlderThanItsLifetime(t *testing.T) {
	srv := newConnServer(t, httptest.NewUnstartedServer(nil))
	srv.Start()
	defer srv.Close()
	lt, clock := testLifetimeTransport(plainBuild)
	client := &http.Client{Transport: lt}
	defer lt.CloseIdleConnections()

	born := map[string]time.Time{}
	for range 60 {
		remote := get(t, client, srv.URL)
		if _, ok := born[remote]; !ok {
			born[remote] = clock.Now()
		}
		assert.Less(t, clock.Now().Sub(born[remote]), testLifetime, "request on %s", remote)
		clock.Advance(testLifetime / 7)
	}
	assert.GreaterOrEqual(t, len(born), 6, "60 requests over ~8.5 lifetimes")
}

// connectProxy is an HTTP CONNECT proxy, the egress path's shape: it counts
// the tunnels it holds open.
type connectProxy struct {
	listener net.Listener
	open     atomic.Int32
	total    atomic.Int32
}

func newConnectProxy(t *testing.T) *connectProxy {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	p := &connectProxy{listener: l}
	t.Cleanup(func() { _ = l.Close() })
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			go p.tunnel(c)
		}
	}()
	return p
}

func (p *connectProxy) tunnel(client net.Conn) {
	defer func() { _ = client.Close() }()
	req, err := http.ReadRequest(bufio.NewReader(client))
	if err != nil || req.Method != http.MethodConnect {
		return
	}
	upstream, err := net.Dial("tcp", req.Host)
	if err != nil {
		_, _ = io.WriteString(client, "HTTP/1.1 502 Bad Gateway\r\n\r\n")
		return
	}
	defer func() { _ = upstream.Close() }()
	p.total.Add(1)
	p.open.Add(1)
	defer p.open.Add(-1)
	_, _ = io.WriteString(client, "HTTP/1.1 200 Connection established\r\n\r\n")
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(upstream, client); done <- struct{}{} }()
	go func() { _, _ = io.Copy(client, upstream); done <- struct{}{} }()
	<-done
}

func tlsBuild(cert *x509.Certificate, h2 bool) func() *http.Transport {
	pool := x509.NewCertPool()
	pool.AddCert(cert)
	return func() *http.Transport {
		tr := newTransport(nil)
		tr.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
		tr.ForceAttemptHTTP2 = h2
		return tr
	}
}

// The egress path: HTTPS through a CONNECT proxy, HTTP/1.1. A rotation leaves
// the proxy with no tunnel of the old generation once its request ends, which
// is what lets the proxy drain.
func TestLifetimeTransport_TunnelsThroughAProxyEndWithTheirRequest(t *testing.T) {
	srv := newConnServer(t, httptest.NewUnstartedServer(nil))
	srv.StartTLS()
	defer srv.Close()
	proxy := newConnectProxy(t)
	lt, clock := testLifetimeTransport(tlsBuild(srv.Certificate(), false))
	client := &http.Client{Transport: lt}
	defer lt.CloseIdleConnections()
	ctx := WithProxyURL(context.Background(), "http://user:secret@"+proxy.listener.Addr().String())
	do := func(path string) *http.Response {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+path, strings.NewReader("{}"))
		require.NoError(t, err)
		resp, err := client.Do(req)
		require.NoError(t, err)
		assert.Equal(t, 1, resp.ProtoMajor)
		return resp
	}
	finish := func(resp *http.Response) {
		_, err := io.Copy(io.Discard, resp.Body)
		require.NoError(t, err)
		require.NoError(t, resp.Body.Close())
	}

	slow := do("/slow")
	finish(do("/"))
	require.EqualValues(t, 2, proxy.open.Load(), "a tunnel for the slow request, one idle")

	clock.Advance(testLifetime)
	finish(do("/"))
	assert.Eventually(t, func() bool { return proxy.open.Load() == 2 }, 2*time.Second, 5*time.Millisecond,
		"the idle old tunnel closed, the new one is open, the slow one still works")
	close(srv.release)
	finish(slow)
	assert.Eventually(t, func() bool { return proxy.open.Load() == 1 }, 2*time.Second, 5*time.Millisecond,
		"only the new generation's tunnel is left")
	assert.EqualValues(t, 3, proxy.total.Load())
}

// HTTP/2: a retired connection still carrying a stream closes when the stream
// ends (the body's Close; HTTP/2 has no PutIdleConn).
func TestLifetimeTransport_HTTP2ConnectionClosesWhenItsLastStreamEnds(t *testing.T) {
	unstarted := httptest.NewUnstartedServer(nil)
	unstarted.EnableHTTP2 = true
	srv := newConnServer(t, unstarted)
	srv.StartTLS()
	defer srv.Close()
	lt, clock := testLifetimeTransport(tlsBuild(srv.Certificate(), true))
	client := &http.Client{Transport: lt}
	defer lt.CloseIdleConnections()

	slow, err := client.Get(srv.URL + "/slow") //nolint:noctx // test
	require.NoError(t, err)
	require.Equal(t, 2, slow.ProtoMajor)
	old := slow.Header.Get("X-Remote")
	assert.Equal(t, old, get(t, client, srv.URL), "HTTP/2 multiplexes onto the open connection")

	clock.Advance(testLifetime)
	fresh := get(t, client, srv.URL)
	assert.NotEqual(t, old, fresh)
	assert.False(t, srv.isClosed(old))

	close(srv.release)
	_, err = io.Copy(io.Discard, slow.Body)
	require.NoError(t, err)
	require.NoError(t, slow.Body.Close())
	assert.Eventually(t, func() bool { return srv.isClosed(old) }, 2*time.Second, 5*time.Millisecond)
	assert.False(t, srv.isClosed(fresh))
}

// A caller may read a body to EOF and never close it, as the proxy does with
// non-streaming responses: the request still ends there, and a retired
// connection still closes.
func TestLifetimeTransport_ConnectionClosesWhenItsLastBodyIsReadButNotClosed(t *testing.T) {
	for _, h2 := range []bool{false, true} {
		t.Run(map[bool]string{false: "HTTP/1.1", true: "HTTP/2"}[h2], func(t *testing.T) {
			unstarted := httptest.NewUnstartedServer(nil)
			unstarted.EnableHTTP2 = h2
			srv := newConnServer(t, unstarted)
			srv.StartTLS()
			defer srv.Close()
			lt, clock := testLifetimeTransport(tlsBuild(srv.Certificate(), h2))
			client := &http.Client{Transport: lt}
			defer lt.CloseIdleConnections()

			slow, err := client.Get(srv.URL + "/slow") //nolint:noctx,bodyclose // test: read to EOF, never closed
			require.NoError(t, err)
			old := slow.Header.Get("X-Remote")

			clock.Advance(testLifetime)
			fresh := get(t, client, srv.URL)
			assert.NotEqual(t, old, fresh)

			close(srv.release)
			_, err = io.Copy(io.Discard, slow.Body)
			require.NoError(t, err)
			assert.Eventually(t, func() bool { return srv.isClosed(old) }, 3*time.Second, 5*time.Millisecond)
			assert.False(t, srv.isClosed(fresh))
		})
	}
}

// Under -race: requests from many goroutines while the clock moves; every one
// succeeds, and the old generations' connections all close, including those of
// requests that entered a generation as it retired.
func TestLifetimeTransport_ConcurrentRequestsAcrossRotations(t *testing.T) {
	srv := newConnServer(t, httptest.NewUnstartedServer(nil))
	srv.Start()
	defer srv.Close()
	lt, clock := testLifetimeTransport(plainBuild)
	client := &http.Client{Transport: lt}
	defer lt.CloseIdleConnections()

	var remotes sync.Map
	var wg sync.WaitGroup
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(2 * time.Millisecond):
				clock.Advance(testLifetime / 3)
			}
		}
	}()
	for range 16 {
		wg.Go(func() {
			for range 50 {
				resp, err := client.Get(srv.URL) //nolint:noctx // test
				if !assert.NoError(t, err) {
					return
				}
				_, err = io.Copy(io.Discard, resp.Body)
				assert.NoError(t, err)
				assert.NoError(t, resp.Body.Close())
				remotes.Store(resp.Header.Get("X-Remote"), true)
			}
		})
	}
	wg.Wait()
	close(stop)

	// A last rotation retires every generation that served the load.
	clock.Advance(testLifetime)
	last := get(t, client, srv.URL)
	assert.Eventually(t, func() bool {
		open := 0
		remotes.Range(func(k, _ any) bool {
			if k.(string) != last && !srv.isClosed(k.(string)) {
				open++
			}
			return true
		})
		return open == 0
	}, 2*time.Second, 10*time.Millisecond, "every old generation's connection is closed")
}

func TestNewRoundTripper_MaxConnectionLifetime(t *testing.T) {
	_, plain := newRoundTripper(nil).(*http.Transport)
	assert.True(t, plain, "no config: today's transport")
	_, plain = newRoundTripper(&HTTPClientConfig{}).(*http.Transport)
	assert.True(t, plain, "zero lifetime: today's transport")
	lt, ok := newRoundTripper(&HTTPClientConfig{MaxConnectionLifetime: time.Minute, IdleConnTimeout: 42 * time.Second}).(*lifetimeTransport)
	require.True(t, ok)
	assert.Equal(t, time.Minute, lt.lifetime)
	assert.Equal(t, 42*time.Second, lt.build().IdleConnTimeout, "generations are built like the plain transport")
}
