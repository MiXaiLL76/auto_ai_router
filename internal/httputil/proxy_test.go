package httputil

import (
	"bufio"
	"context"
	"encoding/base64"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCredentialProxyIsolation(t *testing.T) {
	var directCalls atomic.Int32
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		directCalls.Add(1)
		_, _ = io.WriteString(w, "direct")
	}))
	defer origin.Close()
	newProxy := func(label string) *httptest.Server {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, origin.URL+"/v1/chat/completions", r.URL.String())
			assert.Equal(t, "Basic "+base64.StdEncoding.EncodeToString([]byte("user:secret")), r.Header.Get("Proxy-Authorization"))
			assert.Equal(t, "Bearer provider-key", r.Header.Get("Authorization"))
			_, _ = io.WriteString(w, label)
		}))
		t.Cleanup(server.Close)
		return server
	}
	first, second := newProxy("first"), newProxy("second")
	client := NewHTTPClient(nil)
	defer client.CloseIdleConnections()
	for _, tc := range []struct{ proxy, want string }{
		{strings.Replace(first.URL, "://", "://user:secret@", 1), "first"},
		{strings.Replace(second.URL, "://", "://user:secret@", 1), "second"},
		{"", "direct"},
		{strings.Replace(first.URL, "://", "://user:secret@", 1), "first"},
	} {
		req, err := http.NewRequestWithContext(WithProxyURL(context.Background(), tc.proxy), http.MethodPost, origin.URL+"/v1/chat/completions", nil)
		require.NoError(t, err)
		req.Header.Set("Authorization", "Bearer provider-key")
		resp, err := client.Do(req)
		require.NoError(t, err)
		body, err := io.ReadAll(resp.Body)
		require.NoError(t, resp.Body.Close())
		require.NoError(t, err)
		require.Equal(t, tc.want, string(body))
	}
	require.EqualValues(t, 1, directCalls.Load())
}

func TestCredentialProxyDoesNotFallBackToDirect(t *testing.T) {
	var directCalls atomic.Int32
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { directCalls.Add(1) }))
	defer origin.Close()
	unavailable := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	unavailable.Close()
	client := NewHTTPClient(nil)
	defer client.CloseIdleConnections()
	for _, proxyURL := range []string{unavailable.URL, "http://user:secret%@invalid"} {
		req, err := http.NewRequestWithContext(WithProxyURL(context.Background(), proxyURL), http.MethodGet, origin.URL, nil)
		require.NoError(t, err)
		_, err = client.Do(req)
		require.Error(t, err)
		require.NotContains(t, err.Error(), "secret")
	}
	require.Zero(t, directCalls.Load())
}

func TestDiscoveryUsesCredentialProxy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "unreachable.invalid", r.URL.Host)
		assert.Equal(t, "Bearer discovery-key", r.Header.Get("Authorization"))
		_, _ = io.WriteString(w, r.URL.Path)
	}))
	defer server.Close()
	cred := &config.CredentialConfig{Name: t.Name(), BaseURL: "http://unreachable.invalid", APIKey: "discovery-key", ProxyURL: server.URL}
	for _, path := range []string{"/health", "/v1/models"} {
		body, err := FetchFromProxy(context.Background(), cred, path, testhelpers.NewTestLogger())
		require.NoError(t, err)
		require.Equal(t, path, string(body))
	}
}

func TestCredentialProxyHTTPSConnect(t *testing.T) {
	var connects atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		connects.Add(1)
		assert.Equal(t, http.MethodConnect, r.Method)
		assert.Equal(t, "upstream.invalid:443", r.Host)
		assert.Equal(t, "Basic "+base64.StdEncoding.EncodeToString([]byte("user:secret")), r.Header.Get("Proxy-Authorization"))
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer server.Close()
	client := NewHTTPClient(nil)
	proxyURL := strings.Replace(server.URL, "://", "://user:secret@", 1)
	req, err := http.NewRequestWithContext(WithProxyURL(context.Background(), proxyURL), http.MethodGet, "https://upstream.invalid/v1/models", nil)
	require.NoError(t, err)
	_, err = client.Do(req)
	require.Error(t, err)
	require.Contains(t, err.Error(), "Bad Gateway")
	require.NotContains(t, err.Error(), "secret")
	require.EqualValues(t, 1, connects.Load())
}

func TestCredentialProxySOCKS(t *testing.T) {
	for _, scheme := range []string{"socks5", "socks5h"} {
		t.Run(scheme, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			defer func() { _ = listener.Close() }()
			done := make(chan struct{})
			go func() {
				defer close(done)
				conn, err := listener.Accept()
				if !assert.NoError(t, err) {
					return
				}
				defer func() { _ = conn.Close() }()
				header := make([]byte, 3)
				_, err = io.ReadFull(conn, header)
				if !assert.NoError(t, err) {
					return
				}
				assert.Equal(t, []byte{5, 1, 0}, header)
				_, _ = conn.Write([]byte{5, 0})
				header = make([]byte, 5)
				_, err = io.ReadFull(conn, header)
				if !assert.NoError(t, err) {
					return
				}
				assert.Equal(t, []byte{5, 1, 0, 3}, header[:4])
				address := make([]byte, int(header[4])+2)
				_, err = io.ReadFull(conn, address)
				if !assert.NoError(t, err) {
					return
				}
				assert.Equal(t, "upstream.invalid", string(address[:len(address)-2]))
				_, _ = conn.Write([]byte{5, 0, 0, 1, 127, 0, 0, 1, 0, 80})
				req, err := http.ReadRequest(bufio.NewReader(conn))
				if !assert.NoError(t, err) {
					return
				}
				_ = req.Body.Close()
				assert.Equal(t, "/v1/models", req.URL.Path)
				_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nproxied")
			}()
			client := NewHTTPClient(nil)
			req, err := http.NewRequestWithContext(WithProxyURL(context.Background(), scheme+"://"+listener.Addr().String()), http.MethodGet, "http://upstream.invalid/v1/models", nil)
			require.NoError(t, err)
			resp, err := client.Do(req)
			require.NoError(t, err)
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, resp.Body.Close())
			require.NoError(t, err)
			require.Equal(t, "proxied", string(body))
			<-done
		})
	}
}
