package proxy

// Integration tests for error status propagation through a chain of real routers:
//
//	client → router1 → router2 → router3 → provider
//
// Each hop is a real Proxy instance connected to the next one via a proxy
// credential over HTTP. The invariant under test: whatever final error status
// the deepest hop produces must arrive UNCHANGED at the client of the
// outermost router (a 408 stays 408, a 429 stays 429, etc.), together with
// the error response body. Statuses may only be replaced when a hop has no
// upstream response at all (transport failure → 502/408).

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/require"
)

// buildChainRouter builds a real Proxy with a single proxy credential pointing
// at upstreamURL and serves it over HTTP so the next router can chain to it.
// The credential APIKey equals the downstream master key ("master-key") so
// auth succeeds at every hop.
func buildChainRouter(t *testing.T, upstreamURL string, timeout time.Duration) *httptest.Server {
	t.Helper()
	prx := NewTestProxyBuilder().
		WithSingleCredential("downstream", config.ProviderTypeProxy, upstreamURL, "master-key").
		WithRequestTimeout(timeout).
		Build()
	return newIPv4Server(t, http.HandlerFunc(prx.ProxyRequest))
}

// buildThreeRouterChain wires provider ← router3 ← router2 ← router1 and
// returns the outermost router (router1). Servers are closed via t.Cleanup.
func buildThreeRouterChain(t *testing.T, providerURL string) *Proxy {
	t.Helper()
	router3Srv := buildChainRouter(t, providerURL, 5*time.Second)
	t.Cleanup(router3Srv.Close)
	router2Srv := buildChainRouter(t, router3Srv.URL, 5*time.Second)
	t.Cleanup(router2Srv.Close)
	return NewTestProxyBuilder().
		WithSingleCredential("router2", config.ProviderTypeProxy, router2Srv.URL, "master-key").
		WithRequestTimeout(5 * time.Second).
		Build()
}

// TestRouterChain_ErrorStatusPropagatedThroughChain verifies that ANY error
// status produced by the provider behind router3 travels unchanged through
// router3 → router2 → router1 to the client. Covers both non-retryable
// statuses (plain passthrough) and retryable ones (400, 401, 402, 403, 404, 429,
// 5xx) — with no alternative credentials at any hop, the original response
// must be returned, never remapped to a different code.
func TestRouterChain_ErrorStatusPropagatedThroughChain(t *testing.T) {
	statuses := []int{
		http.StatusBadRequest,            // 400 — retryable class
		http.StatusUnauthorized,          // 401 — retryable class (auth)
		http.StatusPaymentRequired,       // 402 — retryable class (payment)
		http.StatusForbidden,             // 403 — retryable class (auth)
		http.StatusNotFound,              // 404 — retryable class
		http.StatusRequestTimeout,        // 408 — non-retryable passthrough
		http.StatusConflict,              // 409 — non-retryable passthrough
		http.StatusRequestEntityTooLarge, // 413 — non-retryable passthrough
		http.StatusUnprocessableEntity,   // 422 — non-retryable passthrough
		http.StatusTooManyRequests,       // 429 — retryable class (rate limit)
		http.StatusInternalServerError,   // 500 — retryable class
		http.StatusBadGateway,            // 502 — retryable class
		http.StatusServiceUnavailable,    // 503 — retryable class
		529,                              // non-standard (Anthropic overloaded)
	}

	for _, status := range statuses {
		t.Run(fmt.Sprintf("status_%d", status), func(t *testing.T) {
			provider := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{"message": "provider_error_marker", "type": "upstream_error"},
				})
			}))
			defer provider.Close()

			router1 := buildThreeRouterChain(t, provider.URL)
			w := doChainRequest(router1)

			require.Equal(t, status, w.Code,
				"status %d from the provider must reach the outermost client unchanged", status)
			require.NotContains(t, w.Body.String(), "provider_error_marker")
			switch status {
			case http.StatusBadRequest:
				require.Contains(t, w.Body.String(), "Invalid request")
			case http.StatusTooManyRequests:
				require.Contains(t, w.Body.String(), "Rate limit exceeded")
			case http.StatusRequestTimeout, http.StatusGatewayTimeout:
				require.Contains(t, w.Body.String(), "Request timed out")
			default:
				require.Contains(t, w.Body.String(), "Request failed")
			}
		})
	}
}

// TestRouterChain_TimeoutAtDeepestHop_408Propagated verifies the case where
// the error originates mid-chain rather than at the provider: router3's
// upstream hangs, router3 converts its client timeout into a 408 response,
// and that 408 must reach router1's client as 408 (not 502 or any other
// remapped status).
func TestRouterChain_TimeoutAtDeepestHop_408Propagated(t *testing.T) {
	// Provider hangs until router3's client gives up and closes the connection.
	// The body must be drained first — otherwise the server never starts the
	// background read that detects the disconnect and cancels the context.
	provider := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second): // safety net so server.Close never hangs
		}
	}))
	defer provider.Close()

	// router3 has a short timeout so it times out against the hanging provider;
	// router2 and router1 have generous timeouts so the 408 they receive comes
	// from router3's response, not from their own timers.
	router3Srv := buildChainRouter(t, provider.URL, 200*time.Millisecond)
	defer router3Srv.Close()
	router2Srv := buildChainRouter(t, router3Srv.URL, 5*time.Second)
	defer router2Srv.Close()
	router1 := NewTestProxyBuilder().
		WithSingleCredential("router2", config.ProviderTypeProxy, router2Srv.URL, "master-key").
		WithRequestTimeout(5 * time.Second).
		Build()

	w := doChainRequest(router1)

	require.Equal(t, http.StatusRequestTimeout, w.Code,
		"408 produced by router3's timeout must reach the outermost client as 408")
	require.Contains(t, w.Body.String(), "timeout_error",
		"OpenAI-compatible timeout error body must be preserved through the chain")
}
