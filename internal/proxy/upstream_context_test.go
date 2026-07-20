package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestUpstreamRequestContext_DrainDisabled verifies that with
// drainUpstreamOnAbort off (the default), the upstream context is exactly
// r.Context(): canceling the incoming request cancels the upstream call
// immediately, with no grace period.
func TestUpstreamRequestContext_DrainDisabled(t *testing.T) {
	p := NewTestProxyBuilder().WithDrainUpstreamOnAbort(false).Build()

	reqCtx, cancelReq := context.WithCancel(context.Background())
	defer cancelReq()
	r := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil).WithContext(reqCtx)

	upstreamCtx, cancelUpstream := p.upstreamRequestContext(r)
	defer cancelUpstream()

	if upstreamCtx != reqCtx {
		t.Fatalf("expected upstreamCtx to be r.Context() unchanged when drain is disabled")
	}

	cancelReq()
	select {
	case <-upstreamCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("upstream context should be canceled immediately when drain is disabled")
	}
}

// TestUpstreamRequestContext_DrainEnabled_SurvivesClientAbort verifies that
// with drainUpstreamOnAbort on, canceling the incoming request does NOT
// immediately cancel the upstream context — it must stay alive long enough
// for the abort path to drain the real response instead of tearing down the
// TCP connection.
func TestUpstreamRequestContext_DrainEnabled_SurvivesClientAbort(t *testing.T) {
	p := NewTestProxyBuilder().WithDrainUpstreamOnAbort(true).Build()

	reqCtx, cancelReq := context.WithCancel(context.Background())
	//nolint:staticcheck // request-scoped value used only to assert propagation below
	reqCtx = context.WithValue(reqCtx, testTraceKey{}, "trace-123")
	r := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil).WithContext(reqCtx)

	upstreamCtx, cancelUpstream := p.upstreamRequestContext(r)
	defer cancelUpstream()

	if v, _ := upstreamCtx.Value(testTraceKey{}).(string); v != "trace-123" {
		t.Fatalf("expected upstream context to preserve request values (e.g. OTEL span), got %q", v)
	}

	cancelReq()

	select {
	case <-upstreamCtx.Done():
		t.Fatal("upstream context must not cancel immediately when drain is enabled")
	case <-time.After(200 * time.Millisecond):
		// Expected: still alive shortly after the client disconnected.
	}

	if err := upstreamCtx.Err(); err != nil {
		t.Fatalf("expected upstream context to still be live, got err: %v", err)
	}
}

// TestUpstreamRequestContext_DrainEnabled_CancelFuncStopsImmediately verifies
// that calling the returned cancel func (the normal request-completion path)
// still tears down the detached context right away, so no goroutine or
// connection is held open past the request's own lifetime.
func TestUpstreamRequestContext_DrainEnabled_CancelFuncStopsImmediately(t *testing.T) {
	p := NewTestProxyBuilder().WithDrainUpstreamOnAbort(true).Build()

	r := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	upstreamCtx, cancelUpstream := p.upstreamRequestContext(r)

	cancelUpstream()

	select {
	case <-upstreamCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("upstream context should cancel immediately once the request completes normally")
	}
}

type testTraceKey struct{}
