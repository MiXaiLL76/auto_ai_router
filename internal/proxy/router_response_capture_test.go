package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// collectingRouterResponseManager is a stub router-response manager that
// keeps the events it was given, so a test can assert on what the router
// would have published.
type collectingRouterResponseManager struct {
	mu     sync.Mutex
	events []*kafkalog.RouterResponseEvent
}

func (m *collectingRouterResponseManager) LogRouterResponse(event *kafkalog.RouterResponseEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, event)
	return nil
}

func (m *collectingRouterResponseManager) IsEnabled() bool { return true }
func (m *collectingRouterResponseManager) IsHealthy() bool { return true }
func (m *collectingRouterResponseManager) Stats() kafkalog.Stats {
	return kafkalog.Stats{}
}
func (m *collectingRouterResponseManager) Shutdown(_ context.Context) error { return nil }

func (m *collectingRouterResponseManager) snapshot() []*kafkalog.RouterResponseEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*kafkalog.RouterResponseEvent, len(m.events))
	copy(out, m.events)
	return out
}

// TestRouterResponse_CapturesAndPublishesTheClientBody is the end-to-end
// check of the feature: a request that gets a normal answer must produce a
// router-response event carrying that answer (masked), on its own topic.
//
// Without the feature there is nowhere to look for "what did this key
// actually get": the spend log carries the usage and the cost, and the
// request log carries what was asked, but not what came back.
func TestRouterResponse_CapturesAndPublishesTheClientBody(t *testing.T) {
	const answer = `{
	  "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "the model's own answer", "role": "assistant"}}],
	  "created": 1794422795,
	  "id": "chatcmpl-abc",
	  "model": "gpt-4o-mini",
	  "object": "chat.completion",
	  "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15}
	}`

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(answer))
	}))
	defer upstream.Close()

	mgr := &collectingRouterResponseManager{}
	prx := NewTestProxyBuilder().
		WithSingleCredential("cred", "openai", upstream.URL, "k").
		WithMaxProviderRetries(0).
		WithRouterResponseLog(mgr, true).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, http.StatusOK, rec.Code, "the request itself must succeed: %s", rec.Body.String())
	assert.Contains(t, rec.Body.String(), "the model's own answer",
		"the client must still get the real answer")

	events := mgr.snapshot()
	require.Len(t, events, 1, "exactly one router-response event must be published")
	event := events[0]
	assert.NotContains(t, event.ResponseBody, "the model's own answer",
		"the published body must be masked")
	assert.Contains(t, event.ResponseBody, routerResponseRedactedPlaceholder)
	assert.Contains(t, event.ResponseBody, "gpt-4o-mini",
		"the model must survive in the published body")
	assert.Equal(t, http.StatusOK, event.HTTPStatus)
	assert.Equal(t, "success", event.Status)
	assert.Equal(t, "cred", event.Credential)
	assert.NotEmpty(t, event.RequestID)
}

// TestRouterResponse_NotPublishedWhenDisabled pins that the capture costs
// nothing on a router that does not use the feature: with the path off, no
// event is published and the body's text is not held anywhere.
func TestRouterResponse_NotPublishedWhenDisabled(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"secret"}}],"model":"m"}`))
	}))
	defer upstream.Close()

	mgr := &collectingRouterResponseManager{}
	// The manager is a stub that reports itself enabled, but the proxy's own
	// flag is what gates the capture -- so this also pins that the two
	// agree: a manager that says "enabled" without the proxy knowing means
	// no capture.
	prx := NewTestProxyBuilder().
		WithSingleCredential("cred", "openai", upstream.URL, "k").
		WithMaxProviderRetries(0).
		Build()
	prx.routerResponseLog = mgr
	prx.routerResponsesEnabled = false

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Empty(t, mgr.snapshot(), "nothing must be published when the path is off")
}

// TestRouterResponse_CapturesTheFailedBody pins that a failed request's
// answer is captured too: the client got a body, and that body is what an
// operator needs to see.
func TestRouterResponse_CapturesTheFailedBody(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"the provider's words"}}`))
	}))
	defer upstream.Close()

	mgr := &collectingRouterResponseManager{}
	prx := NewTestProxyBuilder().
		WithSingleCredential("cred", "openai", upstream.URL, "k").
		WithMaxProviderRetries(0).
		WithRouterResponseLog(mgr, true).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	events := mgr.snapshot()
	require.Len(t, events, 1, "the failed answer must be captured too")
	assert.Equal(t, "failure", events[0].Status)
	assert.Equal(t, http.StatusBadRequest, events[0].HTTPStatus)
	assert.NotEmpty(t, events[0].ErrorClass, "the failure must be classifiable from the row")
}

// TestRouterResponse_PublishedEvenWhenTheModelCannotBePriced pins that the
// response stream does not depend on the billing configuration.
//
// The cost resolution fails closed when a model's price is missing (the
// spend row is dropped and the error returned), so publishing the response
// after it would mean a router that cannot price a model also cannot show
// what that model answered -- which is exactly when an operator is looking
// for it. The body itself is captured before any of this, at the point the
// response was produced.
func TestRouterResponse_PublishedEvenWhenTheModelCannotBePriced(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"the answer"}}],"model":"a-model-with-no-price"}`))
	}))
	defer upstream.Close()

	mgr := &collectingRouterResponseManager{}
	prx := NewTestProxyBuilder().
		WithSingleCredential("cred", "openai", upstream.URL, "k").
		WithMaxProviderRetries(0).
		WithRouterResponseLog(mgr, true).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"a-model-with-no-price","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	events := mgr.snapshot()
	require.Len(t, events, 1,
		"the answer must be published even when the model cannot be priced")
	assert.Contains(t, events[0].ResponseBody, routerResponseRedactedPlaceholder)
}

// TestRouterResponse_EmbeddingIsNotPublished pins that an embedding answer
// produces NO router-response event at all -- not a row with an empty body.
//
// An embedding body is the vector of the client's own input, not the model's
// words, and it is ~40 KB; an image body is a base64 blob that hits the
// capture cap every time. Together they were a fifth of the stream's daily
// volume in rows that say nothing about the exchange.
func TestRouterResponse_EmbeddingIsNotPublished(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-3-small"}`))
	}))
	defer upstream.Close()

	mgr := &collectingRouterResponseManager{}
	prx := NewTestProxyBuilder().
		WithSingleCredential("cred", "openai", upstream.URL, "k").
		WithMaxProviderRetries(0).
		WithRouterResponseLog(mgr, true).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"text-embedding-3-small","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Empty(t, mgr.snapshot(),
		"an embedding answer must not produce a router-response event")
}

// TestRouterResponse_CapturesTheSimulatedPath pins that the forced-stream
// path -- how a non-streaming client is served in production -- captures the
// reassembled body too. Without this, the feature would be silent for every
// deployment that runs with force_stream_for_ttft on.
func TestRouterResponse_CapturesTheSimulatedPath(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"the streamed answer\"}}]}\n\n" +
			"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	mgr := &collectingRouterResponseManager{}
	prx := NewTestProxyBuilder().
		WithSingleCredential("cred", "openai", upstream.URL, "k").
		WithMaxProviderRetries(0).
		WithForceStreamForTTFT(true).
		WithRouterResponseLog(mgr, true).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	events := mgr.snapshot()
	require.Len(t, events, 1, "the forced-stream path must capture the answer too")
	assert.NotContains(t, events[0].ResponseBody, "the streamed answer",
		"the captured body must be masked")
	assert.Contains(t, events[0].ResponseBody, routerResponseRedactedPlaceholder)
}
