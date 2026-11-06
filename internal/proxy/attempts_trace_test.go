package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	pricingmodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// recordClientSpans returns a context with a recording span in it, as the
// server span is in production, and a function that ends it and returns the
// aar.* attributes of the client spans beneath it (one per upstream
// request), in the order they began.
func recordClientSpans(t *testing.T) (context.Context, func() []map[string]any) {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	ctx, root := provider.Tracer("test").Start(context.Background(), "POST /v1/chat/completions")
	return ctx, func() []map[string]any {
		root.End()
		var clients []sdktrace.ReadOnlySpan
		for _, span := range recorder.Ended() {
			if span.SpanKind() == trace.SpanKindClient {
				clients = append(clients, span)
			}
		}
		sort.Slice(clients, func(i, j int) bool { return clients[i].StartTime().Before(clients[j].StartTime()) })
		attempts := make([]map[string]any, 0, len(clients))
		for _, span := range clients {
			attrs := map[string]any{}
			for _, kv := range span.Attributes() {
				if strings.HasPrefix(string(kv.Key), "aar.") {
					attrs[string(kv.Key)] = kv.Value.AsInterface()
				}
			}
			attempts = append(attempts, attrs)
		}
		return attempts
	}
}

// tracedChatRequest sends one chat request through prx and returns the
// response and its upstream requests' attributes.
func tracedChatRequest(t *testing.T, prx *Proxy) (*httptest.ResponseRecorder, []map[string]any) {
	t.Helper()
	ctx, clientSpans := recordClientSpans(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w, clientSpans()
}

func statusServer(t *testing.T, status int) *httptest.Server {
	t.Helper()
	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		if status != http.StatusOK {
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "failed"})
			return
		}
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("chatcmpl-123", "gpt-4", "Hello"))
	}))
	t.Cleanup(server.Close)
	return server
}

// A rate-limited credential, then the fallback: each upstream request's span
// says which attempt of the request it is, to which credential, and why.
func TestProxyRequest_AttemptsOnTheTrace_Fallback(t *testing.T) {
	primary := statusServer(t, http.StatusTooManyRequests)
	fallback := statusServer(t, http.StatusOK)
	prx := NewTestProxyBuilder().WithPrimaryAndFallback(primary.URL, fallback.URL).Build()

	w, attempts := tracedChatRequest(t, prx)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, attempts, 2)
	assert.Equal(t, map[string]any{
		"aar.credential": "primary", "aar.attempt.number": int64(1), "aar.attempt.kind": httputil.AttemptFirst,
		"aar.connection.reused": false,
	}, attempts[0])
	assert.Equal(t, map[string]any{
		"aar.credential": "fallback", "aar.attempt.number": int64(2), "aar.attempt.kind": httputil.AttemptFallback,
		"aar.attempt.reason": string(RetryReasonRateLimit), "aar.connection.reused": false,
	}, attempts[1])
}

// The spend event of a request that was answered by its second attempt says
// how its time divided: what the first attempt cost, and the serving
// attempt's waits for a connection and for the first byte.
func TestProxyRequest_SpendEventStages(t *testing.T) {
	const refusalTakes, answerTakes = 60 * time.Millisecond, 40 * time.Millisecond
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(refusalTakes)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "failed"})
	}))
	defer primary.Close()
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(answerTakes)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("chatcmpl-123", "gpt-4", "Hello"))
	}))
	defer fallback.Close()
	prx := NewTestProxyBuilder().WithPrimaryAndFallback(primary.URL, fallback.URL).Build()
	events := &stubKafkaManager{enabled: true}
	prx.kafkaLog = events
	setTestModelPrice(prx, "gpt-4", &pricingmodels.ModelPrice{})

	w, _ := tracedChatRequest(t, prx)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, events.events, 1)
	event := events.events[0]
	assert.Equal(t, "fallback", event.CredentialName)
	assert.Equal(t, 2, event.UpstreamAttempts)
	assert.Len(t, event.TraceID, 32)
	require.NotNil(t, event.RetryMs)
	require.NotNil(t, event.ConnWaitMs)
	require.NotNil(t, event.TTFBMs)
	require.NotNil(t, event.ConnReused)
	assert.False(t, *event.ConnReused, "the serving attempt had its connection opened")
	assert.Nil(t, event.ExitRTTMs, "no proxy on the way")
	assert.Nil(t, event.ExitConnectMs)
	assert.GreaterOrEqual(t, *event.RetryMs, refusalTakes.Milliseconds(), "the refused attempt")
	assert.GreaterOrEqual(t, *event.TTFBMs, answerTakes.Milliseconds()-1, "the serving attempt's wait for the first byte")
	assert.Less(t, *event.TTFBMs, refusalTakes.Milliseconds()+answerTakes.Milliseconds(), "and not the refused attempt's")
	require.NotNil(t, event.UpstreamSendMs)
	assert.LessOrEqual(t, *event.UpstreamSendMs+*event.RetryMs+*event.ConnWaitMs+*event.TTFBMs, event.DurationMs,
		"the stages are parts of the request's duration")
}

// A failing credential, then the next one of the same pool: a retry, for
// the reason the first failed.
func TestProxyRequest_AttemptsOnTheTrace_Retry(t *testing.T) {
	broken := statusServer(t, http.StatusInternalServerError)
	working := statusServer(t, http.StatusOK)
	credential := func(name, baseURL string) config.CredentialConfig {
		return config.CredentialConfig{Name: name, Type: config.ProviderTypeOpenAI, APIKey: "key", BaseURL: baseURL, RPM: 100, TPM: 10000}
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential("broken", broken.URL), credential("working", working.URL)).
		WithMaxProviderRetries(1).
		Build()

	// The balancer may start from either credential: ask until a request met
	// the broken one first.
	for range 4 {
		w, attempts := tracedChatRequest(t, prx)
		require.Equal(t, http.StatusOK, w.Code)
		if len(attempts) == 1 {
			assert.Equal(t, "working", attempts[0]["aar.credential"])
			assert.Equal(t, httputil.AttemptFirst, attempts[0]["aar.attempt.kind"])
			continue
		}
		require.Len(t, attempts, 2)
		assert.Equal(t, "broken", attempts[0]["aar.credential"])
		assert.Equal(t, int64(1), attempts[0]["aar.attempt.number"])
		assert.Equal(t, httputil.AttemptFirst, attempts[0]["aar.attempt.kind"])
		assert.NotContains(t, attempts[0], "aar.attempt.reason")
		assert.Equal(t, "working", attempts[1]["aar.credential"])
		assert.Equal(t, int64(2), attempts[1]["aar.attempt.number"])
		assert.Equal(t, httputil.AttemptRetry, attempts[1]["aar.attempt.kind"])
		assert.Equal(t, string(RetryReasonServerErr), attempts[1]["aar.attempt.reason"])
		return
	}
	t.Fatal("no request met the broken credential first")
}

// force_stream_for_ttft reads the stream itself, and when the stream breaks
// it sends the request to the next credential: that send is the request's
// second attempt, after the one whose stream broke.
func TestHandleSimulatedNonStream_RetryIsTheNextAttempt(t *testing.T) {
	secondary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, openAIChatSSE(
			`{"id":"c2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}`,
		))
	}))
	defer secondary.Close()
	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: "http://primary.invalid", APIKey: "primary-key", RPM: 100, TPM: 100000},
		config.CredentialConfig{Name: "secondary", Type: config.ProviderTypeOpenAI, BaseURL: secondary.URL, APIKey: "secondary-key", Priority: 1, RPM: 100, TPM: 100000},
	).WithMaxProviderRetries(2).Build()

	ctx, clientSpans := recordClientSpans(t)
	ctx = httputil.WithAttempts(ctx)
	attempts := httputil.AttemptsFrom(ctx)
	attempts.Begin("primary") // the send whose response is read below
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer master-key")
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(&errReader{data: []byte(openAIChatSSE(
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`,
		)), err: io.ErrUnexpectedEOF}),
	}
	logCtx := &RequestLogContext{StartTime: time.Now(), RequestID: "test-req", Request: req}
	cred := &config.CredentialConfig{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: "http://primary.invalid"}
	w := httptest.NewRecorder()

	prx.handleSimulatedNonStream(w, req, resp, &orchestratedRequest{streaming: false}, cred, "gpt-4o", "gpt-4o",
		logCtx, nil, "http://upstream", []byte(`{"model":"gpt-4o","messages":[]}`),
		false, false, false, time.Now(), map[string]bool{}, map[string]bool{})

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, 2, attempts.Count(), "the first send is counted once, by the loop that sent it")
	assert.Equal(t, []map[string]any{{
		"aar.credential": "secondary", "aar.attempt.number": int64(2), "aar.attempt.kind": httputil.AttemptRetry,
		"aar.attempt.reason": string(RetryReasonNetErr), "aar.connection.reused": false,
	}}, clientSpans())
}
