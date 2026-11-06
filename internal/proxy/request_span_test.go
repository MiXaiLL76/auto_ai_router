package proxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// recordRequestSpan returns a context with a recording span in it, as the
// server span is in production, and a function that ends it and returns
// what the router said on it: its aar.* and error.type attributes, and its
// events' names.
func recordRequestSpan(t *testing.T) (context.Context, func() (map[string]any, []string)) {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	ctx, root := provider.Tracer("test").Start(context.Background(), "POST /v1/chat/completions")
	return ctx, func() (map[string]any, []string) {
		root.End()
		for _, span := range recorder.Ended() {
			if span.SpanContext().SpanID() != root.SpanContext().SpanID() {
				continue
			}
			attrs := map[string]any{}
			for _, kv := range span.Attributes() {
				if key := string(kv.Key); strings.HasPrefix(key, "aar.") || key == "error.type" {
					attrs[key] = kv.Value.AsInterface()
				}
			}
			events := make([]string, 0, len(span.Events()))
			for _, event := range span.Events() {
				events = append(events, event.Name)
			}
			return attrs, events
		}
		t.Fatal("the request's span did not end")
		return nil, nil
	}
}

// spannedLogCtx is a request under ctx's span, sent to the credentials named.
func spannedLogCtx(ctx context.Context, sentTo ...string) *RequestLogContext {
	ctx = httputil.WithAttempts(ctx)
	for _, credential := range sentTo {
		httputil.AttemptsFrom(ctx).Begin(credential)
	}
	return &RequestLogContext{Request: httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)}
}

func TestAnnotateRequestSpan_WhoseWhatAndHow(t *testing.T) {
	ctx, said := recordRequestSpan(t)
	logCtx := spannedLogCtx(ctx, "openai_primary", "openai_fallback")
	logCtx.Token = "sk-test"
	logCtx.TokenInfo = &models.TokenInfo{
		KeyAlias: "billing-bot", UserID: "user-1", TeamID: "team-1", TeamAlias: "Billing",
		OrganizationID: "org-1", OrganizationAlias: "Acme",
	}
	logCtx.Credential = &config.CredentialConfig{Name: "openai_fallback", Type: config.ProviderTypeOpenAI}
	logCtx.RealModelID = "gpt-4o-2024-08-06"
	logCtx.Status = "success"
	logCtx.HTTPStatus = http.StatusOK

	annotateRequestSpan(logCtx)

	attrs, _ := said()
	assert.Equal(t, map[string]any{
		"aar.api_key_hash": litellmdb.HashToken("sk-test"), "aar.key_alias": "billing-bot", "aar.user_id": "user-1",
		"aar.team_id": "team-1", "aar.team_alias": "Billing", "aar.organization_id": "org-1", "aar.organization_alias": "Acme",
		"aar.credential": "openai_fallback", "aar.provider": string(config.ProviderTypeOpenAI), "aar.real_model": "gpt-4o-2024-08-06",
		"aar.upstream_attempts": int64(2), "aar.status": "success",
	}, attrs)
}

func TestAnnotateRequestSpan_AFailureNamesItsClass(t *testing.T) {
	ctx, said := recordRequestSpan(t)
	logCtx := spannedLogCtx(ctx, "openai_primary")
	logCtx.Token = "sk-test"
	logCtx.Credential = &config.CredentialConfig{Name: "openai_primary", Type: config.ProviderTypeOpenAI}
	logCtx.HTTPStatus = http.StatusTooManyRequests

	annotateRequestSpan(logCtx)

	attrs, _ := said()
	assert.Equal(t, map[string]any{
		"aar.api_key_hash": litellmdb.HashToken("sk-test"),
		"aar.credential":   "openai_primary", "aar.provider": string(config.ProviderTypeOpenAI),
		"aar.upstream_attempts": int64(1), "aar.status": "failure", "error.type": "RateLimitError",
	}, attrs, "no status was set: the response's code says how it ended")
}

func TestAnnotateRequestSpan_OnlyWhatTheRequestGotTo(t *testing.T) {
	ctx, said := recordRequestSpan(t)
	logCtx := spannedLogCtx(ctx)
	logCtx.Status = "unknown" // what a request begins with

	annotateRequestSpan(logCtx)

	attrs, _ := said()
	assert.Equal(t, map[string]any{"aar.upstream_attempts": int64(0)}, attrs,
		"no key was read, no credential reached, and no handler said how it ended")
}

// A request that no credential could take: its spend row names a stand-in,
// its span no credential.
func TestAnnotateRequestSpan_NoCredentialCouldTakeIt(t *testing.T) {
	ctx, said := recordRequestSpan(t)
	logCtx := spannedLogCtx(ctx)
	logCtx.Token = "sk-test"
	logCtx.Credential = &config.CredentialConfig{Name: "system", Type: config.ProviderTypeProxy}
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusTooManyRequests

	annotateRequestSpan(logCtx)

	attrs, _ := said()
	assert.Equal(t, map[string]any{
		"aar.api_key_hash": litellmdb.HashToken("sk-test"), "aar.upstream_attempts": int64(0),
		"aar.status": "failure", "error.type": "RateLimitError",
	}, attrs)
}

// A request made inside another has a span of its own beneath the outer
// one's: what it says of itself is there, and the outer span keeps its own.
func TestInnerRequestSpan(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	ctx, outer := provider.Tracer("test").Start(context.Background(), "POST /v1/chat/completions")

	req, done := innerRequestSpan(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx))
	logCtx := &RequestLogContext{Request: req, Token: "sk-test", Status: "failure", HTTPStatus: http.StatusBadGateway}
	logCtx.markFirstToken()
	annotateRequestSpan(logCtx)
	done(http.StatusBadGateway)
	outer.End()

	spans := recorder.Ended()
	require.Len(t, spans, 2)
	inner, root := spans[0], spans[1]
	assert.Equal(t, "POST /v1/chat/completions", inner.Name())
	assert.Equal(t, root.SpanContext().SpanID(), inner.Parent().SpanID())
	assert.Equal(t, codes.Error, inner.Status().Code)
	attrs := map[string]any{}
	for _, kv := range inner.Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}
	assert.Equal(t, map[string]any{
		"http.response.status_code": int64(http.StatusBadGateway), "aar.api_key_hash": litellmdb.HashToken("sk-test"),
		"aar.upstream_attempts": int64(0), "aar.status": "failure", "error.type": "APIConnectionError",
	}, attrs)
	require.Len(t, inner.Events(), 1)
	assert.Equal(t, "first_token", inner.Events()[0].Name)
	assert.Empty(t, root.Attributes(), "the outer request's span is its own")
	assert.Empty(t, root.Events())

	untraced, done := innerRequestSpan(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	assert.False(t, trace.SpanFromContext(untraced.Context()).IsRecording())
	assert.NotPanics(t, func() { done(http.StatusOK) })
}

func TestAnnotateRequestSpan_WithoutASpan(t *testing.T) {
	logCtx := &RequestLogContext{Request: httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), Token: "sk-test"}
	assert.NotPanics(t, func() { annotateRequestSpan(logCtx) })
	assert.NotPanics(t, func() { annotateRequestSpan(&RequestLogContext{}) })
}

func TestMarkFirstToken(t *testing.T) {
	ctx, said := recordRequestSpan(t)
	logCtx := spannedLogCtx(ctx)

	logCtx.markFirstToken()

	assert.False(t, logCtx.CompletionStartTime.IsZero())
	_, events := said()
	assert.Equal(t, []string{"first_token"}, events)

	untraced := &RequestLogContext{}
	untraced.markFirstToken()
	assert.False(t, untraced.CompletionStartTime.IsZero(), "the stamp does not need a span")
}

// Through the handler: the request's span has the moments before the
// request is first sent, in the order the router passes them, then the
// first token of a streamed answer.
func TestProxyRequest_RequestSpanMarksThePhases(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, openAIChatSSE(
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}`,
		))
	}))
	defer upstream.Close()
	prx := NewTestProxyBuilder().WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "key").Build()

	ctx, said := recordRequestSpan(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model": "gpt-4", "stream": true, "messages": [{"role": "user", "content": "Hello"}]}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	_, events := said()
	assert.Equal(t, []string{
		markAuthenticated, markBodyRead, markRouted, markRequestPrepared, markAdmitted, markFirstTokenEvent,
	}, events)
}

// A request refused at the door has the moments it got to and no more.
func TestProxyRequest_RequestSpanMarksStopWhereTheRequestDid(t *testing.T) {
	prx := NewTestProxyBuilder().WithSingleCredential("openai", config.ProviderTypeOpenAI, "http://127.0.0.1:1", "key").Build()

	ctx, said := recordRequestSpan(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer not-the-key")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusUnauthorized, w.Code, w.Body.String())
	_, events := said()
	assert.Empty(t, events)
}

func TestMark_WithoutASpan(t *testing.T) {
	assert.NotPanics(t, func() {
		(&RequestLogContext{}).mark(markBodyRead)
		(*RequestLogContext)(nil).mark(markBodyRead)
	})
}

func TestFinalStatus(t *testing.T) {
	assert.Equal(t, "success", (&RequestLogContext{HTTPStatus: http.StatusOK}).finalStatus())
	assert.Equal(t, "failure", (&RequestLogContext{HTTPStatus: http.StatusBadGateway}).finalStatus())
	assert.Equal(t, "failure", (&RequestLogContext{Status: "failure", HTTPStatus: http.StatusOK}).finalStatus(),
		"a handler's word stands: a stream can fail after its 200")
}

// Through the handler: the request's span names the credential that
// answered, not the first one chosen, and how the request ended.
func TestProxyRequest_RequestSpanNamesWhatServed(t *testing.T) {
	primary := statusServer(t, http.StatusTooManyRequests)
	fallback := statusServer(t, http.StatusOK)
	prx := NewTestProxyBuilder().WithPrimaryAndFallback(primary.URL, fallback.URL).Build()

	ctx, said := recordRequestSpan(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	attrs, _ := said()
	assert.Equal(t, "fallback", attrs["aar.credential"], "the credential that answered")
	assert.Equal(t, int64(2), attrs["aar.upstream_attempts"])
	assert.Equal(t, "success", attrs["aar.status"])
	assert.NotContains(t, attrs, "error.type")
	assert.Equal(t, litellmdb.HashToken("master-key"), attrs["aar.api_key_hash"])
}

// Through the handler: the describe call that the vision fallback makes for
// a request is a request of its own on the trace, a span beneath the outer
// one's that says what it was, with its own attempt beneath it.
func TestProxyRequest_ADescribeHasItsOwnSpan(t *testing.T) {
	prx, _ := newVisionProxy(t, newVisionUpstream(t, &visionUpstream{}).URL, describeFallback())
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	ctx, root := provider.Tracer("test").Start(context.Background(), "POST /v1/chat/completions")

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"glm","messages":[
		{"role":"user","content":[{"type":"text","text":"what is it?"},{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	root.End()
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	said := func(span sdktrace.ReadOnlySpan) map[string]any {
		attrs := map[string]any{}
		for _, kv := range span.Attributes() {
			attrs[string(kv.Key)] = kv.Value.AsInterface()
		}
		return attrs
	}
	rootID := root.SpanContext().SpanID()
	var outer, describe sdktrace.ReadOnlySpan
	attemptsOf := map[trace.SpanID][]any{}
	for _, span := range recorder.Ended() {
		switch {
		case span.SpanContext().SpanID() == rootID:
			outer = span
		case span.SpanKind() == trace.SpanKindClient:
			attemptsOf[span.Parent().SpanID()] = append(attemptsOf[span.Parent().SpanID()], said(span)["aar.attempt.number"])
		case span.Parent().SpanID() == rootID:
			describe = span
		}
	}
	require.NotNil(t, outer)
	require.NotNil(t, describe, "the describe's span, beneath the request's")

	assert.Equal(t, "POST /v1/chat/completions", describe.Name())
	assert.Equal(t, "qwen-vl", said(describe)["gen_ai.request.model"])
	assert.Equal(t, int64(http.StatusOK), said(describe)["http.response.status_code"])
	assert.Equal(t, "success", said(describe)["aar.status"])
	assert.Equal(t, "glm", said(outer)["gen_ai.request.model"], "the outer request's span is its own")
	assert.Equal(t, []any{int64(1)}, attemptsOf[describe.SpanContext().SpanID()], "the describe's one attempt is beneath its span")
	moments := func(span sdktrace.ReadOnlySpan) []string {
		names := make([]string, 0, len(span.Events()))
		for _, event := range span.Events() {
			names = append(names, event.Name)
		}
		return names
	}
	passed := []string{markAuthenticated, markBodyRead, markRouted, markRequestPrepared, markAdmitted}
	assert.Equal(t, passed, moments(describe), "the describe's moments are on its own span")
	assert.Equal(t, passed, moments(outer), "and the request's on the request's, once each")
	assert.Equal(t, []any{int64(1)}, attemptsOf[rootID], "and the request's own beneath the request's")
}
