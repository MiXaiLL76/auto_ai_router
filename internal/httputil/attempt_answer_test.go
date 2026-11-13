package httputil

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// answering is a provider that answers every request with status, header
// and body.
func answering(t *testing.T, status int, header map[string]string, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		for name, value := range header {
			w.Header().Set(name, value)
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// attemptTo sends one attempt of a request to target as the proxy does (the
// body read to its end, then closed) and returns its client span.
func attemptTo(t *testing.T, target string, read func(io.Reader)) sdktrace.ReadOnlySpan {
	t.Helper()
	transport := newTransport(nil)
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	ctx := WithAttempts(recorded.ctx)
	AttemptsFrom(ctx).Begin("cred-a")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, strings.NewReader("{}"))
	require.NoError(t, err)
	resp, err := client.Do(req)
	require.NoError(t, err)
	read(resp.Body)
	require.NoError(t, resp.Body.Close())
	recorded.root.End()

	spans := recorded.attempts()
	require.Len(t, spans, 1)
	return spans[0]
}

func readAll(body io.Reader) { _, _ = io.Copy(io.Discard, body) }

// answer is what a span says of the provider's answer.
func answer(span sdktrace.ReadOnlySpan) map[string]any {
	said := map[string]any{}
	for key, value := range ours(span) {
		if key == "aar.upstream_request_id" || strings.HasPrefix(key, "aar.error.") {
			said[key] = value
		}
	}
	return said
}

func TestAttemptTransport_AFailedAttemptSaysWhatTheProviderSaid(t *testing.T) {
	long := strings.Repeat("x", 3*maxErrorBody)
	for name, tc := range map[string]struct {
		status int
		header map[string]string
		body   string
		want   map[string]any
	}{
		"OpenAI": {
			http.StatusTooManyRequests, map[string]string{"X-Request-Id": "req_123"},
			`{"error":{"message":"Rate limit reached for gpt-4o","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`,
			map[string]any{
				"aar.upstream_request_id": "req_123", "aar.error.message": "Rate limit reached for gpt-4o",
				"aar.error.type": "rate_limit_error", "aar.error.code": "rate_limit_exceeded",
			},
		},
		"Anthropic": {
			529, map[string]string{"Request-Id": "req_abc"},
			`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
			map[string]any{"aar.upstream_request_id": "req_abc", "aar.error.message": "Overloaded", "aar.error.type": "overloaded_error"},
		},
		"Google, streaming": {
			http.StatusTooManyRequests, nil,
			`[{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}]`,
			map[string]any{"aar.error.message": "Resource has been exhausted", "aar.error.type": "RESOURCE_EXHAUSTED", "aar.error.code": "429"},
		},
		"AWS": {
			http.StatusTooManyRequests, map[string]string{"X-Amzn-Requestid": "5f1c"},
			`{"message":"Too many requests"}`,
			map[string]any{"aar.upstream_request_id": "5f1c", "aar.error.message": "Too many requests"},
		},
		"Azure": {
			http.StatusUnauthorized, map[string]string{"Apim-Request-Id": "a-1"},
			`{"error":{"code":"401","message":"Access denied"}}`,
			map[string]any{"aar.upstream_request_id": "a-1", "aar.error.message": "Access denied", "aar.error.code": "401"},
		},
		"an error that is a string": {
			http.StatusBadRequest, nil, `{"error":"failed"}`,
			map[string]any{"aar.error.message": "failed"},
		},
		"a detail": {
			http.StatusUnprocessableEntity, nil, `{"detail":"model is required"}`,
			map[string]any{"aar.error.message": "model is required"},
		},
		"not JSON": {
			http.StatusBadGateway, nil, "  <html>Bad Gateway</html>\n",
			map[string]any{"aar.error.message": "<html>Bad Gateway</html>"},
		},
		"JSON of no known shape": {
			http.StatusBadRequest, nil, `{"reason":"no"}`,
			map[string]any{"aar.error.message": `{"reason":"no"}`},
		},
		"longer than is kept": {
			http.StatusInternalServerError, nil, `{"error":{"message":"` + long + `"}}`,
			map[string]any{"aar.error.message": (`{"error":{"message":"` + long)[:maxErrorMessage]},
		},
		"nothing": {
			http.StatusServiceUnavailable, nil, "",
			map[string]any{},
		},
		"an id too long to be one": {
			http.StatusBadRequest, map[string]string{"X-Request-Id": strings.Repeat("i", maxAnswerWord+1)}, `{"error":"failed"}`,
			map[string]any{"aar.error.message": "failed"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			span := attemptTo(t, answering(t, tc.status, tc.header, tc.body).URL, readAll)
			assert.Equal(t, tc.want, answer(span))
			assert.Equal(t, "POST cred-a", span.Name())
		})
	}
}

// What the provider said is on the span when the proxy stops reading before
// the body's end, as it does through a limit: with as much as is kept.
func TestAttemptTransport_AnErrorBodyNotReadToItsEnd(t *testing.T) {
	body := strings.Repeat("e", 4*maxErrorBody)
	span := attemptTo(t, answering(t, http.StatusBadGateway, nil, body).URL, func(r io.Reader) {
		_, _ = io.Copy(io.Discard, io.LimitReader(r, 2*maxErrorBody))
	})
	assert.Equal(t, map[string]any{"aar.error.message": body[:maxErrorMessage]}, answer(span))
}

// An answer that is not a failure has the provider's id and nothing else,
// and its body is the provider's own.
func TestAttemptTransport_AnAnswerThatIsNotAFailure(t *testing.T) {
	srv := answering(t, http.StatusOK, map[string]string{"Request-Id": "req_ok", "X-Request-Id": "req_first"}, `{"error":"not one"}`)
	var got string
	span := attemptTo(t, srv.URL, func(r io.Reader) {
		b, _ := io.ReadAll(r)
		got = string(b)
	})
	assert.Equal(t, map[string]any{"aar.upstream_request_id": "req_first"}, answer(span))
	assert.Equal(t, `{"error":"not one"}`, got)
}

// A body the transport did not decode is left alone.
func TestAttemptTransport_AnEncodedErrorBody(t *testing.T) {
	srv := answering(t, http.StatusBadRequest, map[string]string{"Content-Encoding": "br"}, "\x0b\x02\x80no\x03")
	assert.Empty(t, answer(attemptTo(t, srv.URL, readAll)))
}

// A request that is not an attempt keeps otelhttp's name, and says what was
// answered all the same.
func TestAttemptTransport_TheNameOfARequestThatIsNotAnAttempt(t *testing.T) {
	srv := answering(t, http.StatusNotFound, map[string]string{"X-Request-Id": "req_9"}, `{"error":{"message":"no such price list"}}`)
	transport := newTransport(nil)
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	req, err := http.NewRequestWithContext(recorded.ctx, http.MethodGet, srv.URL, http.NoBody)
	require.NoError(t, err)
	resp, err := client.Do(req)
	require.NoError(t, err)
	readAll(resp.Body)
	require.NoError(t, resp.Body.Close())
	recorded.root.End()

	spans := recorded.attempts()
	require.Len(t, spans, 1)
	assert.Equal(t, "HTTP GET", spans[0].Name())
	assert.Equal(t, map[string]any{"aar.upstream_request_id": "req_9", "aar.error.message": "no such price list"}, answer(spans[0]))
}

// Without a recording span nothing is kept and the body is the provider's.
func TestAttemptTransport_AFailureWithoutTracing(t *testing.T) {
	srv := answering(t, http.StatusTooManyRequests, nil, `{"error":"failed"}`)
	transport := newTransport(nil)
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	ctx := WithAttempts(context.Background())
	AttemptsFrom(ctx).Begin("cred-a")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, strings.NewReader("{}"))
	require.NoError(t, err)
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	_, wrapped := resp.Body.(*errorBody)
	assert.False(t, wrapped)
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.Equal(t, `{"error":"failed"}`, string(body))
}

func TestErrorFields(t *testing.T) {
	for body, want := range map[string][3]string{
		``:                                    {},
		`null`:                                {},
		`"text"`:                              {},
		`[]`:                                  {},
		`{"error":null}`:                      {},
		`{"error":{"message":7,"code":4.5}}`:  {"7", "", "4.5"},
		`{"error":{"type":"t","status":"s"}}`: {"", "t", ""},
		`{"error":{"status":"s"}}`:            {"", "s", ""},
		`{"error":{},"message":"m"}`:          {"m", "", ""},
		`{"message":"m","detail":"d"}`:        {"m", "", ""},
		`{"error":{"message":"m"}`:            {},
	} {
		message, kind, code := errorFields([]byte(body))
		assert.Equal(t, want, [3]string{message, kind, code}, body)
	}
}
