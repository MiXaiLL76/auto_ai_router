package requestid

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
)

func TestMiddlewareSetsHeaderAndContext(t *testing.T) {
	var gotFromContext string
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotFromContext = FromContext(r.Context())
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	Middleware(false)(next).ServeHTTP(rec, req)

	headerID := rec.Header().Get(Header)
	require.NotEmpty(t, headerID)
	tid, err := trace.TraceIDFromHex(headerID)
	require.NoError(t, err, "X-Request-Id must be a valid 32-hex trace ID")
	assert.True(t, tid.IsValid())
	assert.Equal(t, headerID, gotFromContext, "context value must match the response header")
}

func TestMiddlewareAssignsDistinctIDsPerRequest(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	handler := Middleware(false)(next)

	rec1 := httptest.NewRecorder()
	handler.ServeHTTP(rec1, httptest.NewRequest(http.MethodGet, "/", nil))
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, httptest.NewRequest(http.MethodGet, "/", nil))

	assert.NotEqual(t, rec1.Header().Get(Header), rec2.Header().Get(Header))
}

func TestMiddlewareHonorsTrustedIncomingTraceparent(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")

	rec := httptest.NewRecorder()
	Middleware(true)(next).ServeHTTP(rec, req)

	assert.Equal(t, "4bf92f3577b34da6a3ce929d0e0e4736", rec.Header().Get(Header))
}

func TestMiddlewareIgnoresIncomingTraceparentWhenNotTrusted(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")

	rec := httptest.NewRecorder()
	Middleware(false)(next).ServeHTTP(rec, req)

	assert.NotEqual(t, "4bf92f3577b34da6a3ce929d0e0e4736", rec.Header().Get(Header))
}

func TestCanonical(t *testing.T) {
	assert.Equal(t, "4bf92f3577b34da6a3ce929d0e0e4736", Canonical("4bf92f3577b34da6a3ce929d0e0e4736"))
	assert.Empty(t, Canonical(""))
	assert.Empty(t, Canonical("4BF92F3577B34DA6A3CE929D0E0E4736"), "uppercase is not canonical")
	assert.Empty(t, Canonical("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"), "non-hex is rejected")
	assert.Empty(t, Canonical("00000000000000000000000000000000"), "all-zero trace ID is invalid")
	assert.Empty(t, Canonical("4bf92f3577b34da6"), "short input is rejected")
}

func TestFromContextEmptyWhenUnset(t *testing.T) {
	assert.Empty(t, FromContext(context.TODO()))
	assert.Empty(t, FromContext(httptest.NewRequest(http.MethodGet, "/", nil).Context()))
}
