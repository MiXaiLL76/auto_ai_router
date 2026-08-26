package requestid

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMiddlewareSetsHeaderAndContext(t *testing.T) {
	var gotFromContext string
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotFromContext = FromContext(r.Context())
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	Middleware(next).ServeHTTP(rec, req)

	headerID := rec.Header().Get(Header)
	require.NotEmpty(t, headerID)
	_, err := uuid.Parse(headerID)
	require.NoError(t, err, "X-Request-Id must be a valid UUID")
	assert.Equal(t, headerID, gotFromContext, "context value must match the response header")
}

func TestMiddlewareAssignsDistinctIDsPerRequest(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	handler := Middleware(next)

	rec1 := httptest.NewRecorder()
	handler.ServeHTTP(rec1, httptest.NewRequest(http.MethodGet, "/", nil))
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, httptest.NewRequest(http.MethodGet, "/", nil))

	assert.NotEqual(t, rec1.Header().Get(Header), rec2.Header().Get(Header))
}

func TestFromContextEmptyWhenUnset(t *testing.T) {
	assert.Empty(t, FromContext(context.TODO()))
	assert.Empty(t, FromContext(httptest.NewRequest(http.MethodGet, "/", nil).Context()))
}
