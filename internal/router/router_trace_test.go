package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestServeHTTP_Trace_Route verifies that GET /trace is routed to the trace handler
// and returns HTTP 200 with application/json content type.
func TestServeHTTP_Trace_Route(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/trace", nil)
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code, "GET /trace must return 200")
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"),
		"GET /trace must return application/json")
}

// TestServeHTTP_Trace_ValidJSON verifies that GET /trace returns a valid
// ProxyTraceResponse JSON body (not 404 and not empty).
func TestServeHTTP_Trace_ValidJSON(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/trace", nil)
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.NotEmpty(t, w.Body.Bytes())

	var resp httputil.ProxyTraceResponse
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err, "GET /trace body must be valid ProxyTraceResponse JSON")
	assert.NotEmpty(t, resp.Status, "trace response must have a non-empty status")
}

func TestServeHTTP_Trace_UnverifiableTokenFallsBackToPublic(t *testing.T) {
	prx := createProxyWithConfig([]config.CredentialConfig{
		{Name: "team-a", APIKey: "key1", BaseURL: "http://team-a.example", RPM: 100, Scopes: []string{"team-a"}},
	}, nil)
	prx.LiteLLMDB = unavailableScopeDB{NoopManager: litellmdb.NewNoopManager()}
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/trace", nil)
	req.Header.Set("Authorization", "Bearer stale-key")
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var resp httputil.ProxyTraceResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "unhealthy", resp.Status)
	assert.Empty(t, resp.Credentials)
}

// TestServeHTTP_Trace_DepthQueryParam verifies that the depth query parameter is
// accepted without error (router must not return 404 or 400).
func TestServeHTTP_Trace_DepthQueryParam(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	for _, depth := range []string{"0", "1", "10"} {
		t.Run("depth="+depth, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/trace?depth="+depth, nil)
			w := httptest.NewRecorder()

			r.ServeHTTP(w, req)

			assert.Equal(t, http.StatusOK, w.Code,
				"GET /trace?depth=%s must return 200", depth)
		})
	}
}

// TestServeHTTP_VisualTrace_Route verifies that GET /vtrace is routed to the visual
// trace handler and returns HTTP 200 with text/html content type.
func TestServeHTTP_VisualTrace_Route(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/vtrace", nil)
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code, "GET /vtrace must return 200")
	assert.Contains(t, w.Header().Get("Content-Type"), "text/html",
		"GET /vtrace must return text/html")
	body := w.Body.String()
	assert.NotEmpty(t, body, "GET /vtrace must return a non-empty HTML body")
	assert.Contains(t, body, `id="diagramViewport"`,
		"visual trace must include a pannable diagram viewport")
	assert.Contains(t, body, `id="diagramZoomIn"`,
		"visual trace must include zoom controls")
	assert.Contains(t, body, `id="diagramReset"`,
		"visual trace must include a view reset control")
}

// TestServeHTTP_Trace_NotFound_OtherPaths verifies that /trace-related paths that
// were not explicitly registered still return 404 (no accidental broad prefix match).
func TestServeHTTP_Trace_NotFound_OtherPaths(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	paths := []string{"/trace/extra", "/vtrace/extra", "/traces"}

	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest("GET", path, nil)
			w := httptest.NewRecorder()

			r.ServeHTTP(w, req)

			assert.Equal(t, http.StatusNotFound, w.Code,
				"path %q should return 404", path)
		})
	}
}

// TestHandleTrace_DirectHandler verifies that calling r.handleTrace directly
// returns a valid JSON trace response (unit-tests the thin router handler).
func TestHandleTrace_DirectHandler(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/trace", nil)
	w := httptest.NewRecorder()

	r.handleTrace(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var resp httputil.ProxyTraceResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.NotEmpty(t, resp.Status)
}

// TestHandleVisualTrace_DirectHandler verifies that calling r.handleVisualTrace directly
// returns an HTML response (unit-tests the thin router handler).
func TestHandleVisualTrace_DirectHandler(t *testing.T) {
	prx := createTestProxy()
	r := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/vtrace", nil)
	w := httptest.NewRecorder()

	r.handleVisualTrace(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Header().Get("Content-Type"), "text/html")
	assert.NotEmpty(t, w.Body.String())
}
