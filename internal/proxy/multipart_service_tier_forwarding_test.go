package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type capturedMultipartRequest struct {
	path          string
	contentType   string
	headers       http.Header
	contentLength int64
	body          []byte
}

func captureMultipartRequest(r *http.Request) (capturedMultipartRequest, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return capturedMultipartRequest{}, err
	}
	return capturedMultipartRequest{
		path:          r.URL.Path,
		contentType:   r.Header.Get("Content-Type"),
		headers:       r.Header.Clone(),
		contentLength: r.ContentLength,
		body:          body,
	}, nil
}

func writeMultipartTestSuccess(w http.ResponseWriter, path string) {
	w.Header().Set("Content-Type", "application/json")
	if path == "/v1/images/edits" {
		_, _ = io.WriteString(w, `{"created":1,"data":[{"b64_json":"aW1hZ2U="}]}`)
		return
	}
	_, _ = io.WriteString(w, `{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
}

func serviceTierMultipartBody(t *testing.T, boundary string) ([]byte, string, []byte) {
	t.Helper()
	image := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 1, 2, 3}
	body, contentType := buildMultipartTestBody(t, boundary,
		multipartTestPart{disposition: `form-data; name="service_tier"`, data: []byte("priority")},
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-4")},
		multipartTestPart{disposition: `form-data; name="prompt"`, data: []byte("test")},
		multipartTestPart{disposition: `form-data; name="extra_body"`, data: []byte(`{"service_tier":"flex","keep":"yes"}`)},
		multipartTestPart{disposition: `form-data; name="image"; filename="input.png"`, contentType: "image/png", data: image},
		multipartTestPart{disposition: `form-data; name="service_tier"`, data: []byte("default")},
	)
	return body, contentType, image
}

func assertForwardedMultipartSanitized(t *testing.T, captured capturedMultipartRequest, wantPath string, wantImage []byte) {
	t.Helper()
	assert.Equal(t, wantPath, captured.path)
	assert.Equal(t, int64(len(captured.body)), captured.contentLength)
	_, parts := parseMultipartTestBody(t, captured.body, captured.contentType)
	names := multipartPartNames(parts)
	assert.NotContains(t, names, "service_tier")
	assert.NotContains(t, names, "extra_body[service_tier]")
	assert.NotContains(t, names, "extra_body.service_tier")
	assert.Contains(t, names, "model")
	assert.Contains(t, names, "prompt")
	assert.Contains(t, names, "image")

	for _, part := range parts {
		switch part.name {
		case "extra_body":
			assert.JSONEq(t, `{"keep":"yes"}`, string(part.data))
		case "image":
			assert.Equal(t, wantImage, part.data)
			assert.Equal(t, "input.png", part.filename)
			assert.Equal(t, "image/png", part.header.Get("Content-Type"))
		}
	}
}

func TestMultipartServiceTierNeverReachesDirectProxyOrAIRUpstream(t *testing.T) {
	tests := []struct {
		name     string
		provider config.ProviderType
		path     string
	}{
		{name: "image edit proxy", provider: config.ProviderTypeProxy, path: "/v1/images/edits"},
		{name: "image edit AIR", provider: config.ProviderTypeAIR, path: "/v1/images/edits"},
		{name: "image edit direct OpenAI", provider: config.ProviderTypeOpenAI, path: "/v1/images/edits"},
		{name: "chat multipart proxy", provider: config.ProviderTypeProxy, path: "/v1/chat/completions"},
		{name: "chat multipart AIR", provider: config.ProviderTypeAIR, path: "/v1/chat/completions"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capturedCh := make(chan capturedMultipartRequest, 1)
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				captured, err := captureMultipartRequest(r)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				capturedCh <- captured
				writeMultipartTestSuccess(w, r.URL.Path)
			}))
			defer upstream.Close()

			proxy := NewTestProxyBuilder().
				WithSingleCredential("upstream", tt.provider, upstream.URL, "upstream-key").
				WithMasterKey("master-key").
				Build()
			body, contentType, image := serviceTierMultipartBody(t, "forward-boundary-"+string(tt.provider))
			req := httptest.NewRequest(http.MethodPost, tt.path, bytes.NewReader(body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", contentType)
			recorder := httptest.NewRecorder()

			proxy.ProxyRequest(recorder, req)

			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			assertForwardedMultipartSanitized(t, <-capturedCh, tt.path, image)
		})
	}
}

func TestMultipartSanitizationUpdatesRequestIntegrityHeaders(t *testing.T) {
	capturedCh := make(chan capturedMultipartRequest, 2)
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, err := captureMultipartRequest(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		capturedCh <- captured
		writeMultipartTestSuccess(w, r.URL.Path)
	}))
	defer upstream.Close()
	proxy := NewTestProxyBuilder().
		WithSingleCredential("proxy", config.ProviderTypeProxy, upstream.URL, "key").
		WithMasterKey("master-key").
		Build()

	changedBody, changedContentType, _ := serviceTierMultipartBody(t, "changed-header-boundary")
	changedReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(changedBody))
	changedReq.Header.Set("Authorization", "Bearer master-key")
	changedReq.Header.Set("Content-Type", changedContentType)
	changedReq.Header.Set("Content-Length", "999999")
	changedReq.Header.Set("Content-MD5", "stale")
	changedReq.Header.Set("Digest", "stale")
	changedReq.Header.Set("Content-Digest", "stale")
	changedReq.Header.Set("Repr-Digest", "stale")
	changedReq.Header.Set("ETag", "stale")
	changedReq.Header.Set("Content-Encoding", "gzip")
	changedReq.ContentLength = 999999
	changedRecorder := httptest.NewRecorder()
	proxy.ProxyRequest(changedRecorder, changedReq)
	require.Equal(t, http.StatusOK, changedRecorder.Code, changedRecorder.Body.String())
	changedCaptured := <-capturedCh
	assert.Equal(t, int64(len(changedCaptured.body)), changedCaptured.contentLength)
	for _, header := range []string{"Content-MD5", "Digest", "Content-Digest", "Repr-Digest", "ETag", "Content-Encoding"} {
		assert.Empty(t, changedCaptured.headers.Get(header), header)
	}

	unchangedBody, unchangedContentType := buildMultipartTestBody(t, "unchanged-header-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-4")},
		multipartTestPart{disposition: `form-data; name="prompt"`, data: []byte("test")},
	)
	unchangedReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(unchangedBody))
	unchangedReq.Header.Set("Authorization", "Bearer master-key")
	unchangedReq.Header.Set("Content-Type", unchangedContentType)
	unchangedReq.Header.Set("Content-MD5", "keep")
	unchangedReq.Header.Set("Digest", "keep")
	unchangedReq.Header.Set("X-Custom", "keep")
	unchangedRecorder := httptest.NewRecorder()
	proxy.ProxyRequest(unchangedRecorder, unchangedReq)
	require.Equal(t, http.StatusOK, unchangedRecorder.Code, unchangedRecorder.Body.String())
	unchangedCaptured := <-capturedCh
	assert.Equal(t, "keep", unchangedCaptured.headers.Get("Content-MD5"))
	assert.Equal(t, "keep", unchangedCaptured.headers.Get("Digest"))
	assert.Equal(t, "keep", unchangedCaptured.headers.Get("X-Custom"))
}

func TestMalformedMultipartStopsBeforeCredentialRetryOrFallback(t *testing.T) {
	var primaryCalls atomic.Int32
	var fallbackCalls atomic.Int32
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer primary.Close()
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		writeMultipartTestSuccess(w, "/v1/images/edits")
	}))
	defer fallback.Close()
	proxy := NewTestProxyBuilder().
		WithPrimaryAndFallback(primary.URL, fallback.URL).
		WithMaxProviderRetries(2).
		WithMasterKey("master-key").
		Build()
	body, contentType, _ := serviceTierMultipartBody(t, "malformed-forward-boundary")
	body = body[:len(body)-10]
	req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", contentType)
	recorder := httptest.NewRecorder()

	proxy.ProxyRequest(recorder, req)

	require.Equal(t, http.StatusBadRequest, recorder.Code)
	var errorBody struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &errorBody))
	assert.Equal(t, "Invalid multipart request body", errorBody.Error.Message)
	assert.Zero(t, primaryCalls.Load())
	assert.Zero(t, fallbackCalls.Load())
}

func TestMultipartRetryAndFallbackReuseSanitizedBody(t *testing.T) {
	t.Run("provider retry", func(t *testing.T) {
		capturedCh := make(chan capturedMultipartRequest, 2)
		first := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			captured, _ := captureMultipartRequest(r)
			capturedCh <- captured
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"rate limited"}}`)
		}))
		defer first.Close()
		second := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			captured, _ := captureMultipartRequest(r)
			capturedCh <- captured
			writeMultipartTestSuccess(w, r.URL.Path)
		}))
		defer second.Close()
		proxy := NewTestProxyBuilder().WithCredentials(
			config.CredentialConfig{Name: "first", Type: config.ProviderTypeProxy, BaseURL: first.URL, APIKey: "first", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "second", Type: config.ProviderTypeProxy, BaseURL: second.URL, APIKey: "second", RPM: 100, TPM: 10000},
		).WithMaxProviderRetries(1).WithMasterKey("master-key").Build()
		body, contentType, image := serviceTierMultipartBody(t, "retry-boundary")
		req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", contentType)
		recorder := httptest.NewRecorder()
		proxy.ProxyRequest(recorder, req)
		require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
		assertForwardedMultipartSanitized(t, <-capturedCh, "/v1/images/edits", image)
		assertForwardedMultipartSanitized(t, <-capturedCh, "/v1/images/edits", image)
	})

	t.Run("fallback", func(t *testing.T) {
		capturedCh := make(chan capturedMultipartRequest, 2)
		primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			captured, _ := captureMultipartRequest(r)
			capturedCh <- captured
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"rate limited"}}`)
		}))
		defer primary.Close()
		fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			captured, _ := captureMultipartRequest(r)
			capturedCh <- captured
			writeMultipartTestSuccess(w, r.URL.Path)
		}))
		defer fallback.Close()
		proxy := NewTestProxyBuilder().
			WithPrimaryAndFallback(primary.URL, fallback.URL).
			WithMasterKey("master-key").
			Build()
		body, contentType, image := serviceTierMultipartBody(t, "fallback-boundary")
		req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", contentType)
		recorder := httptest.NewRecorder()
		proxy.ProxyRequest(recorder, req)
		require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
		assertForwardedMultipartSanitized(t, <-capturedCh, "/v1/images/edits", image)
		assertForwardedMultipartSanitized(t, <-capturedCh, "/v1/images/edits", image)
	})
}
