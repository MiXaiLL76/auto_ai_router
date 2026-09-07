package proxy

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type unexpectedImageTransport struct{ t *testing.T }

func (tr unexpectedImageTransport) RoundTrip(*http.Request) (*http.Response, error) {
	tr.t.Error("Invalid image request reached the provider")
	return nil, http.ErrNotSupported
}

func TestImageValidationBeforeForwarding(t *testing.T) {
	for _, provider := range []config.ProviderType{config.ProviderTypeGemini, config.ProviderTypeVertexAI} {
		for _, mode := range []string{"native", "litellm"} {
			for _, tt := range []struct {
				name, fields, param, code string
			}{
				{"missing prompt", ``, "prompt", "missing_required_parameter"},
				{"empty prompt", `,"prompt":" "`, "prompt", "missing_required_parameter"},
				{"prompt type", `,"prompt":123`, "prompt", "invalid_type"},
				{"size", `,"prompt":"test","size":"invalid-size"`, "size", "invalid_image_size"},
				{"count type", `,"prompt":"test","n":"invalid-count"`, "n", "invalid_type"},
				{"count value", `,"prompt":"test","n":0`, "n", "invalid_value"},
				{"seed", `,"prompt":"test","seed":2147483648`, "seed", "invalid_value"},
			} {
				t.Run(string(provider)+"/"+mode+"/"+tt.name, func(t *testing.T) {
					prx := NewTestProxyBuilder().WithSingleCredential("provider", provider, "http://provider.internal", "key").Build()
					prx.client.Transport = unexpectedImageTransport{t}
					if mode == "litellm" {
						prx.responseCompat = compatlitellm.New()
					}
					req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"gemini-3.1-flash-image"`+tt.fields+`}`))
					req.Header.Set("Authorization", "Bearer master-key")
					req.Header.Set("Content-Type", "application/json")
					w := httptest.NewRecorder()
					prx.ProxyRequest(w, req)
					assertImageClientError(t, w, tt.param, tt.code)
				})
			}
			for _, missing := range []string{"prompt", "image"} {
				t.Run(string(provider)+"/"+mode+"/edit missing "+missing, func(t *testing.T) {
					var body bytes.Buffer
					form := multipart.NewWriter(&body)
					require.NoError(t, form.WriteField("model", "gemini-3.1-flash-image"))
					if missing != "prompt" {
						require.NoError(t, form.WriteField("prompt", "test"))
					}
					if missing != "image" {
						part, err := form.CreateFormFile("image", "test.png")
						require.NoError(t, err)
						_, err = part.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
						require.NoError(t, err)
					}
					require.NoError(t, form.Close())
					prx := NewTestProxyBuilder().WithSingleCredential("provider", provider, "http://provider.internal", "key").Build()
					prx.client.Transport = unexpectedImageTransport{t}
					if mode == "litellm" {
						prx.responseCompat = compatlitellm.New()
					}
					req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &body)
					req.Header.Set("Authorization", "Bearer master-key")
					req.Header.Set("Content-Type", form.FormDataContentType())
					w := httptest.NewRecorder()
					prx.ProxyRequest(w, req)
					assertImageClientError(t, w, missing, "missing_required_parameter")
				})
			}
		}
	}
}

func assertImageClientError(t *testing.T, w *httptest.ResponseRecorder, param, code string) {
	t.Helper()
	require.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	var response APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, "invalid_request_error", response.Error.Type)
	require.NotNil(t, response.Error.Param)
	assert.Equal(t, param, *response.Error.Param)
	require.NotNil(t, response.Error.Code)
	assert.Equal(t, code, *response.Error.Code)
	assert.NotEqual(t, "Invalid request", response.Error.Message)
	assert.NotContains(t, w.Body.String(), "provider.internal")
}

func TestImageUpstreamErrors(t *testing.T) {
	for _, mode := range []string{"native", "litellm"} {
		for _, tt := range []struct {
			body, param, code string
		}{
			{`{"error":{"message":"Invalid type for prompt","param":"prompt","code":"invalid_type"}}`, "prompt", "invalid_type"},
			{`{"error":{"message":"Invalid value for size","param":"size","code":"invalid_value"}}`, "size", "invalid_value"},
			{`{"error":{"message":"Unable to process input image. Request sent to http://air-ru01","status":"INVALID_ARGUMENT"}}`, "image", "invalid_image"},
		} {
			t.Run(mode+"/"+tt.code, func(t *testing.T) {
				prx := NewTestProxyBuilder().WithSingleCredential("gateway", config.ProviderTypeProxy, "http://air-ru01", "key").Build()
				prx.client.Transport = clientErrorTransport{status: http.StatusBadRequest, body: tt.body}
				if mode == "litellm" {
					prx.responseCompat = compatlitellm.New()
				}
				req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"openai/gpt-image-2","prompt":"test"}`))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				prx.ProxyRequest(w, req)
				assertImageClientError(t, w, tt.param, tt.code)
				assert.NotContains(t, w.Body.String(), "air-ru01")
			})
		}
	}
}
