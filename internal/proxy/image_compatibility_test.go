package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/require"
)

func TestImageMiniForwardedCompatibility(t *testing.T) {
	for _, model := range []string{"gpt-image-1-mini", "gpt-image-1", "gpt-image-2"} {
		for _, format := range []string{"generation", "JSON edit", "multipart edit"} {
			for _, quality := range []string{"standard", "hd", "low", "unknown"} {
				t.Run(model+"/"+format+"/"+quality, func(t *testing.T) {
					wantQuality := quality
					if model == "gpt-image-1-mini" {
						if quality == "standard" {
							wantQuality = "medium"
						}
						if quality == "hd" {
							wantQuality = "high"
						}
					}
					called := false
					upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						called = true
						var fields map[string]any
						if format == "multipart edit" {
							require.NoError(t, r.ParseMultipartForm(1<<20))
							defer func() { require.NoError(t, r.MultipartForm.RemoveAll()) }()
							fields = map[string]any{}
							for k, v := range r.MultipartForm.Value {
								fields[k] = v[0]
							}
							for _, name := range []string{"image", "mask"} {
								file, _, err := r.FormFile(name)
								require.NoError(t, err)
								data, err := io.ReadAll(file)
								require.NoError(t, err)
								require.NoError(t, file.Close())
								require.Equal(t, name+" bytes", string(data))
							}
						} else {
							require.NoError(t, json.NewDecoder(r.Body).Decode(&fields))
							require.Equal(t, "image bytes", fields["image"])
							require.Equal(t, "mask bytes", fields["mask"])
						}
						require.Equal(t, model, fields["model"])
						require.Equal(t, wantQuality, fields["quality"])
						require.Equal(t, "unchanged", fields["user"])
						if model == "gpt-image-1-mini" && format != "generation" {
							require.NotContains(t, fields, "input_fidelity")
						} else {
							require.Equal(t, "high", fields["input_fidelity"])
						}
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, `{"data":[{"b64_json":"aW1n"}]}`)
					}))
					defer upstream.Close()
					prx := NewTestProxyBuilder().WithSingleCredential("provider", config.ProviderTypeOpenAI, upstream.URL, "key").Build()
					manager := models.New(testhelpers.NewTestLogger(), 50, []config.ModelRPMConfig{{Name: "public-image", Model: model, RPM: 100, TPM: -1}})
					manager.LoadModelsFromConfig([]config.CredentialConfig{{Name: "provider", Type: config.ProviderTypeOpenAI}})
					prx.modelManager = manager
					prx.balancer.SetModelChecker(manager)
					fields := map[string]string{"model": "public-image", "prompt": "test", "quality": quality, "input_fidelity": "high", "user": "unchanged", "image": "image bytes", "mask": "mask bytes"}
					var body bytes.Buffer
					contentType := "application/json"
					if format == "multipart edit" {
						form := multipart.NewWriter(&body)
						for k, v := range fields {
							if k == "image" || k == "mask" {
								part, err := form.CreateFormFile(k, k+".png")
								require.NoError(t, err)
								_, err = io.WriteString(part, v)
								require.NoError(t, err)
							} else {
								require.NoError(t, form.WriteField(k, v))
							}
						}
						require.NoError(t, form.Close())
						contentType = form.FormDataContentType()
					} else {
						require.NoError(t, json.NewEncoder(&body).Encode(fields))
					}
					path := "/v1/images/edits"
					if format == "generation" {
						path = "/v1/images/generations"
					}
					req := httptest.NewRequest(http.MethodPost, path, &body)
					req.Header.Set("Content-Type", contentType)
					req.Header.Set("Authorization", "Bearer master-key")
					w := httptest.NewRecorder()
					prx.ProxyRequest(w, req)
					require.Equal(t, 200, w.Code, w.Body.String())
					require.True(t, called)
				})
			}
		}
	}
}

func TestGeminiJSONEditForwardedPayload(t *testing.T) {
	for _, provider := range []config.ProviderType{config.ProviderTypeGemini} {
		t.Run(string(provider), func(t *testing.T) {
			called := false
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
				var payload struct {
					Contents []struct {
						Parts []struct {
							Text       string
							InlineData *struct {
								MimeType string
								Data     string
							}
						}
					}
					GenerationConfig map[string]any
				}
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				require.Len(t, payload.Contents, 1)
				parts := payload.Contents[0].Parts
				require.Len(t, parts, 5)
				require.Equal(t, "test", parts[0].Text)
				for _, i := range []int{1, 2, 4} {
					require.NotNil(t, parts[i].InlineData)
					require.Equal(t, "image/png", parts[i].InlineData.MimeType)
					require.Equal(t, "aW1n", parts[i].InlineData.Data)
				}
				require.Equal(t, "Use the provided mask image to constrain the edit.", parts[3].Text)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"aW1n"}}]}}]}`)
			}))
			defer upstream.Close()
			prx := NewTestProxyBuilder().WithSingleCredential("provider", provider, upstream.URL, "key").Build()
			manager := models.New(testhelpers.NewTestLogger(), 50, []config.ModelRPMConfig{{Name: "public-image", Model: "gemini-3.1-flash-image", RPM: 100, TPM: -1}})
			manager.LoadModelsFromConfig([]config.CredentialConfig{{Name: "provider", Type: provider}})
			prx.modelManager = manager
			prx.balancer.SetModelChecker(manager)
			req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", strings.NewReader(`{"model":"public-image","prompt":"test","images":[{"image_url":"data:image/png;base64,aW1n"},"data:image/png;base64,aW1n"],"mask":{"url":"data:image/png;base64,aW1n"},"size":"1024x1024"}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json; charset=utf-8")
			w := httptest.NewRecorder()
			prx.ProxyRequest(w, req)
			require.Equal(t, 200, w.Code, w.Body.String())
			require.True(t, called)
		})
	}
}

func TestGeminiJSONEditPreservesProviderFailure(t *testing.T) {
	prx := NewTestProxyBuilder().WithSingleCredential("provider", config.ProviderTypeGemini, "http://provider.internal", "key").Build()
	prx.client.Transport = clientErrorTransport{status: http.StatusInternalServerError, body: `{"error":{"message":"Provider failure"}}`}
	req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", strings.NewReader(`{"model":"gemini-3.1-flash-image","prompt":"test","image":"data:image/png;base64,aW1n"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusInternalServerError, w.Code)
}
