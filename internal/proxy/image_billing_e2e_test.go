package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testImagePrice = 0.25

// newImageBillingProxy wires a proxy to a fake OpenAI-compatible image upstream
// that answers every request with the given status, content type and body.
func newImageBillingProxy(t *testing.T, status int, contentType, body string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	return newImageBillingProxyForCredential(t, config.ProviderTypeOpenAI, status, contentType, body)
}

// newImageBillingProxyForCredential is newImageBillingProxy reaching the
// upstream through a credential of the given type.
func newImageBillingProxyForCredential(t *testing.T, credentialType config.ProviderType, status int, contentType, body string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(upstream.Close)

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "image-upstream",
			Type:    credentialType,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"image-model": {OutputCostPerImage: testImagePrice},
	})
	prx.priceRegistry = registry
	return prx, dbStub
}

func sendImageRequest(t *testing.T, prx *Proxy, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

func TestProxyRequest_ImageBillingCountsDeliveredImages(t *testing.T) {
	tests := []struct {
		name         string
		requestBody  string
		upstreamBody string
		wantImages   int
	}{
		{
			name:         "provider ignores n and returns a single image",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":3}`,
			upstreamBody: `{"created":1,"data":[{"url":"https://img.example/1.png"}]}`,
			wantImages:   1,
		},
		{
			name:         "provider returns a batch larger than n",
			requestBody:  `{"model":"image-model","prompt":"a cat, day and night"}`,
			upstreamBody: `{"created":1,"data":[{"url":"https://img.example/1.png"},{"url":"https://img.example/2.png"},{"url":"https://img.example/3.png"}],"usage":{"output_tokens":48000,"total_tokens":48000}}`,
			wantImages:   3,
		},
		{
			name:         "generated_images agreeing with data",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":1}`,
			upstreamBody: `{"created":1,"data":[{"url":"https://img.example/1.png"},{"url":"https://img.example/2.png"}],"usage":{"generated_images":2,"output_tokens":32768,"total_tokens":32768}}`,
			wantImages:   2,
		},
		{
			name:         "delivered images win over a larger generated_images",
			requestBody:  `{"model":"image-model","prompt":"a cat"}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.png"},{"error":{"code":"filtered","message":"x"}}],"usage":{"generated_images":2}}`,
			wantImages:   1,
		},
		{
			name:         "generated_images used when data entries are unrecognized",
			requestBody:  `{"model":"image-model","prompt":"a cat"}`,
			upstreamBody: `{"data":[{"image":"https://img.example/1.png"},{"image":"https://img.example/2.png"},{"image":"https://img.example/3.png"}],"usage":{"generated_images":3}}`,
			wantImages:   3,
		},
		{
			name:         "failed entries in data are not billed",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":2}`,
			upstreamBody: `{"created":1,"data":[{"b64_json":"aGVsbG8="},{"error":{"code":"OutputImageSensitiveContentDetected","message":"filtered"}}]}`,
			wantImages:   1,
		},
		{
			name:         "base64 images are counted",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":2,"response_format":"b64_json"}`,
			upstreamBody: `{"created":1,"data":[{"b64_json":"aGVsbG8="},{"b64_json":"d29ybGQ="}]}`,
			wantImages:   2,
		},
		{
			name:         "unknown response shape keeps the requested count",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":2}`,
			upstreamBody: `{"created":1,"images":["https://img.example/1.png","https://img.example/2.png"]}`,
			wantImages:   2,
		},
		{
			name:         "an unrecognized entry does not discard the recognized images",
			requestBody:  `{"model":"image-model","prompt":"a cat"}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.png"},{"b64_json":"aGVsbG8="},{"image":"https://img.example/3.png"}]}`,
			wantImages:   2,
		},
		{
			name:         "requested count is used within what data can hold",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":5}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.png"},{"image":"https://img.example/2.png"}]}`,
			wantImages:   2,
		},
		{
			name:         "generated_images is used within what data can hold",
			requestBody:  `{"model":"image-model","prompt":"a cat"}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.png"},{"image":"https://img.example/2.png"},{"image":"https://img.example/3.png"}],"usage":{"generated_images":2}}`,
			wantImages:   2,
		},
		{
			name:         "empty entries are not images",
			requestBody:  `{"model":"image-model","prompt":"a cat","n":3}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.png"},null,{},{"url":null}]}`,
			wantImages:   1,
		},
		{
			name:         "generated_images is used when data is not an array",
			requestBody:  `{"model":"image-model","prompt":"a cat"}`,
			upstreamBody: `{"data":{"url":"https://img.example/1.png"},"usage":{"generated_images":2}}`,
			wantImages:   2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx, dbStub := newImageBillingProxy(t, http.StatusOK, "application/json", tt.upstreamBody)

			w := sendImageRequest(t, prx, "/v1/images/generations", tt.requestBody)

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			assert.InDelta(t, float64(tt.wantImages)*testImagePrice, dbStub.loggedEntries[0].Spend, 1e-9)
		})
	}
}

func TestProxyRequest_ImageBillingDoesNotChargeUndeliveredResponse(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "empty data array", body: `{"created":1,"data":[]}`},
		{name: "only failed entries", body: `{"data":[{"error":{"code":"OutputImageSensitiveContentDetected","message":"filtered"}}],"usage":{"generated_images":0}}`},
		{name: "zero generated_images without data", body: `{"created":1,"usage":{"generated_images":0}}`},
	}

	for _, path := range []string{"/v1/images/generations", "/v1/images/edits"} {
		for _, tt := range tests {
			t.Run(path+" "+tt.name, func(t *testing.T) {
				prx, dbStub := newImageBillingProxy(t, http.StatusOK, "application/json", tt.body)

				sendImageRequest(t, prx, path, `{"model":"image-model","prompt":"a cat","n":3}`)

				require.Len(t, dbStub.loggedEntries, 1)
				assert.Zero(t, dbStub.loggedEntries[0].Spend)
			})
		}
	}
}

// A successful response the router can't read says nothing about how many
// images were delivered — the client may still have received them — so it is
// billed by the request, not as zero.
func TestProxyRequest_ImageBillingUnreadableResponseKeepsRequestedCount(t *testing.T) {
	tests := []struct {
		name        string
		contentType string
		body        string
	}{
		{name: "non-JSON body", contentType: "text/html", body: "<html>not an image response</html>"},
		{name: "raw image bytes", contentType: "image/png", body: "\x89PNG\r\n\x1a\n"},
		{name: "empty body", contentType: "application/json", body: ""},
		{name: "JSON that is not an object", contentType: "application/json", body: `[{"url":"https://img.example/1.png"}]`},
		{name: "JSON null", contentType: "application/json", body: `null`},
	}

	for _, path := range []string{"/v1/images/generations", "/v1/images/edits"} {
		for _, tt := range tests {
			t.Run(path+" "+tt.name, func(t *testing.T) {
				prx, dbStub := newImageBillingProxy(t, http.StatusOK, tt.contentType, tt.body)

				sendImageRequest(t, prx, path, `{"model":"image-model","prompt":"a cat","n":3}`)

				require.Len(t, dbStub.loggedEntries, 1)
				assert.InDelta(t, 3*testImagePrice, dbStub.loggedEntries[0].Spend, 1e-9)
			})
		}
	}
}

func TestProxyRequest_ImageStreamBillingUsesGeneratedImages(t *testing.T) {
	stream := strings.Join([]string{
		"event: image_generation.partial_succeeded",
		`data: {"type":"image_generation.partial_succeeded","image_index":0,"url":"https://img.example/1.png"}`,
		"",
		"event: image_generation.partial_succeeded",
		`data: {"type":"image_generation.partial_succeeded","image_index":1,"url":"https://img.example/2.png"}`,
		"",
		"event: image_generation.completed",
		`data: {"type":"image_generation.completed","usage":{"generated_images":2,"output_tokens":32768,"total_tokens":32768}}`,
		"",
		"data: [DONE]",
		"",
		"",
	}, "\n")
	prx, dbStub := newImageBillingProxy(t, http.StatusOK, "text/event-stream", stream)

	w := sendImageRequest(t, prx, "/v1/images/generations", `{"model":"image-model","prompt":"a cat, day and night","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, 2*testImagePrice, dbStub.loggedEntries[0].Spend, 1e-9)
}

// Direct provider credentials and proxy (AIR hop) credentials read the
// response on different code paths; both must bill a non-streaming image
// response the same way.
func TestProxyRequest_ImageBillingSameForDirectAndProxyCredentials(t *testing.T) {
	tests := []struct {
		name         string
		contentType  string
		upstreamBody string
		wantImages   int
	}{
		{name: "delivered images", contentType: "application/json", upstreamBody: `{"data":[{"url":"https://img.example/1.png"},{"b64_json":"aGVsbG8="}]}`, wantImages: 2},
		{name: "nothing delivered", contentType: "application/json", upstreamBody: `{"data":[]}`, wantImages: 0},
		{name: "unrecognized entry within n", contentType: "application/json", upstreamBody: `{"data":[{"url":"https://img.example/1.png"},{"image":"https://img.example/2.png"}]}`, wantImages: 2},
		{name: "unreadable body", contentType: "text/html", upstreamBody: `<html>oops</html>`, wantImages: 3},
	}
	for _, credentialType := range []config.ProviderType{config.ProviderTypeOpenAI, config.ProviderTypeProxy} {
		for _, tt := range tests {
			t.Run(string(credentialType)+" "+tt.name, func(t *testing.T) {
				prx, dbStub := newImageBillingProxyForCredential(t, credentialType, http.StatusOK, tt.contentType, tt.upstreamBody)

				w := sendImageRequest(t, prx, "/v1/images/generations", `{"model":"image-model","prompt":"a cat","n":3}`)

				require.Equal(t, http.StatusOK, w.Code)
				require.Len(t, dbStub.loggedEntries, 1)
				assert.InDelta(t, float64(tt.wantImages)*testImagePrice, dbStub.loggedEntries[0].Spend, 1e-9)
			})
		}
	}
}

// A failed request bills no images, even when the provider reported some.
func TestProxyRequest_ImageBillingSkipsFailedRequests(t *testing.T) {
	prx, dbStub := newImageBillingProxy(t, http.StatusBadRequest, "application/json", `{"error":{"message":"bad size"},"data":[{"url":"https://img.example/1.png"}]}`)

	sendImageRequest(t, prx, "/v1/images/generations", `{"model":"image-model","prompt":"a cat","n":2}`)

	require.Len(t, dbStub.loggedEntries, 1)
	assert.Zero(t, dbStub.loggedEntries[0].Spend)
}

// The usage event is not necessarily the last data frame of a direct provider
// stream: a trailing event without usage must not discard the reported count.
func TestProxyRequest_ImageStreamBillingSurvivesTrailingEvents(t *testing.T) {
	stream := strings.Join([]string{
		"event: image_generation.partial_succeeded",
		`data: {"type":"image_generation.partial_succeeded","image_index":0,"url":"https://img.example/1.png"}`,
		"",
		"event: image_generation.partial_succeeded",
		`data: {"type":"image_generation.partial_succeeded","image_index":1,"url":"https://img.example/2.png"}`,
		"",
		"event: image_generation.completed",
		`data: {"type":"image_generation.completed","usage":{"generated_images":2,"output_tokens":32768,"total_tokens":32768}}`,
		"",
		"event: image_generation.done",
		`data: {"type":"image_generation.done","created":1}`,
		"",
		"data: [DONE]",
		"",
		"",
	}, "\n")
	prx, dbStub := newImageBillingProxy(t, http.StatusOK, "text/event-stream", stream)

	w := sendImageRequest(t, prx, "/v1/images/generations", `{"model":"image-model","prompt":"a cat, day and night","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, 2*testImagePrice, dbStub.loggedEntries[0].Spend, 1e-9)
}

func TestProxyRequest_ImageStreamBillingWithoutCountKeepsRequestedCount(t *testing.T) {
	stream := strings.Join([]string{
		"event: image_generation.completed",
		`data: {"type":"image_generation.completed","b64_json":"aGVsbG8="}`,
		"",
		"data: [DONE]",
		"",
		"",
	}, "\n")
	prx, dbStub := newImageBillingProxy(t, http.StatusOK, "text/event-stream", stream)

	sendImageRequest(t, prx, "/v1/images/generations", `{"model":"image-model","prompt":"a cat","stream":true}`)

	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, testImagePrice, dbStub.loggedEntries[0].Spend, 1e-9)
}

func TestProxyRequest_ImageWatermarkDisabledForWatermarkingModels(t *testing.T) {
	for _, tt := range []struct {
		model         string
		wantWatermark string
	}{
		{model: "seedream-4-5-251128", wantWatermark: "false"},
		{model: "dola-seedream-5-0-pro-260628", wantWatermark: "false"},
		{model: "grok-imagine-image", wantWatermark: ""},
	} {
		for _, stream := range []bool{false, true} {
			name := tt.model
			if stream {
				name += " stream"
			}
			t.Run(name, func(t *testing.T) {
				var upstreamBody map[string]any
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					require.NoError(t, json.NewDecoder(r.Body).Decode(&upstreamBody))
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"created":1,"data":[{"url":"https://img.example/1.png"}]}`))
				}))
				defer upstream.Close()

				prx := NewTestProxyBuilder().
					WithCredentials(config.CredentialConfig{
						Name:    "image-upstream",
						Type:    config.ProviderTypeOpenAI,
						BaseURL: upstream.URL,
						APIKey:  "upstream-key",
						RPM:     100,
						TPM:     10000,
					}).
					WithMasterKey("master-key").
					Build()

				body := `{"model":"` + tt.model + `","prompt":"a cat","watermark":true}`
				if tt.wantWatermark == "" {
					body = `{"model":"` + tt.model + `","prompt":"a cat"}`
				}
				if stream {
					body = strings.TrimSuffix(body, "}") + `,"stream":true}`
				}
				w := sendImageRequest(t, prx, "/v1/images/generations", body)

				require.Equal(t, http.StatusOK, w.Code)
				require.NotNil(t, upstreamBody)
				watermark, present := upstreamBody["watermark"]
				if tt.wantWatermark == "" {
					assert.False(t, present, "watermark must not be added for %s", tt.model)
					return
				}
				assert.Equal(t, false, watermark)
				assert.Equal(t, "a cat", upstreamBody["prompt"])
			})
		}
	}
}

// /v1/images/edits is usually sent as multipart/form-data; the watermark
// opt-out must reach the provider there too, exactly once.
func TestProxyRequest_ImageEditMultipartDisablesWatermark(t *testing.T) {
	for _, tt := range []struct {
		model         string
		wantWatermark []string
	}{
		{model: "seededit-3-0-i2i-250628", wantWatermark: []string{"false"}},
		{model: "seedream-4-5-251128", wantWatermark: []string{"false"}},
		{model: "gpt-image-1", wantWatermark: []string{"true"}},
	} {
		t.Run(tt.model, func(t *testing.T) {
			var received url.Values
			var receivedImage []byte
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if assert.NoError(t, r.ParseMultipartForm(1<<20)) {
					received = r.MultipartForm.Value
					if file, _, err := r.FormFile("image"); assert.NoError(t, err) {
						receivedImage, _ = io.ReadAll(file)
						_ = file.Close()
					}
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"created":1,"data":[{"url":"https://img.example/1.png"}]}`))
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{
					Name:    "image-upstream",
					Type:    config.ProviderTypeOpenAI,
					BaseURL: upstream.URL,
					APIKey:  "upstream-key",
					RPM:     100,
					TPM:     10000,
				}).
				WithMasterKey("master-key").
				Build()

			imageData := []byte("\x89PNG\r\n\x1a\n")
			var buf bytes.Buffer
			form := multipart.NewWriter(&buf)
			require.NoError(t, form.WriteField("model", tt.model))
			require.NoError(t, form.WriteField("prompt", "make it blue"))
			require.NoError(t, form.WriteField("watermark", "true"))
			part, err := form.CreateFormFile("image", "input.png")
			require.NoError(t, err)
			_, err = part.Write(imageData)
			require.NoError(t, err)
			require.NoError(t, form.Close())

			req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &buf)
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", form.FormDataContentType())
			w := httptest.NewRecorder()
			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			require.NotNil(t, received)
			assert.Equal(t, tt.wantWatermark, received["watermark"])
			assert.Equal(t, []string{"make it blue"}, received["prompt"])
			assert.Equal(t, imageData, receivedImage)
		})
	}
}
