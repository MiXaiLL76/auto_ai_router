package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
			Type:    config.ProviderTypeOpenAI,
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
		name        string
		contentType string
		body        string
	}{
		{name: "non-JSON body", contentType: "text/html", body: "<html>not an image response</html>"},
		{name: "empty body", contentType: "application/json", body: ""},
		{name: "empty data array", contentType: "application/json", body: `{"created":1,"data":[]}`},
	}

	for _, path := range []string{"/v1/images/generations", "/v1/images/edits"} {
		for _, tt := range tests {
			t.Run(path+" "+tt.name, func(t *testing.T) {
				prx, dbStub := newImageBillingProxy(t, http.StatusOK, tt.contentType, tt.body)

				sendImageRequest(t, prx, path, `{"model":"image-model","prompt":"a cat","n":3}`)

				require.Len(t, dbStub.loggedEntries, 1)
				assert.Zero(t, dbStub.loggedEntries[0].Spend)
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
