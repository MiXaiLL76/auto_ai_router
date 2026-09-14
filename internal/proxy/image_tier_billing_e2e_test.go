package proxy

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// tieredImagePrices mirrors how a price map publishes per-image tiers
// (provider list price × 1.3).
const tieredImagePrices = `{
	"pixel-model": {
		"output_cost_per_image": 0.117,
		"input_cost_per_image": 0.0039,
		"input_images_free_per_request": 1,
		"output_cost_per_image_tiers": [
			{"when": {"layer_decomposition": true}, "max_pixels": 2610000, "output_cost_per_image": 0.02925},
			{"when": {"layer_decomposition": true}, "output_cost_per_image": 0.0585},
			{"max_pixels": 2610000, "output_cost_per_image": 0.0585}
		]
	},
	"quality-model": {
		"output_cost_per_image": 0.104,
		"input_cost_per_image": 0.013,
		"image_request_defaults": {"resolution": "1k", "quality": "auto"},
		"output_cost_per_image_tiers": [
			{"operation": "edit", "when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.078},
			{"operation": "edit", "when": {"resolution": "2k", "quality": "auto"}, "output_cost_per_image": 0.104},
			{"when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.052},
			{"when": {"resolution": "2k", "quality": "auto"}, "output_cost_per_image": 0.078},
			{"when": {"resolution": "1k", "quality": "low"}, "output_cost_per_image": 0.052},
			{"when": {"resolution": "2k", "quality": "low"}, "output_cost_per_image": 0.078},
			{"when": {"resolution": "1k", "quality": "medium"}, "output_cost_per_image": 0.078},
			{"when": {"resolution": "2k", "quality": "medium"}, "output_cost_per_image": 0.104}
		]
	}
}`

func newTieredImageProxy(t *testing.T, contentType, upstreamBody string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	prx, dbStub := newImageBillingProxy(t, http.StatusOK, contentType, upstreamBody)
	var prices map[string]*pricing.ModelPrice
	require.NoError(t, json.Unmarshal([]byte(tieredImagePrices), &prices))
	registry := pricing.NewModelPriceRegistry()
	registry.Update(prices)
	prx.priceRegistry = registry
	return prx, dbStub
}

func TestProxyRequest_ImageTieredBilling(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		requestBody  string
		upstreamType string
		upstreamBody string
		want         float64
	}{
		{
			name:         "pixel tiers: default size above threshold",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"pixel-model","prompt":"a cat"}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg","size":"2496x1664"}],"usage":{"generated_images":1,"input_images":0,"output_tokens":16224,"total_tokens":16224}}`,
			want:         0.117,
		},
		{
			name:         "pixel tiers: small output and two reference images",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"pixel-model","prompt":"merge","size":"1K","image":["https://img.example/a.png","https://img.example/b.png"]}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg","size":"1424x800"}],"usage":{"generated_images":1,"input_images":2,"output_tokens":4450,"total_tokens":4450}}`,
			want:         0.0585 + 0.0039,
		},
		{
			name:         "pixel tiers: layer decomposition",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"pixel-model","image":"https://img.example/a.png","size":"2K","layer_decomposition":true}`,
			upstreamBody: `{"data":[{"url":"https://img.example/0.jpeg","size":"2048x2048","z_index":0},{"url":"https://img.example/1.png","size":"1273x265","z_index":1},{"url":"https://img.example/2.png","size":"492x98","z_index":2}],"usage":{"generated_images":3,"input_images":1,"output_tokens":23107,"total_tokens":23107}}`,
			want:         0.0585 + 2*0.02925,
		},
		{
			name:         "pixel tiers: stream has no sizes, so the fallback price applies",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"pixel-model","prompt":"a cat","stream":true}`,
			upstreamType: "text/event-stream",
			upstreamBody: "event: image_generation.completed\ndata: {\"type\":\"image_generation.completed\",\"usage\":{\"generated_images\":2,\"input_images\":3}}\n\ndata: [DONE]\n\n",
			want:         2*0.117 + 2*0.0039,
		},
		{
			name:         "quality tiers: explicit 2K medium, parameters are case-insensitive",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"quality-model","prompt":"a cat","resolution":"2K","quality":"Medium"}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg","mime_type":"image/jpeg"}],"usage":{"cost_in_usd_ticks":800000000}}`,
			want:         0.104,
		},
		{
			name:         "quality tiers: generation ignores an image field for input billing",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"quality-model","prompt":"a cat","image":{"url":"https://img.example/a.png","type":"image_url"}}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg"}]}`,
			want:         0.052,
		},
		{
			name:         "quality tiers: edit with one image uses the edit default quality",
			path:         "/v1/images/edits",
			requestBody:  `{"model":"quality-model","prompt":"add a scarf","image":{"url":"https://img.example/a.png","type":"image_url"}}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg"}],"usage":{"cost_in_usd_ticks":700000000}}`,
			want:         0.078 + 0.013,
		},
		{
			name:         "quality tiers: edit with several images",
			path:         "/v1/images/edits",
			requestBody:  `{"model":"quality-model","prompt":"merge","images":[{"url":"https://img.example/a.png","type":"image_url"},{"url":"https://img.example/b.png","type":"image_url"},{"url":"https://img.example/c.png","type":"image_url"}]}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg"}]}`,
			want:         0.078 + 3*0.013,
		},
		{
			name:         "quality tiers: nothing delivered bills neither output nor input",
			path:         "/v1/images/edits",
			requestBody:  `{"model":"quality-model","prompt":"merge","images":[{"url":"https://img.example/a.png"},{"url":"https://img.example/b.png"}]}`,
			upstreamBody: `{"created":1,"data":[]}`,
			want:         0,
		},
		{
			name:         "quality tiers: unreadable response bills the request",
			path:         "/v1/images/edits",
			requestBody:  `{"model":"quality-model","prompt":"merge","images":[{"url":"https://img.example/a.png"},{"url":"https://img.example/b.png"}]}`,
			upstreamType: "text/html",
			upstreamBody: `<html>not an image response</html>`,
			want:         0.078 + 2*0.013,
		},
		{
			name:         "pixel tiers: recognized sizes survive an unrecognized entry",
			path:         "/v1/images/generations",
			requestBody:  `{"model":"pixel-model","prompt":"a cat","n":3}`,
			upstreamBody: `{"data":[{"url":"https://img.example/1.jpeg","size":"1424x800"},{"image":"https://img.example/2.jpeg"},{"url":"https://img.example/3.jpeg","size":"2496x1664"}]}`,
			want:         0.0585 + 0.117 + 0.117,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstreamType := tt.upstreamType
			if upstreamType == "" {
				upstreamType = "application/json"
			}
			prx, dbStub := newTieredImageProxy(t, upstreamType, tt.upstreamBody)

			w := sendImageRequest(t, prx, tt.path, tt.requestBody)

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			assert.InDelta(t, tt.want, dbStub.loggedEntries[0].Spend, 1e-9)
		})
	}
}

func TestProxyRequest_ImageTieredBillingMultipartEdit(t *testing.T) {
	prx, dbStub := newTieredImageProxy(t, "application/json", `{"data":[{"b64_json":"aGVsbG8="}]}`)

	var buf bytes.Buffer
	form := multipart.NewWriter(&buf)
	require.NoError(t, form.WriteField("model", "quality-model"))
	require.NoError(t, form.WriteField("prompt", "merge these"))
	require.NoError(t, form.WriteField("quality", "low"))
	require.NoError(t, form.WriteField("resolution", "2k"))
	for _, name := range []string{"a.png", "b.png"} {
		part, err := form.CreateFormFile("image[]", name)
		require.NoError(t, err)
		_, err = part.Write([]byte("\x89PNG\r\n\x1a\n"))
		require.NoError(t, err)
	}
	mask, err := form.CreateFormFile("mask", "mask.png")
	require.NoError(t, err)
	_, err = mask.Write([]byte("\x89PNG\r\n\x1a\n"))
	require.NoError(t, err)
	require.NoError(t, form.Close())

	req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &buf)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", form.FormDataContentType())
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	// 2k low output plus two source images; the mask is not a source image.
	assert.InDelta(t, 0.078+2*0.013, dbStub.loggedEntries[0].Spend, 1e-9)
}
