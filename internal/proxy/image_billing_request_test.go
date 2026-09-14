package proxy

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImageBillingRequestFromBody_JSON(t *testing.T) {
	inline := `"data:image/png;base64,` + strings.Repeat("A", 4096) + `"`
	body := `{
		"model": "m", "prompt": "a cat", "user": "someone@example.com",
		"Resolution": "1k", "resolution": " 2K ",
		"quality": "Low", "layer_decomposition": true, "n": 2,
		"image": [` + inline + `, {"url": "https://img.example/b.png"}, "", null, 5],
		"images": {"url": "https://img.example/c.png"},
		"metadata": {"k": "v"},
		"long": "` + strings.Repeat("x", 300) + `"
	}`

	requested, edit := imageRequestFromBody([]byte(body), "application/json", true)
	assert.Equal(t, 2, requested)
	assert.Equal(t, "edit", edit.Operation)
	assert.Equal(t, 3, edit.InputImages, "two usable image entries plus one images object")
	assert.Equal(t, map[string]string{
		"Resolution":          `"1k"`,
		"resolution":          `"2k"`,
		"quality":             `"low"`,
		"layer_decomposition": "true",
		"n":                   "2",
	}, edit.RequestParams)

	_, generation := imageRequestFromBody([]byte(body), "application/json", false)
	assert.Equal(t, "generation", generation.Operation)
	assert.Zero(t, generation.InputImages, "generation reports source images through the response")
	assert.NotContains(t, generation.RequestParams, "image")
	assert.NotContains(t, generation.RequestParams, "prompt")
	assert.NotContains(t, generation.RequestParams, "user")
}

func TestImageBillingRequestFromBody_Multipart(t *testing.T) {
	var buf bytes.Buffer
	form := multipart.NewWriter(&buf)
	require.NoError(t, form.WriteField("prompt", "merge"))
	require.NoError(t, form.WriteField("quality", "Medium"))
	require.NoError(t, form.WriteField("n", "2"))
	require.NoError(t, form.WriteField("layer_decomposition", "true"))
	require.NoError(t, form.WriteField("long", strings.Repeat("x", 300)))
	for _, field := range []string{"image", "image[]", "images[]", "mask"} {
		part, err := form.CreateFormFile(field, "f.png")
		require.NoError(t, err)
		_, err = part.Write([]byte("\x89PNG\r\n\x1a\n"))
		require.NoError(t, err)
	}
	require.NoError(t, form.Close())

	requested, details := imageRequestFromBody(buf.Bytes(), form.FormDataContentType(), true)
	assert.Equal(t, 2, requested)
	assert.Equal(t, 3, details.InputImages, "image, image[] and images[] count; mask does not")
	assert.Equal(t, map[string]string{
		"quality":             `"medium"`,
		"n":                   "2",
		"layer_decomposition": "true",
	}, details.RequestParams)
}

// The requested count ("n") is read in the same pass as the pricing inputs.
func TestImageRequestFromBody_RequestedCount(t *testing.T) {
	for body, want := range map[string]int{
		`{"model":"m","n":3}`:    3,
		`{"model":"m"}`:          1,
		`{"model":"m","n":0}`:    1,
		`{"model":"m","n":-2}`:   1,
		`{"model":"m","n":2.5}`:  1,
		`{"model":"m","n":"2"}`:  1,
		`{"model":"m","n":null}`: 1,
		`not json`:               1,
		``:                       1,
	} {
		requested, _ := imageRequestFromBody([]byte(body), "application/json", false)
		assert.Equal(t, want, requested, body)
	}

	multipartCount := func(values ...string) int {
		var buf bytes.Buffer
		form := multipart.NewWriter(&buf)
		part, err := form.CreateFormFile("n", "n.txt") // a file named "n" is not the parameter
		require.NoError(t, err)
		_, err = part.Write([]byte("7"))
		require.NoError(t, err)
		for _, value := range values {
			require.NoError(t, form.WriteField("n", value))
		}
		require.NoError(t, form.Close())
		requested, _ := imageRequestFromBody(buf.Bytes(), form.FormDataContentType(), true)
		return requested
	}
	assert.Equal(t, 4, multipartCount(" 4 "))
	assert.Equal(t, 1, multipartCount())
	assert.Equal(t, 1, multipartCount("zero"))
	assert.Equal(t, 1, multipartCount("0", "5"), "only the first n field is read")
	assert.Equal(t, 2, multipartCount("2", "5"))

	requested, _ := imageRequestFromBody([]byte("n=3"), "multipart/form-data", true)
	assert.Equal(t, 1, requested, "multipart without a boundary")
}

func TestImagePixels(t *testing.T) {
	for size, want := range map[string]int64{
		"2048x2048":               4194304,
		" 1424X800 ":              1139200,
		"2K":                      0,
		"":                        0,
		"0x100":                   0,
		"-5x100":                  0,
		"100x":                    0,
		"99999999999x99999999999": 0,
		"1024 × 768":              786432,
		"16:9":                    0,
	} {
		assert.Equal(t, want, imagePixels(size), size)
	}
}

// A stream relayed through a proxy credential (AIR → AIR) must bill the images
// its terminal usage event reports, like a direct provider stream does.
func TestProxyRequest_ImageStreamThroughProxyCredentialUsesGeneratedImages(t *testing.T) {
	stream := strings.Join([]string{
		"event: image_generation.partial_succeeded",
		`data: {"type":"image_generation.partial_succeeded","image_index":0,"url":"https://img.example/1.png","size":"1024x1024"}`,
		"",
		"event: image_generation.partial_succeeded",
		`data: {"type":"image_generation.partial_succeeded","image_index":1,"url":"https://img.example/2.png","size":"1024x1024"}`,
		"",
		"event: image_generation.completed",
		`data: {"type":"image_generation.completed","usage":{"generated_images":2,"input_images":3,"output_tokens":8000,"total_tokens":8000}}`,
		"",
		"data: [DONE]",
		"",
		"",
	}, "\n")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(stream))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "air-hop",
			Type:    config.ProviderTypeProxy,
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
		"image-model": {OutputCostPerImage: testImagePrice, InputCostPerImage: 0.01, InputImagesFreePerRequest: 1},
	})
	prx.priceRegistry = registry

	w := sendImageRequest(t, prx, "/v1/images/generations", `{"model":"image-model","prompt":"a cat, day and night","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, 2*testImagePrice+2*0.01, dbStub.loggedEntries[0].Spend, 1e-9)
}
