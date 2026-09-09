package proxy

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_ImageEditReplacesModelAlias(t *testing.T) {
	const modelAlias = "gpt-image-2-vsellm"
	const realModel = "gpt-image-2"
	imageData := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

	var receivedModel string
	var receivedImage []byte
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, r.ParseMultipartForm(1<<20))
		receivedModel = r.FormValue("model")
		file, _, err := r.FormFile("image")
		require.NoError(t, err)
		receivedImage, err = io.ReadAll(file)
		require.NoError(t, err)
		require.NoError(t, file.Close())

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"created":1,"data":[{"b64_json":"aW1hZ2U="}]}`)
	}))
	defer upstream.Close()

	logger := testhelpers.NewTestLogger()
	modelManager := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: modelAlias, Model: realModel, RPM: 100, TPM: -1},
	})
	modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "azure",
		Type: config.ProviderTypeOpenAI,
	}})
	proxy := NewTestProxyBuilder().
		WithSingleCredential("azure", config.ProviderTypeOpenAI, upstream.URL, "azure-key").
		Build()
	proxy.modelManager = modelManager
	proxy.balancer.SetModelChecker(modelManager)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", modelAlias))
	require.NoError(t, writer.WriteField("prompt", "make it blue"))
	imagePart, err := writer.CreateFormFile("image", "input.png")
	require.NoError(t, err)
	_, err = imagePart.Write(imageData)
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &body)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()

	proxy.ProxyRequest(response, req)

	require.Equal(t, http.StatusOK, response.Code)
	require.Equal(t, realModel, receivedModel)
	require.Equal(t, imageData, receivedImage)
}

func TestProxyRequest_ImageEditPreservesModelAliasForProxy(t *testing.T) {
	const modelAlias = "gpt-image-2-vsellm"
	const realModel = "gpt-image-2"
	imageData := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

	var receivedModel string
	var receivedImage []byte
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, r.ParseMultipartForm(1<<20))
		receivedModel = r.FormValue("model")
		file, _, err := r.FormFile("image")
		require.NoError(t, err)
		receivedImage, err = io.ReadAll(file)
		require.NoError(t, err)
		require.NoError(t, file.Close())

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"created":1,"data":[{"b64_json":"aW1hZ2U="}]}`)
	}))
	defer upstream.Close()

	logger := testhelpers.NewTestLogger()
	modelManager := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: modelAlias, Model: realModel, RPM: 100, TPM: -1},
	})
	builder := NewTestProxyBuilder().
		WithSingleCredential("proxy", config.ProviderTypeProxy, upstream.URL, "proxy-key")
	modelManager.LoadModelsFromConfig(builder.config.Credentials)
	builder.config.ModelManager = modelManager
	proxy := builder.Build()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", modelAlias))
	require.NoError(t, writer.WriteField("prompt", "make it blue"))
	imagePart, err := writer.CreateFormFile("image", "input.png")
	require.NoError(t, err)
	_, err = imagePart.Write(imageData)
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &body)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()

	proxy.ProxyRequest(response, req)

	require.Equal(t, http.StatusOK, response.Code)
	require.Equal(t, modelAlias, receivedModel)
	require.Equal(t, imageData, receivedImage)
}

func TestProxyRequest_ImageEditSendsCanonicalModelToAIR(t *testing.T) {
	tests := []struct {
		name      string
		alias     string
		canonical string
	}{
		{name: "Gemini", alias: "google/gemini-3.1-flash-image", canonical: "gemini-3.1-flash-image"},
		{name: "OpenAI", alias: "openai/gpt-image-1", canonical: "gpt-image-1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			imageData := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}
			var receivedModel string
			var receivedImage []byte
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.NoError(t, r.ParseMultipartForm(1<<20))
				receivedModel = r.FormValue("model")
				file, header, err := r.FormFile("image")
				require.NoError(t, err)
				require.Equal(t, "input.png", header.Filename)
				require.Equal(t, "image/png", header.Header.Get("Content-Type"))
				receivedImage, err = io.ReadAll(file)
				require.NoError(t, err)
				require.NoError(t, file.Close())

				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"created":1,"data":[{"b64_json":"aW1hZ2U="}]}`)
			}))
			defer upstream.Close()

			logger := testhelpers.NewTestLogger()
			modelManager := models.New(logger, 50, []config.ModelRPMConfig{
				{Name: tt.canonical, Credential: "air-ru01", RPM: 100, TPM: -1},
			})
			modelManager.SetAcceptedModelAliases(map[string]string{tt.alias: tt.canonical})
			builder := NewTestProxyBuilder().
				WithSingleCredential("air-ru01", config.ProviderTypeAIR, upstream.URL, "ru01-key")
			modelManager.LoadModelsFromConfig(builder.config.Credentials)
			builder.config.ModelManager = modelManager
			proxy := builder.Build()

			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			require.NoError(t, writer.WriteField("model", tt.alias))
			require.NoError(t, writer.WriteField("prompt", "make it blue"))
			header := make(textproto.MIMEHeader)
			header.Set("Content-Disposition", `form-data; name="image"; filename="input.png"`)
			header.Set("Content-Type", "image/png")
			imagePart, err := writer.CreatePart(header)
			require.NoError(t, err)
			_, err = imagePart.Write(imageData)
			require.NoError(t, err)
			require.NoError(t, writer.Close())

			req := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &body)
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", writer.FormDataContentType())
			response := httptest.NewRecorder()

			proxy.ProxyRequest(response, req)

			require.Equal(t, http.StatusOK, response.Code)
			require.Equal(t, tt.canonical, receivedModel)
			require.Equal(t, imageData, receivedImage)
		})
	}
}
