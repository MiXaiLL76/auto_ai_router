package proxy

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/sosana"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	litellmmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	aimodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_SosanaImageGenerationSuccess(t *testing.T) {
	var createSeen, pollSeen bool
	var imageAuths []string
	imageServer := newSosanaResultImageServer(t, http.StatusOK, "image/png", sosanaResultPNG, &imageAuths)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer sosana-key", r.Header.Get("Authorization"))
		switch r.URL.Path {
		case "/api/banana/create-async":
			createSeen = true
			var req map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			assert.Equal(t, "draw a fox", req["prompt"])
			assert.Equal(t, "nano-banana", req["model"])
			assert.Equal(t, "1:1", req["aspect_ratio"])
			assert.Equal(t, false, req["prompt_optimization"])
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw a fox"}`))
		case "/api/banana/task-1":
			pollSeen = true
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","created_at":"2026-01-01T00:00:00Z","prompt":"draw a fox","optimized_prompt":"A detailed fox illustration","result_file_url":%q}`, imageServer.URL+"/fox.png")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw a fox","size":"1024x1024","n":1,"response_format":"b64_json"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 1)
	assert.Empty(t, resp.Data[0].URL)
	assert.Equal(t, base64.StdEncoding.EncodeToString(sosanaResultPNG), resp.Data[0].B64JSON)
	assert.Equal(t, "A detailed fox illustration", resp.Data[0].RevisedPrompt)
	assert.Equal(t, int64(1767225600), resp.Created)
	assert.NotContains(t, w.Body.String(), imageServer.URL)
	assert.NotContains(t, w.Body.String(), "sosana")
	assert.NotContains(t, w.Body.String(), "cdn")
	assert.Equal(t, []string{""}, imageAuths)
	assert.True(t, createSeen)
	assert.True(t, pollSeen)
}

func TestProxyRequest_SosanaImageEditUsesProviderModelAlias(t *testing.T) {
	imageServer := newSosanaResultImageServer(t, http.StatusOK, "image/png", sosanaResultPNG, nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/banana/create-async", r.URL.Path)
		assert.Equal(t, "Bearer sosana-key", r.Header.Get("Authorization"))

		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.Equal(t, "make it blue", req["prompt"])
		assert.Equal(t, "nano-banana", req["model"])
		assert.Equal(t, false, req["prompt_optimization"])
		imageURLs, ok := req["image_urls"].([]any)
		require.True(t, ok)
		require.Len(t, imageURLs, 1)
		assert.True(t, strings.HasPrefix(imageURLs[0].(string), "data:image/png;base64,"))

		_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","created_at":"2026-01-01T00:00:00Z","prompt":"make it blue","result_file_url":%q}`, imageServer.URL+"/edit.png")
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	prx.modelManager = aimodels.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "public-image", Model: "nano-banana", Credential: "sosana"},
	})

	body, contentType := sosanaMultipartEditBody(t, map[string]string{
		"model":  "public-image",
		"prompt": "make it blue",
		"n":      "1",
	}, map[string][]byte{
		"image": {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'},
	})
	req := httptest.NewRequest("POST", "/v1/images/edits", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", contentType)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 1)
	assert.Empty(t, resp.Data[0].URL)
	assert.Equal(t, base64.StdEncoding.EncodeToString(sosanaResultPNG), resp.Data[0].B64JSON)
}

func TestProxyRequest_SosanaImageGenerationLogsLiteLLMImageSpend(t *testing.T) {
	imageServer := newSosanaResultImageServer(t, http.StatusOK, "image/png", sosanaResultPNG, nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/spend.png")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	spendManager := newCapturedSpendManager()
	priceRegistry := aimodels.NewModelPriceRegistry()
	priceRegistry.Update(map[string]*aimodels.ModelPrice{
		"nano-banana": {OutputCostPerImage: 0.07},
	})

	prx := newSosanaTestProxy(upstream.URL, nil)
	prx.LiteLLMDB = spendManager
	prx.priceRegistry = priceRegistry
	prx.modelManager = aimodels.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "public-image", Model: "nano-banana", Credential: "sosana"},
	})

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"public-image","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, spendManager.entries, 1)
	entry := spendManager.entries[0]
	assert.Equal(t, "/v1/images/generations", entry.CallType)
	assert.Equal(t, "public-image", entry.Model)
	assert.Equal(t, "sosana:public-image", entry.ModelID)
	assert.Equal(t, "sosana", entry.CustomLLMProvider)
	assert.Equal(t, "success", entry.Status)
	assert.Equal(t, 0, entry.TotalTokens)
	assert.InDelta(t, 0.07, entry.Spend, 0.0000001)
	var metadata map[string]any
	require.NoError(t, json.Unmarshal([]byte(entry.Metadata), &metadata))
	costBreakdown, ok := metadata["cost_breakdown"].(map[string]any)
	require.True(t, ok)
	assert.InDelta(t, 0.07, costBreakdown["total_cost"].(float64), 0.0000001)
}

func TestProxyRequest_SosanaImageGenerationWritesLiteLLMSpendLogIntegration(t *testing.T) {
	dbURL := os.Getenv("LITELLM_DATABASE_URL")
	if dbURL == "" {
		t.Skip("LITELLM_DATABASE_URL not set, skipping LiteLLM spend-log integration test")
	}

	imageServer := newSosanaResultImageServer(t, http.StatusOK, "image/png", sosanaResultPNG, nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/spend.png")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	manager, err := litellmdb.New(&litellmmodels.Config{
		DatabaseURL:      dbURL,
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     1,
		LogFlushInterval: 100 * time.Millisecond,
	})
	require.NoError(t, err)
	defer func() {
		_ = manager.Shutdown(context.Background())
	}()

	alias := fmt.Sprintf("sosana-spend-test-%d", time.Now().UnixNano())
	priceRegistry := aimodels.NewModelPriceRegistry()
	priceRegistry.Update(map[string]*aimodels.ModelPrice{
		"nano-banana": {OutputCostPerImage: 0.07},
	})

	prx := newSosanaTestProxy(upstream.URL, nil)
	prx.LiteLLMDB = manager
	prx.priceRegistry = priceRegistry
	prx.modelManager = aimodels.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: alias, Model: "nano-banana", Credential: "sosana"},
	})

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"`+alias+`","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var requestID string
	var spend float64
	var model string
	var provider string
	var status string
	var totalTokens int
	for {
		err = manager.GetPool().QueryRow(ctx, `
			SELECT request_id, spend, model, custom_llm_provider, status, total_tokens
			FROM "LiteLLM_SpendLogs"
			WHERE model = $1 AND custom_llm_provider = 'sosana'
			ORDER BY "startTime" DESC
			LIMIT 1
		`, alias).Scan(&requestID, &spend, &model, &provider, &status, &totalTokens)
		if err == nil {
			break
		}
		if ctx.Err() != nil {
			require.NoError(t, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	defer func() {
		_, _ = manager.GetPool().Exec(context.Background(), `DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1`, requestID)
	}()

	assert.Equal(t, alias, model)
	assert.Equal(t, "sosana", provider)
	assert.Equal(t, "success", status)
	assert.Equal(t, 0, totalTokens)
	assert.InDelta(t, 0.07, spend, 0.0000001)
}

func TestProxyRequest_SosanaRejectsNonImageEndpoint(t *testing.T) {
	called := false
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"nano-banana","messages":[]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "provider supports only image generation")
	assert.False(t, called)
}

func TestProxyRequest_SosanaRejectsURLResponseFormat(t *testing.T) {
	called := false
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1,"response_format":"url"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "response_format=url")
	assert.False(t, called)
}

func TestProxyRequest_IncompatibleImageRequestWithSosanaReturnsLocalError(t *testing.T) {
	called := false
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","tools":[{"type":"google_search"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "tools is unsupported")
	assert.False(t, called)
}

func TestProxyRequest_IncompatibleImageEditJPEGWithSosanaReturnsLocalError(t *testing.T) {
	called := false
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	body, contentType := sosanaMultipartEditBody(t, map[string]string{
		"model":  "nano-banana",
		"prompt": "make it blue",
	}, map[string][]byte{
		"image": {0xff, 0xd8, 0xff, 0xdb, 0, 0x43},
	})
	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/images/edits", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", contentType)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "PNG input")
	assert.False(t, called)
}

func TestProxyRequest_SosanaCreateHTTPErrorMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"detail":"sosana balance secret marker"}`))
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusPaymentRequired, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "balance secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.Contains(t, logBuf.String(), "balance secret")
}

func TestProxyRequest_SosanaRetriesCreateWithNextCredential(t *testing.T) {
	var createAuths []string
	imageServer := newSosanaResultImageServer(t, http.StatusOK, "image/png", sosanaResultPNG, nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			createAuths = append(createAuths, r.Header.Get("Authorization"))
			if r.Header.Get("Authorization") == "Bearer sosana-key-a" {
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"detail":"first credential rate limited"}`))
				return
			}
			assert.Equal(t, "Bearer sosana-key-b", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"uid":"task-2","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-2":
			assert.Equal(t, "Bearer sosana-key-b", r.Header.Get("Authorization"))
			_, _ = fmt.Fprintf(w, `{"uid":"task-2","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/retry.png")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "sosana-a", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-a", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "sosana-b", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-b", RPM: 100, TPM: 10000},
		).
		WithMaxProviderRetries(1).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, []string{"Bearer sosana-key-a", "Bearer sosana-key-b"}, createAuths)
	assert.Contains(t, w.Body.String(), base64.StdEncoding.EncodeToString(sosanaResultPNG))
	assert.NotContains(t, w.Body.String(), imageServer.URL)
}

func TestProxyRequest_SosanaDoesNotRetryCreateTransportError(t *testing.T) {
	deadServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := deadServer.URL
	deadServer.Close()

	liveCalled := false
	liveServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		liveCalled = true
		_, _ = w.Write([]byte(`{"uid":"task-2","status":"COMPLETED","prompt":"draw","result_file_url":"https://cdn.sosana.art/unwanted.png"}`))
	}))
	defer liveServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "sosana-a", Type: config.ProviderTypeSosana, BaseURL: deadURL, APIKey: "sosana-key-a", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "sosana-b", Type: config.ProviderTypeSosana, BaseURL: liveServer.URL, APIKey: "sosana-key-b", RPM: 100, TPM: 10000},
		).
		WithMaxProviderRetries(1).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadGateway, w.Code)
	assert.False(t, liveCalled)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "unwanted.png")
}

func TestProxyRequest_SosanaPollHTTPErrorMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
		case "/api/banana/task-1":
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"detail":"poll secret marker"}`))
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "poll secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.Contains(t, logBuf.String(), "poll secret")
}

func TestProxyRequest_SosanaImageResultHTTPErrorMasked(t *testing.T) {
	var logBuf bytes.Buffer
	imageServer := newSosanaResultImageServer(t, http.StatusInternalServerError, "text/plain", []byte("storage secret marker"), nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/missing.png")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "storage secret")
	assert.NotContains(t, w.Body.String(), imageServer.URL)
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.Contains(t, logBuf.String(), "storage secret")
	assert.Contains(t, logBuf.String(), "result_host=127.0.0.1")
}

func TestProxyRequest_SosanaImageResultNonImageMasked(t *testing.T) {
	var logBuf bytes.Buffer
	imageServer := newSosanaResultImageServer(t, http.StatusOK, "text/plain", []byte("not an image secret marker"), nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/text.txt")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "not an image")
	assert.NotContains(t, w.Body.String(), imageServer.URL)
	assert.Contains(t, logBuf.String(), "non-PNG content")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.Contains(t, logBuf.String(), "not an image secret marker")
}

func TestProxyRequest_SosanaImageResultJPEGMasked(t *testing.T) {
	var logBuf bytes.Buffer
	imageServer := newSosanaResultImageServer(t, http.StatusOK, "image/jpeg", []byte{0xff, 0xd8, 0xff, 0xdb, 0, 0x43}, nil)
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/image.jpg")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), imageServer.URL)
	assert.Contains(t, logBuf.String(), "non-PNG content")
	assert.NotContains(t, logBuf.String(), imageServer.URL)
}

func TestProxyRequest_SosanaImageResultRedirectMaskedAndNotFollowed(t *testing.T) {
	allowPrivateSosanaResultURLsForTest(t)

	var logBuf bytes.Buffer
	targetCalled := false
	targetServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetCalled = true
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(sosanaResultPNG)
	}))
	defer targetServer.Close()

	redirectServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, targetServer.URL+"/private.png", http.StatusFound)
	}))
	defer redirectServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, redirectServer.URL+"/redirect")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.False(t, targetCalled)
	assert.NotContains(t, w.Body.String(), targetServer.URL)
}

func TestDownloadSosanaResultImageRejectsUnsafeProductionURL(t *testing.T) {
	var logBuf bytes.Buffer
	called := false
	imageServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer imageServer.Close()

	resultURL := imageServer.URL + "/private.png"
	prx := newSosanaTestProxy("https://sosana.art", &logBuf)
	cred := &config.CredentialConfig{Name: "sosana", Type: config.ProviderTypeSosana, BaseURL: "https://sosana.art"}
	image, _, statusCode, err := prx.downloadSosanaResultImage(context.Background(), cred, "nano-banana", sosana.BananaTaskResponse{
		Status:        sosana.StatusCompleted,
		ResultFileURL: &resultURL,
	}, nil, &RequestLogContext{})

	require.Error(t, err)
	assert.Nil(t, image)
	assert.Equal(t, http.StatusBadGateway, statusCode)
	assert.False(t, called)
	assert.Contains(t, logBuf.String(), "unsafe result URL")
	assert.Contains(t, logBuf.String(), "result_host=127.0.0.1")
}

func TestValidateSosanaResultURLBlocksUnsafeHosts(t *testing.T) {
	tests := []string{
		"http://main-r2.sosana.blog/image.png",
		"https://localhost/image.png",
		"https://127.0.0.1/image.png",
		"https://169.254.169.254/latest/meta-data",
		"https://100.64.0.1/image.png",
		"https://example.com/image.png",
	}

	for _, rawURL := range tests {
		t.Run(rawURL, func(t *testing.T) {
			parsed, err := parseSosanaResultURL(rawURL)
			require.NoError(t, err)
			require.Error(t, validateSosanaResultURL(context.Background(), parsed))
		})
	}
}

func TestValidateSosanaResultURLAllowsLocalOnlyWithTestHook(t *testing.T) {
	allowPrivateSosanaResultURLsForTest(t)

	parsed, err := parseSosanaResultURL("http://127.0.0.1/image.png")
	require.NoError(t, err)
	require.NoError(t, validateSosanaResultURL(context.Background(), parsed))
}

func TestProxyRequest_SosanaImageResultTimeoutMasked(t *testing.T) {
	allowPrivateSosanaResultURLsForTest(t)

	var logBuf bytes.Buffer
	imageServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(sosanaResultPNG)
	}))
	defer imageServer.Close()

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = fmt.Fprintf(w, `{"uid":"task-1","status":"COMPLETED","prompt":"draw","result_file_url":%q}`, imageServer.URL+"/slow.png")
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	prx.requestTimeout = 5 * time.Millisecond
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusRequestTimeout, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "slow.png")
	assert.Contains(t, logBuf.String(), "result image download failed")
	assert.Contains(t, logBuf.String(), "result_host=127.0.0.1")
}

func TestProxyRequest_SosanaDoesNotRetryAfterTaskCreated(t *testing.T) {
	createCalls := 0
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			createCalls++
			assert.Equal(t, "Bearer sosana-key-a", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			assert.Equal(t, "Bearer sosana-key-a", r.Header.Get("Authorization"))
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"detail":"poll failed after task was created"}`))
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "sosana-a", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-a", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "sosana-b", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-b", RPM: 100, TPM: 10000},
		).
		WithMaxProviderRetries(1).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Equal(t, 1, createCalls)
	assert.NotContains(t, w.Body.String(), "poll failed")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
}

func TestProxyRequest_SosanaTaskFailedMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"FAILED","created_at":"2026-01-01T00:00:00Z","prompt":"draw","error":"failed secret marker"}`))
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "failed secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.Contains(t, logBuf.String(), "failed secret")
}

func TestProxyRequest_SosanaTaskModeratedMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"MODERATED","created_at":"2026-01-01T00:00:00Z","prompt":"draw","error":"moderation secret marker"}`))
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "content_policy_violation")
	assert.NotContains(t, w.Body.String(), "moderation secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.Contains(t, logBuf.String(), "moderation secret")
}

func TestProxyRequest_SosanaTimeoutMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	prx.requestTimeout = 5 * time.Millisecond
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusRequestTimeout, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
}

var sosanaResultPNG = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 0}

func newSosanaResultImageServer(t *testing.T, status int, contentType string, body []byte, auths *[]string) *httptest.Server {
	t.Helper()
	allowPrivateSosanaResultURLsForTest(t)

	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auths != nil {
			*auths = append(*auths, r.Header.Get("Authorization"))
		}
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
}

func allowPrivateSosanaResultURLsForTest(t *testing.T) {
	t.Helper()

	previous := allowPrivateSosanaResultURLForTests
	allowPrivateSosanaResultURLForTests = func(parsed *url.URL) bool {
		if parsed.Scheme != "http" {
			return false
		}
		host := parsed.Hostname()
		if strings.EqualFold(host, "localhost") {
			return true
		}
		ip := net.ParseIP(host)
		return ip != nil && isUnsafeSosanaResultIP(ip)
	}
	t.Cleanup(func() {
		allowPrivateSosanaResultURLForTests = previous
	})
}

func newSosanaTestProxy(baseURL string, logBuf *bytes.Buffer) *Proxy {
	logger := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	if logBuf != nil {
		logger = slog.New(slog.NewTextHandler(logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	}
	return NewTestProxyBuilder().
		WithSingleCredential("sosana", config.ProviderTypeSosana, baseURL, "sosana-key").
		WithRequestTimeout(30 * time.Second).
		withLogger(logger).
		Build()
}

func (b *TestProxyBuilder) withLogger(logger *slog.Logger) *TestProxyBuilder {
	b.config.Logger = logger
	b.config.TokenManager = createTestTokenManager(logger)
	b.config.ModelManager = createTestModelManager(logger)
	return b
}

func sosanaMultipartEditBody(t *testing.T, fields map[string]string, files map[string][]byte) ([]byte, string) {
	t.Helper()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	for key, value := range fields {
		require.NoError(t, writer.WriteField(key, value))
	}
	for key, data := range files {
		part, err := writer.CreateFormFile(key, key+".png")
		require.NoError(t, err)
		_, err = part.Write(data)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Close())
	return buf.Bytes(), writer.FormDataContentType()
}

type capturedSpendManager struct {
	*litellmdb.NoopManager
	entries []*litellmmodels.SpendLogEntry
}

func newCapturedSpendManager() *capturedSpendManager {
	return &capturedSpendManager{NoopManager: litellmdb.NewNoopManager()}
}

func (m *capturedSpendManager) IsEnabled() bool {
	return true
}

func (m *capturedSpendManager) IsHealthy() bool {
	return true
}

func (m *capturedSpendManager) LogSpend(entry *litellmmodels.SpendLogEntry) error {
	m.entries = append(m.entries, entry)
	return nil
}
