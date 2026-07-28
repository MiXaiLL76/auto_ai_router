package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildProxyBodyTestProxyWithMM constructs a Proxy using the provided model manager.
// This lets us inject a model manager that has a model:real-name mapping.
func buildProxyBodyTestProxyWithMM(credURL string, mm *models.Manager) *Proxy {
	builder := NewTestProxyBuilder().
		WithSingleCredential("upstream", config.ProviderTypeProxy, credURL, "upstream-key").
		WithMasterKey("master-key")
	mm.LoadModelsFromConfig(builder.config.Credentials)
	builder.config.ModelManager = mm
	return builder.Build()
}

// buildProxyBodyTestProxyWithFallbackMM is the same but with primary + fallback credential.
func buildProxyBodyTestProxyWithFallbackMM(primaryURL, fallbackURL string, mm *models.Manager) *Proxy {
	builder := NewTestProxyBuilder().
		WithPrimaryAndFallback(primaryURL, fallbackURL).
		WithMasterKey("master-key")
	mm.LoadModelsFromConfig(builder.config.Credentials)
	builder.config.ModelManager = mm
	return builder.Build()
}

// TestProxyBody_NoAlias verifies that when modelID == realModelID (no alias configured),
// the proxy forwards the exact original body (proxyBody == body).
func TestProxyBody_NoAlias(t *testing.T) {
	var receivedBody []byte

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		receivedBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("id-1", "gpt-4", "ok"))
	}))
	defer upstream.Close()

	logger := testhelpers.NewTestLogger()
	mm := models.New(logger, 50, []config.ModelRPMConfig{}) // no alias mapping

	prx := buildProxyBodyTestProxyWithMM(upstream.URL, mm)

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	require.NotEmpty(t, receivedBody)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(receivedBody, &parsed))

	// The proxy must forward the model name unchanged when there is no alias.
	assert.Equal(t, "gpt-4", parsed["model"], "model in forwarded body must equal original when no alias configured")
}

// TestProxyBody_WithAlias verifies that when a model alias is configured
// (Name "anthropic/claude-sonnet-4.6" -> real "global.anthropic.claude-sonnet-4-6"),
// proxyBody restores the alias so the upstream proxy receives the original name,
// while body carries the real provider name.
func TestProxyBody_WithAlias(t *testing.T) {
	const modelAlias = "anthropic/claude-sonnet-4.6"
	const modelReal = "global.anthropic.claude-sonnet-4-6"

	var receivedModel string

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(bodyBytes, &parsed))
		if m, ok := parsed["model"].(string); ok {
			receivedModel = m
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// A proxy hop may expose its provider-facing model in the response even
		// though it accepted the public alias in the request.
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("id-2", modelReal, "ok"))
	}))
	defer upstream.Close()

	logger := testhelpers.NewTestLogger()
	mm := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: modelAlias, Model: modelReal},
	})

	prx := buildProxyBodyTestProxyWithMM(upstream.URL, mm)

	// Client sends the alias name
	reqBody, err := json.Marshal(map[string]interface{}{
		"model":    modelAlias,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(reqBody)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	// The proxy-type upstream must receive the original alias name, NOT the provider real name.
	assert.Equal(t, modelAlias, receivedModel,
		"proxy upstream should receive the alias name (proxyBody), not the real provider name")
	var clientResponse map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &clientResponse))
	assert.Equal(t, modelAlias, clientResponse["model"],
		"client response should restore the public alias exposed by this router")
}

func TestProxyBody_GlobalModelAliasRestoresClientVisibleModel(t *testing.T) {
	const publicModel = "openai/gpt-4o-mini"
	const backendModel = "backend-gpt-4o-mini"

	var receivedModel string
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var requestBody map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&requestBody))
		receivedModel, _ = requestBody["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("id-global-alias", backendModel, "ok"))
	}))
	defer upstream.Close()

	logger := testhelpers.NewTestLogger()
	manager := models.New(logger, 50, nil)
	manager.SetModelAliases(map[string]string{publicModel: backendModel})
	prx := buildProxyBodyTestProxyWithMM(upstream.URL, manager)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, backendModel, receivedModel)
	var response map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, publicModel, response["model"])
}

// TestProxyBody_FallbackReceivesAlias verifies that when the primary proxy returns 429
// and a fallback proxy credential is tried, the fallback also receives the alias name
// (proxyBody), not the internal real provider name.
func TestProxyBody_FallbackReceivesAlias(t *testing.T) {
	const modelAlias = "anthropic/claude-sonnet-4.6"
	const modelReal = "global.anthropic.claude-sonnet-4-6"

	var primaryCalls int32
	var fallbackReceivedModel string

	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "rate_limit_exceeded",
			"message": "rate limited",
		})
	}))
	defer primary.Close()

	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		var parsed map[string]interface{}
		if json.Unmarshal(bodyBytes, &parsed) == nil {
			if m, ok := parsed["model"].(string); ok {
				fallbackReceivedModel = m
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("id-fb", modelAlias, "ok from fallback"))
	}))
	defer fallback.Close()

	logger := testhelpers.NewTestLogger()
	mm := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: modelAlias, Model: modelReal},
	})

	prx := buildProxyBodyTestProxyWithFallbackMM(primary.URL, fallback.URL, mm)

	reqBody, err := json.Marshal(map[string]interface{}{
		"model":    modelAlias,
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
	})
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(reqBody)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), atomic.LoadInt32(&primaryCalls), "primary should be called once")

	// The fallback proxy must receive the alias, not the real provider name.
	assert.Equal(t, modelAlias, fallbackReceivedModel,
		"fallback proxy should receive alias name (proxyBody), not real provider name %q", modelReal)
}

// TestProxyBody_NoAlias_FallbackBodyIdentical verifies that when there is no alias,
// primary and fallback both receive the exact same body.
func TestProxyBody_NoAlias_FallbackBodyIdentical(t *testing.T) {
	var primaryBody, fallbackBody []byte

	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		primaryBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primary.Close()

	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		fallbackBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("id-ok", "gpt-4", "ok"))
	}))
	defer fallback.Close()

	logger := testhelpers.NewTestLogger()
	mm := models.New(logger, 50, []config.ModelRPMConfig{}) // no alias

	prx := buildProxyBodyTestProxyWithFallbackMM(primary.URL, fallback.URL, mm)

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	require.NotEmpty(t, primaryBody)
	require.NotEmpty(t, fallbackBody)

	// Without alias, body == proxyBody, so both upstreams see the same bytes.
	assert.Equal(t, primaryBody, fallbackBody,
		"primary and fallback should receive identical body when no alias is configured")
}

// TestProxyBody_OrchestratedRequest_ProxyBodySet verifies the orchestratedRequest struct
// fields directly: proxyBody equals body when no alias, and proxyBody contains the alias
// when modelID != realModelID.
func TestProxyBody_OrchestratedRequest_ProxyBodySet(t *testing.T) {
	t.Run("no alias: proxyBody equals body", func(t *testing.T) {
		logger := testhelpers.NewTestLogger()
		mm := models.New(logger, 50, []config.ModelRPMConfig{})

		builder := NewTestProxyBuilder().
			WithSingleCredential("c", config.ProviderTypeProxy, "http://nowhere.invalid", "k").
			WithMasterKey("master-key")
		mm.LoadModelsFromConfig(builder.config.Credentials)
		builder.config.ModelManager = mm
		prx := builder.Build()

		reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		logCtx := &RequestLogContext{}

		prepared, ok := prx.orchestrateRequest(w, req, logCtx)
		require.True(t, ok)
		require.NotNil(t, prepared)

		assert.Equal(t, prepared.modelID, prepared.realModelID,
			"modelID and realModelID must be equal when no alias configured")
		assert.Equal(t, prepared.body, prepared.proxyBody,
			"proxyBody must equal body when there is no alias (modelID == realModelID)")
	})

	t.Run("alias configured: proxyBody has alias, body has real name", func(t *testing.T) {
		const modelAlias = "my-claude"
		const modelReal = "global.anthropic.claude-sonnet-4-6"

		logger := testhelpers.NewTestLogger()
		mm := models.New(logger, 50, []config.ModelRPMConfig{
			{Name: modelAlias, Model: modelReal},
		})

		builder := NewTestProxyBuilder().
			WithSingleCredential("c", config.ProviderTypeProxy, "http://nowhere.invalid", "k").
			WithMasterKey("master-key")
		mm.LoadModelsFromConfig(builder.config.Credentials)
		builder.config.ModelManager = mm
		prx := builder.Build()

		reqBodyStr, err := json.Marshal(map[string]interface{}{
			"model":    modelAlias,
			"messages": []map[string]string{{"role": "user", "content": "hi"}},
		})
		require.NoError(t, err)

		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(reqBodyStr)))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		logCtx := &RequestLogContext{}

		prepared, ok := prx.orchestrateRequest(w, req, logCtx)
		require.True(t, ok)
		require.NotNil(t, prepared)

		assert.Equal(t, modelAlias, prepared.modelID)
		assert.Equal(t, modelReal, prepared.realModelID)

		// body must contain the real provider name
		var bodyParsed map[string]interface{}
		require.NoError(t, json.Unmarshal(prepared.body, &bodyParsed))
		assert.Equal(t, modelReal, bodyParsed["model"],
			"prepared.body must contain the real provider model name")

		// proxyBody must contain the original alias
		var proxyBodyParsed map[string]interface{}
		require.NoError(t, json.Unmarshal(prepared.proxyBody, &proxyBodyParsed))
		assert.Equal(t, modelAlias, proxyBodyParsed["model"],
			"prepared.proxyBody must restore the alias name for proxy forwarding")
	})
}
