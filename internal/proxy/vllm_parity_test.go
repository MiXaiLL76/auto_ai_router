package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// vllmCapture records what a fake vLLM server received.
type vllmCapture struct {
	mu     sync.Mutex
	path   string
	auth   string
	hasKey bool
	body   map[string]any
}

func (c *vllmCapture) snapshot() (path, auth string, hasAuth bool, body map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.path, c.auth, c.hasKey, c.body
}

func newFakeVLLM(t *testing.T, capture *vllmCapture) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		capture.mu.Lock()
		capture.path = r.URL.Path
		capture.auth = r.Header.Get("Authorization")
		_, capture.hasKey = r.Header["Authorization"]
		capture.body = body
		capture.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/responses" {
			_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"qwen-36-35b-fp8","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`))
			return
		}
		if r.URL.Path == "/v1/embeddings" {
			_, _ = w.Write([]byte(`{"object":"list","model":"qwen3-embedding-8b","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":7,"total_tokens":7}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"qwen-36-35b-fp8","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newVLLMProxy builds a proxy in front of one keyless vLLM credential that serves the
// deployments a LiteLLM database would describe: a model group alias resolving to a real
// vLLM model name with default sampling params, and an embedding model that also has
// (irrelevant) defaults which must never reach a non-chat request.
func newVLLMProxy(t *testing.T, upstreamURL string, db *organizationPolicyTestDB) *Proxy {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	credential := config.CredentialConfig{
		Name: "ray-service-prod", Type: config.ProviderTypeVLLM, BaseURL: upstreamURL, RPM: 100, TPM: 100000,
	}

	manager := routermodels.New(logger, 100, nil)
	manager.SetCredentials([]config.CredentialConfig{credential})
	manager.UpdateDBModels([]config.ModelRPMConfig{
		{
			Name: "qwen-flash", Model: "qwen-36-35b-fp8", Credential: credential.Name, RPM: -1, TPM: -1,
			DefaultParams: map[string]any{
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
				"temperature":          0.7,
				"top_k":                20.0,
			},
		},
		{
			Name: "qwen3-embedding-8b", Credential: credential.Name, RPM: -1, TPM: -1,
			DefaultParams: map[string]any{"temperature": 0.5},
		},
	}, nil, []config.CredentialConfig{credential})

	prices := routermodels.NewModelPriceRegistry()
	prices.MergeDB(map[string]*routermodels.ModelPrice{
		"qwen-flash":         {InputCostPerToken: 2e-7, OutputCostPerToken: 1e-6},
		"qwen3-embedding-8b": {InputCostPerToken: 1e-8},
	})

	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	prx := builder.Build()
	prx.LiteLLMDB = db
	prx.priceRegistry = prices
	return prx
}

func newVLLMTestDB() *organizationPolicyTestDB {
	return &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "key-owner", TeamID: "team-1"},
	}}
}

func TestVLLM_ChatRequestEndToEnd(t *testing.T) {
	capture := &vllmCapture{}
	upstream := newFakeVLLM(t, capture)
	db := newVLLMTestDB()
	prx := newVLLMProxy(t, upstream.URL, db)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		stringsReader(`{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}],"temperature":0.1}`))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("X-OpenWebUI-User-Email", "Ivan.Petrov@example.com")
	req.Header.Set("X-OpenWebUI-User-Id", "S-1-5-21-1-2-3-4")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	path, _, hasAuth, body := capture.snapshot()
	assert.Equal(t, "/v1/chat/completions", path)
	assert.False(t, hasAuth, "a keyless vLLM must not receive an Authorization header")

	// The client asked for the alias; vLLM must be sent the real model, with no
	// LiteLLM provider prefix.
	assert.Equal(t, "qwen-36-35b-fp8", body["model"])
	// Deployment defaults fill in what the client left out ...
	assert.Equal(t, map[string]any{"enable_thinking": false}, body["chat_template_kwargs"])
	assert.EqualValues(t, 20, body["top_k"])
	// ... and never override what the client sent (LiteLLM merges params under kwargs).
	assert.EqualValues(t, 0.1, body["temperature"])

	require.Len(t, db.logs, 1)
	log := db.logs[0]
	assert.Equal(t, "qwen-36-35b-fp8", log.Model, "model is the deployment's real name")
	assert.Equal(t, "qwen-flash", log.ModelGroup, "model_group is the name the client asked for")
	assert.Equal(t, "vllm", log.CustomLLMProvider)
	assert.Equal(t, "S-1-5-21-1-2-3-4", log.UserID, "the user header replaces the key owner in the spend record")
	assert.Equal(t, "Ivan.Petrov@example.com", log.EndUser, "end user keeps the caller's spelling")
	assert.Equal(t, "team-1", log.TeamID)
	assert.Equal(t, "success", log.Status)
}

func TestVLLM_DefaultsApplyWhenClientSendsNothing(t *testing.T) {
	capture := &vllmCapture{}
	upstream := newFakeVLLM(t, capture)
	db := newVLLMTestDB()
	prx := newVLLMProxy(t, upstream.URL, db)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		stringsReader(`{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	_, _, _, body := capture.snapshot()
	assert.EqualValues(t, 0.7, body["temperature"])
	assert.EqualValues(t, 20, body["top_k"])

	require.Len(t, db.logs, 1)
	assert.Equal(t, "key-owner", db.logs[0].UserID, "without identity headers the key owner is recorded")
	assert.Empty(t, db.logs[0].EndUser, "the key owner is never turned into an end user")
}

func TestVLLM_DefaultsNeverReachEmbeddings(t *testing.T) {
	capture := &vllmCapture{}
	upstream := newFakeVLLM(t, capture)
	db := newVLLMTestDB()
	prx := newVLLMProxy(t, upstream.URL, db)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		stringsReader(`{"model":"qwen3-embedding-8b","input":"hello"}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	path, _, _, body := capture.snapshot()
	assert.Equal(t, "/v1/embeddings", path)
	assert.NotContains(t, body, "temperature", "sampling defaults are chat-only")

	require.Len(t, db.logs, 1)
	assert.Equal(t, "qwen3-embedding-8b", db.logs[0].Model)
	assert.Equal(t, "qwen3-embedding-8b", db.logs[0].ModelGroup)
}

// vLLM serves /v1/responses natively, so a Responses request must reach it as-is
// instead of being rewritten into Chat Completions.
func TestVLLM_ResponsesAPIIsPassedThrough(t *testing.T) {
	capture := &vllmCapture{}
	upstream := newFakeVLLM(t, capture)
	db := newVLLMTestDB()
	prx := newVLLMProxy(t, upstream.URL, db)

	req := httptest.NewRequest(http.MethodPost, "/v1/responses",
		stringsReader(`{"model":"qwen-flash","input":"hi"}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	path, _, _, body := capture.snapshot()
	assert.Equal(t, "/v1/responses", path, "the native endpoint is used")
	assert.Equal(t, "hi", body["input"], "the Responses-shaped body is forwarded unchanged")
	assert.NotContains(t, body, "messages")
	assert.Equal(t, "qwen-36-35b-fp8", body["model"], "the real model name replaces the alias")

	require.Len(t, db.logs, 1)
	assert.Equal(t, "qwen-36-35b-fp8", db.logs[0].Model)
	assert.Equal(t, "qwen-flash", db.logs[0].ModelGroup)
	assert.Equal(t, 10, db.logs[0].PromptTokens)
	assert.Equal(t, 5, db.logs[0].CompletionTokens)
}
