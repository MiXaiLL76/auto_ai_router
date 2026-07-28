package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	aimodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type clientAuthTestDB struct {
	litellmdb.Manager
	mu         sync.Mutex
	tokens     map[string]*dbmodels.TokenInfo
	errors     map[string]error
	nilResults map[string]bool
	seen       []string
}

func (m *clientAuthTestDB) IsEnabled() bool                        { return true }
func (m *clientAuthTestDB) IsHealthy() bool                        { return true }
func (m *clientAuthTestDB) LogSpend(*dbmodels.SpendLogEntry) error { return nil }

func (m *clientAuthTestDB) ValidateToken(_ context.Context, rawToken string) (*dbmodels.TokenInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seen = append(m.seen, rawToken)
	if err := m.errors[rawToken]; err != nil {
		return nil, err
	}
	if m.nilResults[rawToken] {
		return nil, nil
	}
	info := m.tokens[rawToken]
	if info == nil {
		return nil, litellmdb.ErrTokenNotFound
	}
	clone := *info
	clone.Models = append([]string(nil), info.Models...)
	clone.UserModels = append([]string(nil), info.UserModels...)
	clone.TeamModels = append([]string(nil), info.TeamModels...)
	clone.TeamMemberModels = append([]string(nil), info.TeamMemberModels...)
	if err := clone.Validate(""); err != nil {
		return nil, err
	}
	return &clone, nil
}

func (m *clientAuthTestDB) seenTokens() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.seen...)
}

func newClientAuthTestProxy(t *testing.T, db litellmdb.Manager, upstreamURL string, provider config.ProviderType, providerKey string) *Proxy {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	staticModels := []config.ModelRPMConfig{
		{Name: "backend-chat", Model: "backend-chat", Credential: "provider", RPM: 100, TPM: 10000},
		{Name: "backend-embed", Model: "backend-embed", Credential: "provider", RPM: 100, TPM: 10000},
	}
	manager := aimodels.New(logger, 100, staticModels)
	manager.SetModelAliases(map[string]string{
		"public/chat":         "backend-chat",
		"public/chat-premium": "backend-chat",
		"public/embed":        "backend-embed",
	})
	credentials := []config.CredentialConfig{{
		Name: "provider", Type: provider, BaseURL: upstreamURL, APIKey: providerKey, RPM: 100, TPM: 10000,
	}}
	manager.LoadModelsFromConfig(credentials)
	builder := NewTestProxyBuilder().
		WithCredentials(credentials...).
		WithMasterKey("master-key")
	builder.config.ModelManager = manager
	prx := builder.Build()
	prx.LiteLLMDB = db
	return prx
}

func assertAuthErrorShape(t *testing.T, recorder *httptest.ResponseRecorder, status int) {
	t.Helper()
	require.Equal(t, status, recorder.Code)
	var response APIErrorResponse
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
	assert.Equal(t, errorTypeForStatus(status), response.Error.Type)
}

func TestClientAuthenticationRejectsInvalidCredentialsWithoutChangingErrorContract(t *testing.T) {
	db := &clientAuthTestDB{
		tokens: map[string]*dbmodels.TokenInfo{"valid-x-key": {Token: "hash"}},
		errors: map[string]error{
			"invalid-key": litellmdb.ErrTokenNotFound,
			"expired-key": litellmdb.ErrTokenExpired,
			"blocked-key": litellmdb.ErrTokenBlocked,
		},
	}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")

	tests := []struct {
		name       string
		headers    map[string]string
		wantStatus int
		wantText   string
	}{
		{name: "missing", wantStatus: http.StatusUnauthorized, wantText: "Missing Authorization header"},
		{name: "empty x api key", headers: map[string]string{"x-api-key": ""}, wantStatus: http.StatusUnauthorized, wantText: "Missing Authorization header"},
		{name: "malformed bearer", headers: map[string]string{"Authorization": "Basic invalid"}, wantStatus: http.StatusUnauthorized, wantText: "Invalid Authorization header format"},
		{name: "invalid bearer", headers: map[string]string{"Authorization": "Bearer invalid-key"}, wantStatus: http.StatusUnauthorized, wantText: "Invalid token"},
		{name: "invalid x api key", headers: map[string]string{"x-api-key": "invalid-key"}, wantStatus: http.StatusUnauthorized, wantText: "Invalid token"},
		{name: "expired bearer", headers: map[string]string{"Authorization": "Bearer expired-key"}, wantStatus: http.StatusUnauthorized, wantText: "Token expired"},
		{name: "blocked bearer preserves forbidden contract", headers: map[string]string{"Authorization": "Bearer blocked-key"}, wantStatus: http.StatusForbidden, wantText: "Token blocked"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			for key, value := range tt.headers {
				req.Header.Set(key, value)
			}
			w := httptest.NewRecorder()
			logCtx := &RequestLogContext{Request: req}

			ok := prx.authenticateRequest(w, req, logCtx, true)

			assert.False(t, ok)
			assertAuthErrorShape(t, w, tt.wantStatus)
			assert.Contains(t, w.Body.String(), tt.wantText)
		})
	}
}

func TestAIRProxyMarkerRequiresMasterKeyPeer(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse(
			"chatcmpl-air-marker",
			"backend-chat",
			"ok",
		))
	}))
	defer upstream.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"virtual-key": {Token: "hash"},
	}}
	prx := newClientAuthTestProxy(t, db, upstream.URL, config.ProviderTypeOpenAI, "provider-key")

	for _, tt := range []struct {
		name       string
		token      string
		header     string
		wantHeader bool
	}{
		{name: "external virtual key cannot spoof marker", token: "virtual-key", header: HeaderAIRProxyClient},
		{name: "external virtual key cannot spoof legacy marker", token: "virtual-key", header: HeaderLegacyAIRProxyClient},
		{name: "master key accepts current marker", token: "master-key", header: HeaderAIRProxyClient, wantHeader: true},
		{name: "master key accepts legacy marker", token: "master-key", header: HeaderLegacyAIRProxyClient, wantHeader: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"public/chat","messages":[{"role":"user","content":"hi"}]}`,
			))
			req.Header.Set("Authorization", "Bearer "+tt.token)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(tt.header, "1")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			if tt.wantHeader {
				assert.Equal(t, "provider", w.Header().Get("X-Credential-Name"))
			} else {
				assert.Empty(t, w.Header().Get("X-Credential-Name"))
			}
		})
	}
}

func TestAIRCredentialSendsCurrentAndLegacyProxyMarkers(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "1", r.Header.Get(HeaderAIRProxyClient))
		assert.Equal(t, "1", r.Header.Get(HeaderLegacyAIRProxyClient))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse(
			"chatcmpl-air-peer",
			"public/chat",
			"ok",
		))
	}))
	defer upstream.Close()

	prx := newClientAuthTestProxy(t, &clientAuthTestDB{}, upstream.URL, config.ProviderTypeAIR, "remote-master-key")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public/chat","messages":[{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
}

func TestDirectProviderRestoresClientVisibleModel(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse(
			"chatcmpl-direct-alias",
			"backend-chat",
			"ok",
		))
	}))
	defer upstream.Close()

	prx := newClientAuthTestProxy(t, nil, upstream.URL, config.ProviderTypeOpenAI, "provider-key")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public/chat","messages":[{"role":"user","content":"hello"}]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	var response map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
	assert.Equal(t, "public/chat", response["model"])
}

func TestDirectProviderStreamingDisablesProxyBuffering(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chatcmpl-direct-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"backend-chat\",\"choices\":[]}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer upstream.Close()

	prx := newClientAuthTestProxy(t, nil, upstream.URL, config.ProviderTypeOpenAI, "provider-key")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public/chat","messages":[{"role":"user","content":"hello"}],"stream":true}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "no", recorder.Header().Get(accelBufferingHeader))
}

func TestExplicitClientModelSurfaceRejectsBackendBeforeProviderAndSpend(t *testing.T) {
	var providerCalls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		providerCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-internal","object":"chat.completion","created":1,"model":"backend-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer provider.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"unrestricted-key": {Token: "unrestricted-hash"},
	}}
	prx := newClientAuthTestProxy(t, db, provider.URL, config.ProviderTypeOpenAI, "provider-key")
	prx.modelManager.SetClientModelIDs([]string{"public/chat", "public/embed"})

	rejected := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"backend-chat","messages":[{"role":"user","content":"must not route"}]}`,
	))
	rejected.Header.Set("Authorization", "Bearer unrestricted-key")
	rejected.Header.Set("Content-Type", "application/json")
	rejectedRecorder := httptest.NewRecorder()
	prx.ProxyRequest(rejectedRecorder, rejected)

	testhelpers.AssertJSONErrorResponse(t, rejectedRecorder, http.StatusNotFound, "not_found_error", "Model backend-chat not found")
	assert.Zero(t, providerCalls.Load(), "backend-only ID reached the provider")

	internal := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"backend-chat","messages":[{"role":"user","content":"trusted internal hop"}]}`,
	))
	internal.Header.Set("Authorization", "Bearer master-key")
	internal.Header.Set("Content-Type", "application/json")
	internalRecorder := httptest.NewRecorder()
	prx.ProxyRequest(internalRecorder, internal)

	assert.Equal(t, http.StatusOK, internalRecorder.Code)
	assert.Equal(t, int32(1), providerCalls.Load(), "master-key internal routing must remain available")
}

func TestAcceptedModelAliasPreservesRestrictedTokenRouting(t *testing.T) {
	var providerModel string
	provider := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		providerModel, _ = body["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-accepted","object":"chat.completion","created":1,"model":"backend-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer provider.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"restricted-key": {
			Token:  "restricted-hash",
			Models: []string{"public/chat"},
		},
	}}
	prx := newClientAuthTestProxy(t, db, provider.URL, config.ProviderTypeOpenAI, "provider-key")
	prx.modelManager.SetClientModelIDs([]string{"public/chat"})
	prx.modelManager.SetAcceptedModelAliases(map[string]string{"legacy-chat": "public/chat"})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"legacy-chat","messages":[{"role":"user","content":"hello"}]}`,
	))
	req.Header.Set("Authorization", "Bearer restricted-key")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
	assert.Equal(t, "backend-chat", providerModel)
}

func TestExplicitClientModelSurfacePreservesRestrictedACLPrecedenceForUnknownModel(t *testing.T) {
	var providerCalls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		providerCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"unexpected-provider-call"}`))
	}))
	defer provider.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"restricted-key": {
			Token:  "restricted-hash",
			Models: []string{"public/chat"},
		},
		"unrestricted-key": {
			Token: "unrestricted-hash",
		},
	}}
	prx := newClientAuthTestProxy(t, db, provider.URL, config.ProviderTypeOpenAI, "provider-key")
	prx.modelManager.SetClientModelIDs([]string{"public/chat", "public/embed"})

	request := func(key string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
			`{"model":"unknown-client-model","messages":[{"role":"user","content":"hello"}]}`,
		))
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		writer := httptest.NewRecorder()
		prx.ProxyRequest(writer, req)
		return writer
	}

	restricted := request("restricted-key")
	assertAuthErrorShape(t, restricted, http.StatusForbidden)
	assert.Contains(t, restricted.Body.String(), "Model not allowed")

	unrestricted := request("unrestricted-key")
	testhelpers.AssertJSONErrorResponse(t, unrestricted, http.StatusNotFound, "not_found_error", "Model unknown-client-model not found")

	trusted := request("master-key")
	testhelpers.AssertJSONErrorResponse(t, trusted, http.StatusNotFound, "not_found_error", "Model unknown-client-model not found")
	assert.Zero(t, providerCalls.Load())
}

func TestClientAuthenticationAuthorizationTakesPrecedenceOverXAPIKey(t *testing.T) {
	db := &clientAuthTestDB{
		tokens: map[string]*dbmodels.TokenInfo{"valid-x-key": {Token: "hash"}},
		errors: map[string]error{"invalid-bearer": litellmdb.ErrTokenNotFound},
	}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer invalid-bearer")
	req.Header.Set("x-api-key", "valid-x-key")
	w := httptest.NewRecorder()

	ok := prx.authenticateRequest(w, req, &RequestLogContext{Request: req}, true)

	assert.False(t, ok)
	assertAuthErrorShape(t, w, http.StatusUnauthorized)
	assert.Equal(t, []string{"invalid-bearer"}, db.seenTokens())
}

func TestAuthenticateClientRequestScopedIsTransportIndependent(t *testing.T) {
	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"tenant-key": {Token: "tenant-hash", KeyName: "team-a"},
	}}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")

	var scopeKeys []string
	for _, headers := range []map[string]string{
		{"Authorization": "Bearer tenant-key"},
		{"x-api-key": "tenant-key"},
	} {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		for key, value := range headers {
			req.Header.Set(key, value)
		}
		w := httptest.NewRecorder()
		info, visibility, ok := prx.AuthenticateClientRequestScoped(w, req)
		require.True(t, ok)
		require.NotNil(t, info)
		assert.Equal(t, "team-a", info.KeyName)
		scopeKeys = append(scopeKeys, visibility.Key())
	}

	require.Len(t, scopeKeys, 2)
	assert.Equal(t, scopeKeys[0], scopeKeys[1])
	assert.NotEqual(t, scope.PublicContext().Key(), scopeKeys[0])
}

func TestClientAuthenticationRejectsAmbiguousHeadersAndNormalizesOWS(t *testing.T) {
	tests := []struct {
		name       string
		headers    http.Header
		wantOK     bool
		wantSeen   []string
		wantStatus int
		wantText   string
	}{
		{
			name: "lowercase authorization remains authoritative",
			headers: http.Header{
				"authorization": {"Bearer invalid-bearer"},
				"X-Api-Key":     {"valid-x-key"},
			},
			wantSeen:   []string{"invalid-bearer"},
			wantStatus: http.StatusUnauthorized,
			wantText:   "Invalid token",
		},
		{
			name: "duplicate authorization is rejected before validation",
			headers: http.Header{
				"Authorization": {"Bearer valid-x-key", "Bearer valid-x-key"},
			},
			wantStatus: http.StatusUnauthorized,
			wantText:   "Invalid Authorization header format",
		},
		{
			name: "mixed case duplicate authorization is rejected",
			headers: http.Header{
				"Authorization": {"Bearer valid-x-key"},
				"authorization": {"Bearer valid-x-key"},
			},
			wantStatus: http.StatusUnauthorized,
			wantText:   "Invalid Authorization header format",
		},
		{
			name: "duplicate x api key is rejected before validation",
			headers: http.Header{
				"X-Api-Key": {"valid-x-key", "valid-x-key"},
			},
			wantStatus: http.StatusUnauthorized,
			wantText:   "Invalid Authorization header format",
		},
		{
			name: "empty authoritative authorization does not fall back",
			headers: http.Header{
				"Authorization": {"   "},
				"X-Api-Key":     {"valid-x-key"},
			},
			wantStatus: http.StatusUnauthorized,
			wantText:   "Missing Authorization header",
		},
		{
			name: "bearer scheme and optional whitespace are normalized",
			headers: http.Header{
				"Authorization": {" \tbeArEr   valid-x-key\t "},
			},
			wantOK:   true,
			wantSeen: []string{"valid-x-key"},
		},
		{
			name: "x api key optional whitespace is normalized",
			headers: http.Header{
				"X-Api-Key": {" \tvalid-x-key\t "},
			},
			wantOK:   true,
			wantSeen: []string{"valid-x-key"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db := &clientAuthTestDB{
				tokens: map[string]*dbmodels.TokenInfo{"valid-x-key": {Token: "hash"}},
				errors: map[string]error{"invalid-bearer": litellmdb.ErrTokenNotFound},
			}
			prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			req.Header = tt.headers.Clone()
			w := httptest.NewRecorder()

			ok := prx.authenticateRequest(w, req, &RequestLogContext{Request: req}, true)

			assert.Equal(t, tt.wantOK, ok)
			assert.Equal(t, tt.wantSeen, db.seenTokens())
			if !tt.wantOK {
				assertAuthErrorShape(t, w, tt.wantStatus)
				assert.Contains(t, w.Body.String(), tt.wantText)
			}
		})
	}
}

func TestClientAuthenticationFailsClosedOnNilValidationResult(t *testing.T) {
	db := &clientAuthTestDB{nilResults: map[string]bool{"nil-result": true}}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer nil-result")
	w := httptest.NewRecorder()

	ok := prx.authenticateRequest(w, req, &RequestLogContext{Request: req}, true)

	assert.False(t, ok)
	assertAuthErrorShape(t, w, http.StatusUnauthorized)
	assert.Contains(t, w.Body.String(), "Invalid token")
}

func TestInferenceModelACLIntersectsParentScopesBeforeAliasResolution(t *testing.T) {
	var upstreamCalls int
	var upstreamModel string
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		var body map[string]interface{}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		upstreamModel, _ = body["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/embeddings" {
			_, _ = w.Write([]byte(`{"object":"list","data":[],"model":"backend-embed","usage":{"prompt_tokens":1,"total_tokens":1}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-auth","object":"chat.completion","created":1,"model":"backend-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"restricted-key": {
			Token:      "restricted-hash",
			Models:     []string{"public/chat", "backend-chat"},
			TeamID:     "team-alt",
			TeamModels: []string{"public/chat"},
		},
	}}
	prx := newClientAuthTestProxy(t, db, upstream.URL, config.ProviderTypeOpenAI, "provider-key")

	allowed := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public/chat","messages":[{"role":"user","content":"hello"}]}`,
	))
	allowed.Header.Set("x-api-key", "restricted-key")
	allowed.Header.Set("Content-Type", "application/json")
	allowedWriter := httptest.NewRecorder()
	prx.ProxyRequest(allowedWriter, allowed)

	require.Equal(t, http.StatusOK, allowedWriter.Code)
	assert.Equal(t, "backend-chat", upstreamModel, "routing must resolve the alias only after ACL")
	assert.Equal(t, 1, upstreamCalls)

	// The child key explicitly includes the internal target, but its parent team
	// and project do not. LiteLLM treats the applicable scopes as an intersection,
	// so AIR must reject the target before its internal alias routing can widen
	// the caller's effective permission.
	equivalentTarget := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"backend-chat","messages":[{"role":"user","content":"hello"}]}`,
	))
	equivalentTarget.Header.Set("x-api-key", "restricted-key")
	equivalentTarget.Header.Set("Content-Type", "application/json")
	equivalentTargetWriter := httptest.NewRecorder()
	prx.ProxyRequest(equivalentTargetWriter, equivalentTarget)

	assertAuthErrorShape(t, equivalentTargetWriter, http.StatusForbidden)
	assert.Equal(t, 1, upstreamCalls)

	// Two public products may intentionally share a provider target. A scope
	// for one alias must not grant or advertise its sibling alias implicitly.
	siblingAlias := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public/chat-premium","messages":[{"role":"user","content":"hello"}]}`,
	))
	siblingAlias.Header.Set("x-api-key", "restricted-key")
	siblingAlias.Header.Set("Content-Type", "application/json")
	siblingAliasWriter := httptest.NewRecorder()
	prx.ProxyRequest(siblingAliasWriter, siblingAlias)

	assertAuthErrorShape(t, siblingAliasWriter, http.StatusForbidden)
	assert.Equal(t, 1, upstreamCalls)

	disallowed := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(
		`{"model":"public/embed","input":"hello"}`,
	))
	disallowed.Header.Set("Authorization", "Bearer restricted-key")
	disallowed.Header.Set("Content-Type", "application/json")
	disallowedWriter := httptest.NewRecorder()
	prx.ProxyRequest(disallowedWriter, disallowed)

	assertAuthErrorShape(t, disallowedWriter, http.StatusForbidden)
	assert.Contains(t, disallowedWriter.Body.String(), "Model not allowed")
	assert.Equal(t, 1, upstreamCalls, "a disallowed model must be rejected before provider dispatch")
}

func TestInferenceAllTeamModelsWithoutTeamPolicy(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-auth","object":"chat.completion","created":1,"model":"backend-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	for _, tt := range []struct {
		name       string
		strict     bool
		wantStatus int
	}{
		{name: "compatibility", wantStatus: http.StatusOK},
		{name: "strict", strict: true, wantStatus: http.StatusForbidden},
	} {
		t.Run(tt.name, func(t *testing.T) {
			db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
				"gateway-key": {
					Token:  "gateway-hash",
					Models: []string{dbmodels.AllTeamModels},
				},
			}}
			prx := newClientAuthTestProxy(t, db, upstream.URL, config.ProviderTypeOpenAI, "provider-key")
			prx.strictAllTeamModelsACL = tt.strict
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"backend-chat","messages":[{"role":"user","content":"hello"}]}`,
			))
			req.Header.Set("Authorization", "Bearer gateway-key")
			req.Header.Set("Content-Type", "application/json")
			writer := httptest.NewRecorder()

			prx.ProxyRequest(writer, req)

			assert.Equal(t, tt.wantStatus, writer.Code)
		})
	}
}

func TestInferenceRejectsBlockedParentsAndNoDefaultPersonalUserBeforeProvider(t *testing.T) {
	var upstreamCalls int
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"unexpected"}`))
	}))
	defer upstream.Close()

	blocked := true
	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"blocked-team": {
			Token:       "blocked-team-hash",
			Models:      []string{"public/chat"},
			TeamID:      "team",
			TeamModels:  []string{"public/chat"},
			TeamBlocked: &blocked,
		},
		"no-default-user": {
			Token:      "no-default-user-hash",
			Models:     []string{"public/chat"},
			UserID:     "personal-user",
			UserModels: []string{dbmodels.NoDefaultModels, "public/chat"},
		},
	}}
	prx := newClientAuthTestProxy(t, db, upstream.URL, config.ProviderTypeOpenAI, "provider-key")

	for _, key := range []string{"blocked-team", "no-default-user"} {
		t.Run(key, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"public/chat","messages":[{"role":"user","content":"hello"}]}`,
			))
			req.Header.Set("Authorization", "Bearer "+key)
			req.Header.Set("Content-Type", "application/json")
			writer := httptest.NewRecorder()

			prx.ProxyRequest(writer, req)

			assertAuthErrorShape(t, writer, http.StatusForbidden)
		})
	}
	assert.Equal(t, 0, upstreamCalls)
}

func TestInferenceModelACLWildcardTreatsOnlyStarAsSyntax(t *testing.T) {
	var upstreamCalls int
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		var body map[string]interface{}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-wildcard","object":"chat.completion","created":1,"model":"backend-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"wildcard-allowed": {
			Token:  "wildcard-allowed-hash",
			Models: []string{"public/*"},
		},
		"regex-looking-denied": {
			Token:  "regex-looking-denied-hash",
			Models: []string{"public/ch.t*"},
		},
	}}
	prx := newClientAuthTestProxy(t, db, upstream.URL, config.ProviderTypeOpenAI, "provider-key")

	request := func(key string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
			`{"model":"public/chat","messages":[{"role":"user","content":"hello"}]}`,
		))
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		writer := httptest.NewRecorder()
		prx.ProxyRequest(writer, req)
		return writer
	}

	assert.Equal(t, http.StatusOK, request("wildcard-allowed").Code)
	assertAuthErrorShape(t, request("regex-looking-denied"), http.StatusForbidden)
	assert.Equal(t, 1, upstreamCalls)
}

func TestInferenceModelACLRejectsBeforeRoutingAcrossPublicEndpoints(t *testing.T) {
	prx := newClientAuthTestProxy(t, nil, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	type requestFixture struct {
		path        string
		contentType string
		body        []byte
	}
	fixtures := []requestFixture{
		{path: "/v1/chat/completions", contentType: "application/json", body: []byte(`{"model":"public/blocked","messages":[{"role":"user","content":"hello"}]}`)},
		{path: "/v1/completions", contentType: "application/json", body: []byte(`{"model":"public/blocked","prompt":"hello"}`)},
		{path: "/v1/embeddings", contentType: "application/json", body: []byte(`{"model":"public/blocked","input":"hello"}`)},
		{path: "/v1/images/generations", contentType: "application/json", body: []byte(`{"model":"public/blocked","prompt":"hello"}`)},
		{path: "/v1/responses", contentType: "application/json", body: []byte(`{"model":"public/blocked","input":"hello"}`)},
	}

	var multipartBody bytes.Buffer
	multipartWriter := multipart.NewWriter(&multipartBody)
	require.NoError(t, multipartWriter.WriteField("model", "public/blocked"))
	imagePart, err := multipartWriter.CreateFormFile("image", "input.png")
	require.NoError(t, err)
	_, err = imagePart.Write([]byte("not-a-real-image"))
	require.NoError(t, err)
	require.NoError(t, multipartWriter.Close())
	fixtures = append(fixtures, requestFixture{
		path:        "/v1/images/edits",
		contentType: multipartWriter.FormDataContentType(),
		body:        multipartBody.Bytes(),
	})

	for _, fixture := range fixtures {
		t.Run(fixture.path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, fixture.path, bytes.NewReader(fixture.body))
			req.Header.Set("Content-Type", fixture.contentType)
			w := httptest.NewRecorder()
			logCtx := &RequestLogContext{
				Request:   req,
				TokenInfo: &dbmodels.TokenInfo{Models: []string{"public/allowed"}},
			}

			_, _, _, _, ok := prx.readRequestBodyAndSelectModel(w, req, logCtx)

			assert.False(t, ok)
			assertAuthErrorShape(t, w, http.StatusForbidden)
			assert.Equal(t, "public/blocked", logCtx.PublicModelID)
			assert.Equal(t, "public/blocked", logCtx.ModelID, "ACL must run before alias resolution")
		})
	}
}

func TestXAPIKeyAuthenticatesAndClientCredentialsNeverReachUpstream(t *testing.T) {
	tests := []struct {
		name             string
		provider         config.ProviderType
		providerKey      string
		clientHeader     string
		clientCredential string
		wantProviderAuth string
	}{
		{name: "direct provider x api key", provider: config.ProviderTypeOpenAI, providerKey: "provider-key", clientHeader: "x-api-key", clientCredential: "master-key", wantProviderAuth: "Bearer provider-key"},
		{name: "proxy provider x api key", provider: config.ProviderTypeProxy, providerKey: "provider-key", clientHeader: "x-api-key", clientCredential: "master-key", wantProviderAuth: "Bearer provider-key"},
		{name: "proxy without provider key does not forward client bearer", provider: config.ProviderTypeProxy, clientHeader: "Authorization", clientCredential: "Bearer master-key"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotAuthorization string
			var gotXAPIKey string
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotAuthorization = r.Header.Get("Authorization")
				gotXAPIKey = r.Header.Get("x-api-key")
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"chatcmpl-header","object":"chat.completion","created":1,"model":"backend-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
			}))
			defer upstream.Close()

			prx := newClientAuthTestProxy(t, nil, upstream.URL, tt.provider, tt.providerKey)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"public/chat","messages":[{"role":"user","content":"hello"}]}`,
			))
			req.Header.Set(tt.clientHeader, tt.clientCredential)
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			assert.Equal(t, tt.wantProviderAuth, gotAuthorization)
			assert.Empty(t, gotXAPIKey)
		})
	}
}
