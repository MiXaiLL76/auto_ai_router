package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type organizationPolicyTestDB struct {
	litellmdb.NoopManager
	tokens map[string]*dbmodels.TokenInfo
	logs   []*dbmodels.SpendLogEntry
}

func (d *organizationPolicyTestDB) IsEnabled() bool           { return true }
func (d *organizationPolicyTestDB) IsHealthy() bool           { return true }
func (d *organizationPolicyTestDB) SpendLoggingEnabled() bool { return true }

func (d *organizationPolicyTestDB) ValidateToken(_ context.Context, rawToken string) (*dbmodels.TokenInfo, error) {
	info := d.tokens[rawToken]
	if info == nil {
		return nil, litellmdb.ErrTokenNotFound
	}
	return info.Clone(), nil
}

func (d *organizationPolicyTestDB) LogSpend(entry *dbmodels.SpendLogEntry) error {
	d.logs = append(d.logs, entry)
	return nil
}

func writeProxyPolicyPrices(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(path, []byte(body), 0600))
	return path
}

func newOrganizationPolicyProxy(t *testing.T, upstreamURL string, db *organizationPolicyTestDB, policies []config.OrganizationPolicyConfig) *Proxy {
	t.Helper()
	credential := config.CredentialConfig{
		Name: "provider", Type: config.ProviderTypeOpenAI, BaseURL: upstreamURL, APIKey: "provider-key", RPM: 100, TPM: 10000,
	}
	return newOrganizationPolicyProxyWithCredential(t, db, policies, credential)
}

func newOrganizationPolicyProxyWithCredential(
	t *testing.T,
	db *organizationPolicyTestDB,
	policies []config.OrganizationPolicyConfig,
	credential config.CredentialConfig,
) *Proxy {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "route-a", Credential: credential.Name},
		{Name: "route-b", Credential: credential.Name},
	})
	manager.SetModelAliases(map[string]string{"public/shared": "route-a"})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})
	registry, err := routermodels.LoadOrganizationPolicies(policies, manager, routermodels.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: false,
	})
	require.NoError(t, err)

	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	builder.config.OrganizationPolicies = registry
	prx := builder.Build()
	prx.LiteLLMDB = db
	prx.priceRegistry = routermodels.NewModelPriceRegistry()
	prx.priceRegistry.ReplaceFilePrices(map[string]*routermodels.ModelPrice{
		"route-a": {InputCostPerToken: 0.001},
		"route-b": {InputCostPerToken: 0.001},
	})
	return prx
}

func TestOrganizationPolicy_ShadowMappingUsesExactPriceAndMetadata(t *testing.T) {
	var calls atomic.Int32
	var upstreamModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		upstreamModel, _ = body["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"route-b","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", OrganizationID: "org-1"},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.001,"output_cost_per_token":0.002}}`),
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}})
	// The exact organization tariff suffices without any default-registry prices.
	prx.priceRegistry.ReplaceFilePrices(nil)
	listed, err := prx.ListModelsForToken(db.tokens["token"], scope.PublicContext())
	require.NoError(t, err)
	require.Len(t, listed.Data, 1)
	assert.Equal(t, "public/shared", listed.Data[0].ID)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), calls.Load())
	assert.Equal(t, "route-b", upstreamModel)
	require.Len(t, db.logs, 1)
	log := db.logs[0]
	assert.InDelta(t, 0.02, log.Spend, 1e-9)
	assert.Equal(t, "route-b", log.Model)
	assert.Equal(t, "route-b", log.ModelGroup)
	assert.Equal(t, "provider:route-b", log.ModelID)
	assert.Equal(t, "org-1", log.OrganizationID)
	var metadata map[string]any
	require.NoError(t, json.Unmarshal([]byte(log.Metadata), &metadata))
	spendMetadata := metadata["spend_logs_metadata"].(map[string]any)
	assert.Equal(t, "public/shared", spendMetadata["public_model_name"])
	assert.Equal(t, "route-b", spendMetadata["canonical_model_name"])
	assert.Equal(t, "profile-1", spendMetadata["billing_profile_id"])
	assert.Equal(t, "public/shared", spendMetadata["billing_price_model_name"])
	assert.Equal(t, "org-1", spendMetadata["billing_organization_id"])
	assert.NotEmpty(t, spendMetadata["billing_profile_sha256"])
}

func TestOrganizationPolicy_MissingExactPriceRejectsBeforeProvider(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", OrganizationID: "org-1"},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"route-b":{"input_cost_per_token":0.001}}`),
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
	assert.Equal(t, int32(0), calls.Load())
	assert.Empty(t, db.logs)
}

func TestOrganizationPolicy_MissingExactPriceIsAbsentFromListing(t *testing.T) {
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{}}
	prx := newOrganizationPolicyProxy(t, "http://provider.invalid", db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"route-b":{"input_cost_per_token":0.001}}`),
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}})
	token := &dbmodels.TokenInfo{DirectOrganizationID: "org-1", OrganizationID: "org-1"}

	response, err := prx.ListModelsForToken(token, scope.PublicContext())
	require.NoError(t, err)
	for _, model := range response.Data {
		assert.NotEqual(t, "public/shared", model.ID)
	}
}

func TestOrganizationPolicy_ACLDenialPrecedesMissingPrice(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", OrganizationID: "org-1", Models: []string{"other"}},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"route-b":{"input_cost_per_token":0.001}}`),
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}})
	prx.strictAllTeamModelsACL = true

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Equal(t, int32(0), calls.Load())
	assert.Empty(t, db.logs)
}

func TestOrganizationPolicy_DanglingConfiguredIdentityRejectsBeforeProvider(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", DirectOrganizationDangling: true},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.001}}`),
	}})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Equal(t, int32(0), calls.Load())
	assert.Empty(t, db.logs)
}

func TestOrganizationPolicy_TeamFallbackSetsAccountingOrganization(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"route-a","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", TeamID: "team-1", TeamOrganizationID: "org-team"},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-team",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.001}}`),
	}})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, db.logs, 1)
	assert.Equal(t, "org-team", db.logs[0].OrganizationID)
}

func TestOrganizationPolicy_DirectOrganizationTakesPrecedenceOverTeam(t *testing.T) {
	var upstreamModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		upstreamModel, _ = body["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"route-b","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {
			Token:                "token-hash",
			UserID:               "user-1",
			TeamID:               "team-1",
			DirectOrganizationID: "org-direct",
			OrganizationID:       "org-direct",
			TeamOrganizationID:   "org-team",
		},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{
		{
			OrganizationID:  "org-direct",
			PriceProfileID:  "profile-direct",
			ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.001}}`),
			ModelMappings:   map[string]string{"public/shared": "route-b"},
		},
		{
			OrganizationID:  "org-team",
			PriceProfileID:  "profile-team",
			ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.002}}`),
			ModelMappings:   map[string]string{"public/shared": "route-a"},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "route-b", upstreamModel)
	require.Len(t, db.logs, 1)
	assert.Equal(t, "org-direct", db.logs[0].OrganizationID)
}

func TestOrganizationPolicy_AddsCandidateSetACL(t *testing.T) {
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", OrganizationID: "org-1", Models: []string{"public/shared"}},
	}}
	prx := newOrganizationPolicyProxy(t, "http://provider.example", db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.001}}`),
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}})
	prx.strictAllTeamModelsACL = true

	assert.True(t, prx.isAnyModelAllowedForToken(db.tokens["token"], []string{"public/shared", "route-b", "route-b"}))
	assert.False(t, prx.isAnyModelAllowedForToken(db.tokens["token"], []string{"route-b", "route-b", "route-b"}))
}

func TestOrganizationPolicy_FrozenPriceSurvivesDefaultRegistryChange(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	policy := &routermodels.OrganizationPolicy{
		OrganizationID: "org-1", PriceProfileID: "profile-1", ProfileSHA256: "sha",
	}
	price := &routermodels.ModelPrice{InputCostPerToken: 0.001}
	logCtx := &RequestLogContext{
		OrganizationPolicy:   policy,
		PublicModelID:        "public/shared",
		ModelID:              "route-b",
		RealModelID:          "route-b",
		PriceModelID:         "public/shared",
		ModelPrice:           price,
		billingPriceResolved: true,
		billingPriceModelID:  "public/shared",
		billingPrice:         price,
	}
	registry := routermodels.NewModelPriceRegistry()
	registry.ReplaceFilePrices(map[string]*routermodels.ModelPrice{"route-b": {InputCostPerToken: 9}})
	prx.priceRegistry = registry

	resolvedID, resolved := prx.resolveRetryBillingPrice(logCtx, "public/shared", "route-b", "route-b")

	assert.Equal(t, "public/shared", resolvedID)
	assert.Same(t, price, resolved)
}

func TestOrganizationPolicy_InvisibleTargetRejectsBeforePriceAndProvider(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {
			Token:                "token-hash",
			UserID:               "user-1",
			KeyName:              "scope-b",
			DirectOrganizationID: "org-1",
			OrganizationID:       "org-1",
		},
	}}
	credential := config.CredentialConfig{
		Name: "provider", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "provider-key",
		RPM: 100, TPM: 10000, Scopes: []string{"scope-a"},
	}
	prx := newOrganizationPolicyProxyWithCredential(t, db, []config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.001}}`),
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}}, credential)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
	assert.Equal(t, int32(0), calls.Load())
	assert.Empty(t, db.logs)
}

func stringsReader(value string) *strings.Reader {
	return strings.NewReader(value)
}
