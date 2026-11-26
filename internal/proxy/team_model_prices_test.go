package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// noSpendTrackingDB is an organizationPolicyTestDB that writes no spend logs.
type noSpendTrackingDB struct{ *organizationPolicyTestDB }

func (noSpendTrackingDB) SpendLoggingEnabled() bool { return false }

// teamPricesUpstream answers every chat completion with 10 prompt and 5
// completion tokens and counts the calls.
func teamPricesUpstream(t *testing.T, calls *atomic.Int32) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"route-a","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	t.Cleanup(upstream.Close)
	return upstream
}

// teamPricesProxy prices route-a and route-b at the price list's 0.001/0.002
// per token; the token's team adds a 50% cost margin.
func teamPricesProxy(t *testing.T, upstreamURL string, teamModelPrices string, policies ...config.OrganizationPolicyConfig) (*Proxy, *organizationPolicyTestDB) {
	t.Helper()
	info := &dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", TeamID: "team-1",
		CostMarginConfigs: []dbmodels.CostMarginConfig{{"global": {Percentage: 0.5}}},
	}
	if teamModelPrices != "" {
		info.TeamModelPrices = json.RawMessage(teamModelPrices)
	}
	if len(policies) > 0 {
		info.TeamOrganizationID = policies[0].OrganizationID
		info.OrganizationID = policies[0].OrganizationID
	}
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": info}}
	prx := newOrganizationPolicyProxy(t, upstreamURL, db, policies)
	registry := routermodels.NewModelPriceRegistry()
	registry.Update(map[string]*routermodels.ModelPrice{
		"route-a": {InputCostPerToken: 0.001, OutputCostPerToken: 0.002},
		"route-b": {InputCostPerToken: 0.001, OutputCostPerToken: 0.002},
	})
	prx.priceRegistry = registry
	return prx, db
}

func sendTeamPricesRequest(prx *Proxy, model string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

func spendLogsMetadata(t *testing.T, entry *dbmodels.SpendLogEntry) map[string]any {
	t.Helper()
	var metadata map[string]any
	require.NoError(t, json.Unmarshal([]byte(entry.Metadata), &metadata))
	return metadata["spend_logs_metadata"].(map[string]any)
}

func TestTeamModelPrices_BillsTheTeamPriceWithoutMargin(t *testing.T) {
	var calls atomic.Int32
	prx, db := teamPricesProxy(t, teamPricesUpstream(t, &calls).URL,
		`{"public/shared":{"input_cost_per_token":0.003,"output_cost_per_token":0.004}}`)

	w := sendTeamPricesRequest(prx, "public/shared")
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, db.logs, 1)
	assert.InDelta(t, 10*0.003+5*0.004, db.logs[0].Spend, 1e-12)
	assert.Equal(t, "team_model_prices", spendLogsMetadata(t, db.logs[0])["billing_price_source"])

	// A model without a team row: the price list and the margin.
	w = sendTeamPricesRequest(prx, "route-b")
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, db.logs, 2)
	assert.InDelta(t, (10*0.001+5*0.002)*1.5, db.logs[1].Spend, 1e-12)
	assert.NotContains(t, spendLogsMetadata(t, db.logs[1]), "billing_price_source")
	assert.Equal(t, int32(2), calls.Load())
}

func TestTeamModelPrices_WithoutRowsBillsThePriceList(t *testing.T) {
	var calls atomic.Int32
	prx, db := teamPricesProxy(t, teamPricesUpstream(t, &calls).URL, `{}`)

	w := sendTeamPricesRequest(prx, "public/shared")
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, db.logs, 1)
	assert.InDelta(t, (10*0.001+5*0.002)*1.5, db.logs[0].Spend, 1e-12)
	assert.NotContains(t, spendLogsMetadata(t, db.logs[0]), "billing_price_source")
}

// Any error in the field refuses every request of the team before the
// provider is called, whether spend is tracked or not: the team is never
// billed at the price list instead of its own prices.
func TestTeamModelPrices_UnusableFieldRejectsEveryTeamRequest(t *testing.T) {
	field := `{"public/shared":{"input_cost_per_token":-1}}`
	for _, spendTracking := range []bool{true, false} {
		t.Run("spend tracking "+strconv.FormatBool(spendTracking), func(t *testing.T) {
			var calls atomic.Int32
			prx, db := teamPricesProxy(t, teamPricesUpstream(t, &calls).URL, field)
			if !spendTracking {
				prx.LiteLLMDB = noSpendTrackingDB{db}
			}

			w := sendTeamPricesRequest(prx, "route-b")

			assert.Equal(t, http.StatusServiceUnavailable, w.Code)
			assert.Contains(t, w.Body.String(), "Model pricing unavailable")
			assert.Equal(t, int32(0), calls.Load())
			if spendTracking {
				assert.Empty(t, db.logs)
			}
		})
	}
}

func TestTeamModelPrices_OrganizationTariffConflict(t *testing.T) {
	policy := config.OrganizationPolicyConfig{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"public/shared":{"input_cost_per_token":0.005,"output_cost_per_token":0.005}}`),
		AllowlistSet:    true,
		ModelAllowlist:  []string{"public/shared"},
	}
	tests := []struct {
		name  string
		field string
		model string
		want  int
	}{
		{"team row for the model", `{"public/shared":{"input_cost_per_token":0.003}}`, "public/shared", http.StatusServiceUnavailable},
		{"unusable field, model outside the tariff", `{"route-b":{"input_cost_per_token":-1}}`, "route-b", http.StatusServiceUnavailable},
		{"team row for another model", `{"route-b":{"input_cost_per_token":0.003}}`, "public/shared", http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			prx, db := teamPricesProxy(t, teamPricesUpstream(t, &calls).URL, tt.field, policy)

			w := sendTeamPricesRequest(prx, tt.model)

			require.Equal(t, tt.want, w.Code, w.Body.String())
			if tt.want != http.StatusOK {
				assert.Equal(t, int32(0), calls.Load())
				assert.Empty(t, db.logs)
				return
			}
			require.Len(t, db.logs, 1)
			assert.InDelta(t, (10*0.005+5*0.005)*1.5, db.logs[0].Spend, 1e-12)
			assert.NotContains(t, spendLogsMetadata(t, db.logs[0]), "billing_price_source")
		})
	}
}

// The model name matches exactly, as an organization tariff's does.
func TestTeamModelPrices_KeyIsTheExactRequestedModel(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := &RequestLogContext{TokenInfo: &dbmodels.TokenInfo{
		TeamID: "team-1", TeamModelPrices: json.RawMessage(`{"public/shared":{"input_cost_per_token":0.003}}`),
	}}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	require.True(t, prx.admitTeamModelPrice(httptest.NewRecorder(), req, "Public/Shared", logCtx))
	assert.Nil(t, logCtx.teamModelPrice)

	require.True(t, prx.admitTeamModelPrice(httptest.NewRecorder(), req, "public/shared", logCtx))
	require.NotNil(t, logCtx.teamModelPrice)
	assert.Equal(t, 0.003, logCtx.teamModelPrice.InputCostPerToken)
}

// Images of a row's image tool are billed at the price list row it names,
// and not at all when it names the billed row itself under another name;
// a later price list reload changes neither.
func TestTeamModelPrices_ImageToolRow(t *testing.T) {
	images := &converter.TokenUsage{ImageToolGenerations: 1}
	team := &routermodels.ModelPrice{InputCostPerToken: 0.003}

	t.Run("the billed row itself", func(t *testing.T) {
		prx := NewTestProxyBuilder().Build()
		general := &routermodels.ModelPrice{InputCostPerToken: 0.001, OutputCostPerImage: 0.04, ImageGenerationToolModel: "alias-a"}
		registry := routermodels.NewModelPriceRegistry()
		registry.Update(map[string]*routermodels.ModelPrice{"route-a": general, "alias-a": general})
		prx.priceRegistry = registry
		logCtx := &RequestLogContext{teamModelPrice: team}
		_, billed := prx.resolveBillingPrice(logCtx, "public/shared", "route-a", "route-a")

		reloaded := *general
		registry.Update(map[string]*routermodels.ModelPrice{"route-a": &reloaded, "alias-a": &reloaded})

		assert.Zero(t, billed.CalculateCostsWithResolver(images, prx.billingPriceResolver(logCtx)).ImageGenerationToolCost)
	})

	t.Run("another row", func(t *testing.T) {
		prx := NewTestProxyBuilder().Build()
		general := &routermodels.ModelPrice{InputCostPerToken: 0.001, ImageGenerationToolModel: "m1"}
		image := &routermodels.ModelPrice{OutputCostPerImage: 0.04}
		registry := routermodels.NewModelPriceRegistry()
		registry.Update(map[string]*routermodels.ModelPrice{"openrouter/m1": general, "m1": image})
		prx.priceRegistry = registry
		logCtx := &RequestLogContext{teamModelPrice: team}
		_, billed := prx.resolveBillingPrice(logCtx, "openrouter/m1", "openrouter/m1", "openrouter/m1")

		// Without its own row, openrouter/m1 now resolves to m1.
		registry.Update(map[string]*routermodels.ModelPrice{"m1": image})

		assert.InDelta(t, 0.04, billed.CalculateCostsWithResolver(images, prx.billingPriceResolver(logCtx)).ImageGenerationToolCost, 1e-12)
	})
}

func TestTeamModelPrices_KafkaEventCarriesThePriceSource(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)
	assert.Empty(t, event.BillingPriceSource)
	encoded, err := json.Marshal(event)
	require.NoError(t, err)
	assert.NotContains(t, string(encoded), "billing_price_source")

	logCtx.billedAtTeamModelPrice = true
	event = prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)
	assert.Equal(t, "team_model_prices", event.BillingPriceSource)
}

// The budget reservation is at the team price with no margin: a key whose
// remaining budget covers the price list estimate with margin, but not the
// team's, is refused before the provider is called.
func TestTeamModelPrices_ReservationUsesTheTeamPrice(t *testing.T) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	client, err := ratelimit.NewValkeyClient(config.RedisConfig{InitAddresses: []string{addr}, ForceSingleClient: true})
	require.NoError(t, err)
	t.Cleanup(client.Close)

	tests := []struct {
		name  string
		field string
		want  int
	}{
		{"price list with margin fits", "", http.StatusOK},
		{"team price does not fit", `{"public/shared":{"input_cost_per_token":0.001,"output_cost_per_token":0.004}}`, http.StatusPaymentRequired},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			prx, db := teamPricesProxy(t, teamPricesUpstream(t, &calls).URL, tt.field)
			prx.tiktokenEnabled = false
			prx.budgetReservationEnabled = true
			prx.budgetReserver = budget.New(client, "test-team-prices-"+strconv.FormatInt(time.Now().UnixNano(), 10)+":", time.Minute)
			// 1000 completion tokens: 3.0 at the price list with margin, 4.0 at the team price.
			db.tokens["token"].MaxBudget = pointerTo(3.5)

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"public/shared","max_completion_tokens":1000,"messages":[{"role":"user","content":"hi"}]}`))
			req.Header.Set("Authorization", "Bearer token")
			w := httptest.NewRecorder()
			prx.ProxyRequest(w, req)

			require.Equal(t, tt.want, w.Code, w.Body.String())
			if tt.want == http.StatusPaymentRequired {
				assert.Equal(t, int32(0), calls.Load())
			}
		})
	}
}

// A retry after the price list lost the model's row is still served and
// billed at the team price chosen for the request.
func TestTeamModelPrices_RetryAfterThePriceListChanged(t *testing.T) {
	var prx *Proxy
	var failed, served atomic.Int32
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		failed.Add(1)
		prx.priceRegistry.Update(map[string]*routermodels.ModelPrice{"other": {InputCostPerToken: 9}})
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer down.Close()
	up := teamPricesUpstream(t, &served)

	info := &dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", TeamID: "team-1",
		CostMarginConfigs: []dbmodels.CostMarginConfig{{"global": {Percentage: 0.5}}},
		TeamModelPrices:   json.RawMessage(`{"public/shared":{"input_cost_per_token":0.003,"output_cost_per_token":0.004}}`),
	}
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": info}}
	credentials := []config.CredentialConfig{
		{Name: "down", Type: config.ProviderTypeOpenAI, BaseURL: down.URL, APIKey: "k", RPM: 100, TPM: 10000, Priority: 1},
		{Name: "up", Type: config.ProviderTypeOpenAI, BaseURL: up.URL, APIKey: "k", RPM: 100, TPM: 10000, Priority: 2},
	}
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "route-a", Credential: "down"},
		{Name: "route-a", Credential: "up"},
	})
	manager.SetModelAliases(map[string]string{"public/shared": "route-a"})
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)
	builder := NewTestProxyBuilder().WithCredentials(credentials...).WithMasterKey("master-key").WithMaxProviderRetries(2)
	builder.config.ModelManager = manager
	prx = builder.Build()
	prx.LiteLLMDB = db
	setTestModelPrice(prx, "route-a", &routermodels.ModelPrice{InputCostPerToken: 0.001, OutputCostPerToken: 0.002})

	w := sendTeamPricesRequest(prx, "public/shared")

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), failed.Load())
	assert.Equal(t, int32(1), served.Load())
	require.Len(t, db.logs, 1)
	assert.InDelta(t, 10*0.003+5*0.004, db.logs[0].Spend, 1e-12)
	assert.Equal(t, "team_model_prices", spendLogsMetadata(t, db.logs[0])["billing_price_source"])
}
