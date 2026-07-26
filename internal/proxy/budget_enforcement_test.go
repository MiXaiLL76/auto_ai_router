package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func pointerTo[T any](value T) *T { return &value }

func TestBudgetLevelsFollowLiteLLMHierarchy(t *testing.T) {
	info := &dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", TeamID: "team-1", OrganizationID: "org-1",
		MaxBudget: pointerTo(10.0), TeamMaxBudget: pointerTo(20.0), OrgMaxBudget: pointerTo(30.0),
		TeamMemberMaxBudget: pointerTo(4.0), OrgMemberMaxBudget: pointerTo(5.0),
	}
	levels := budgetLevels(info)
	require.Len(t, levels, 5)
	assert.Equal(t, []string{
		"token:token-hash", "team:team-1", "org:org-1",
		"teammember:team-1:user-1", "orgmember:org-1:user-1",
	}, []string{levels[0].entity, levels[1].entity, levels[2].entity, levels[3].entity, levels[4].entity})
}

func TestBudgetLevelsUsePersonalUserLimitsWithoutTeam(t *testing.T) {
	rpm, tpm := int64(7), int64(800)
	levels := budgetLevels(&dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", UserRPMLimit: &rpm, UserTPMLimit: &tpm,
	})
	require.Len(t, levels, 2)
	assert.Equal(t, "user:user-1", levels[1].entity)
	assert.Equal(t, &rpm, levels[1].rpm)
	assert.Equal(t, &tpm, levels[1].tpm)
}

func TestEstimateCompletionTokensUsesRequestLimitThenSafeDefault(t *testing.T) {
	proxy := &Proxy{defaultEstimatedCompletionTokens: 321}
	assert.Equal(t, 42, proxy.estimateCompletionTokens([]byte(`{"max_completion_tokens":42}`)))
	assert.Equal(t, 321, proxy.estimateCompletionTokens([]byte(`{"messages":[]}`)))
	assert.Equal(t, 321, proxy.estimateCompletionTokens([]byte(`not-json`)))
}

// TestEstimateRequestCostUsesPublicModelPriceBeforeRealModelPrice guards
// against the budget-reservation path regressing to main's pre-fix behavior
// (real-model-first pricing, ignoring the client-facing alias entirely).
func TestEstimateRequestCostUsesPublicModelPriceBeforeRealModelPrice(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	registry := routermodels.NewModelPriceRegistry()
	registry.Update(map[string]*routermodels.ModelPrice{
		"gpt-5.2-chat": {
			OutputCostPerToken: 0.01,
		},
		"gpt-chat-latest": {
			OutputCostPerToken: 1.00,
		},
	})
	prx.priceRegistry = registry

	cost, ok := prx.estimateRequestCost(
		&RequestLogContext{},
		"gpt-5.2-chat",
		"gpt-5.2-chat",
		"gpt-chat-latest",
		[]byte(`{"messages":[],"max_completion_tokens":10}`),
	)

	require.True(t, ok)
	assert.InDelta(t, 0.10, cost, 0.000000001)
}

// TestEstimateRequestCostPrefersPublicModelPriceOverDistinctModelIDPrice
// covers the 3-way-divergent case (PublicModelID != ModelID != RealModelID,
// all three priced) that the earlier two-ID-collision tests couldn't
// distinguish: it proves PublicModelID wins specifically over ModelID, not
// just over RealModelID.
func TestEstimateRequestCostPrefersPublicModelPriceOverDistinctModelIDPrice(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	registry := routermodels.NewModelPriceRegistry()
	registry.Update(map[string]*routermodels.ModelPrice{
		"gpt-5.2-chat-alias": { // PublicModelID: client-facing alias
			OutputCostPerToken: 0.01,
		},
		"gpt-5.2-chat": { // ModelID: resolved canonical name
			OutputCostPerToken: 0.5,
		},
		"gpt-chat-latest": { // RealModelID: provider deployment name
			OutputCostPerToken: 1.00,
		},
	})
	prx.priceRegistry = registry

	cost, ok := prx.estimateRequestCost(
		&RequestLogContext{},
		"gpt-5.2-chat-alias",
		"gpt-5.2-chat",
		"gpt-chat-latest",
		[]byte(`{"messages":[],"max_completion_tokens":10}`),
	)

	require.True(t, ok)
	assert.InDelta(t, 0.10, cost, 0.000000001)
}

// TestResolveBillingPriceCachesAcrossCalls verifies budget reservation and
// final spend logging observe the same resolved price even if the registry
// is reloaded between the two calls on the same logCtx.
func TestResolveBillingPriceCachesAcrossCalls(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	registry := routermodels.NewModelPriceRegistry()
	registry.Update(map[string]*routermodels.ModelPrice{
		"gpt-5.2-chat": {OutputCostPerToken: 0.01},
	})
	prx.priceRegistry = registry

	logCtx := &RequestLogContext{}
	firstID, firstPrice := prx.resolveBillingPrice(logCtx, "gpt-5.2-chat", "gpt-5.2-chat", "gpt-chat-latest")
	require.NotNil(t, firstPrice)
	assert.Equal(t, "gpt-5.2-chat", firstID)

	// Simulate a background price-registry reload landing between the two
	// call sites (estimateRequestCost, then logSpendToLiteLLMDB).
	registry.Update(map[string]*routermodels.ModelPrice{
		"gpt-chat-latest": {OutputCostPerToken: 1.00},
	})

	secondID, secondPrice := prx.resolveBillingPrice(logCtx, "gpt-5.2-chat", "gpt-5.2-chat", "gpt-chat-latest")
	assert.Same(t, firstPrice, secondPrice)
	assert.Equal(t, firstID, secondID)
}

func TestEnforceBudgetAndRateLimitsRejectsUnknownPriceWhenSpendTrackingEnabled(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	prx.kafkaLog = &stubKafkaManager{enabled: true}
	prx.priceRegistry = routermodels.NewModelPriceRegistry()
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := testLogCtx(t)

	allowed := prx.enforceBudgetAndRateLimits(
		recorder, request, logCtx, "new-model", "provider/new-model-v2", []byte(`{"messages":[]}`),
	)

	assert.False(t, allowed)
	assert.Equal(t, http.StatusServiceUnavailable, recorder.Code)
	assert.Equal(t, "failure", logCtx.Status)
	assert.Nil(t, logCtx.ModelPrice)
}

func TestEnforceBudgetAndRateLimitsCachesResolvedAliasPrice(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	prx.kafkaLog = &stubKafkaManager{enabled: true}
	price := &routermodels.ModelPrice{InputCostPerToken: 0.000001}
	setTestModelPrice(prx, "model-alias", price)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := testLogCtx(t)

	allowed := prx.enforceBudgetAndRateLimits(
		recorder, request, logCtx, "model-alias", "provider/model-v2", []byte(`{"messages":[]}`),
	)

	assert.True(t, allowed)
	assert.Same(t, price, logCtx.ModelPrice)
	assert.Equal(t, "model-alias", logCtx.PriceModelID)
}

func TestProxyRequestRejectsUnknownPriceBeforeProvider(t *testing.T) {
	var providerCalls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		providerCalls.Add(1)
		_, _ = w.Write([]byte(`{"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer provider.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, provider.URL, "provider-key").
		Build()
	prx.kafkaLog = &stubKafkaManager{enabled: true}
	prx.priceRegistry = routermodels.NewModelPriceRegistry()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[]}`))
	request.Header.Set("Authorization", "Bearer master-key")
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, request)

	assert.Equal(t, http.StatusServiceUnavailable, recorder.Code)
	assert.Zero(t, providerCalls.Load())
}
