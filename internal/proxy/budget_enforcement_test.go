package proxy

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
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

func TestBudgetLevelsNameTheirBudgetLevel(t *testing.T) {
	levels := budgetLevels(&dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", TeamID: "team-1", OrganizationID: "org-1",
	})
	got := make([]dbmodels.BudgetLevel, 0, len(levels))
	for _, level := range levels {
		got = append(got, level.level)
	}
	assert.Equal(t, []dbmodels.BudgetLevel{
		dbmodels.BudgetLevelKey, dbmodels.BudgetLevelTeam, dbmodels.BudgetLevelOrganization,
		dbmodels.BudgetLevelTeamMember, dbmodels.BudgetLevelOrganizationMember,
	}, got)
	personal := budgetLevels(&dbmodels.TokenInfo{Token: "token-hash", UserID: "user-1"})
	require.Len(t, personal, 2)
	assert.Equal(t, dbmodels.BudgetLevelUser, personal[1].level)
}

// Each key budget window is a level of its own, right after the key, with a
// counter named after the window's start: a new window starts a new counter.
// A window AIR cannot enforce (auth refuses its key) gets no level.
func TestBudgetLevelsIncludeTheKeyBudgetWindows(t *testing.T) {
	start := time.Date(2026, 3, 10, 21, 0, 0, 0, time.UTC)
	levels := budgetLevels(&dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1",
		BudgetWindows: []dbmodels.BudgetWindow{
			{BudgetLimit: dbmodels.BudgetLimit{Duration: "1d", MaxBudget: 10}, Start: start, Spend: 4},
			{BudgetLimit: dbmodels.BudgetLimit{Duration: "2d", MaxBudget: 1}},
		},
	})
	require.Len(t, levels, 3)
	window := levels[1]
	assert.Equal(t, "token:token-hash:window:1d:"+strconv.FormatInt(start.Unix(), 10), window.entity)
	assert.Equal(t, dbmodels.BudgetLevelKey, window.level)
	assert.Equal(t, "1d", window.window)
	assert.Equal(t, 10.0, *window.maxBudget)
	assert.Equal(t, 4.0, window.dbSpend)
	assert.Nil(t, window.rpm)
	assert.Nil(t, window.tpm)
	assert.Equal(t, "user:user-1", levels[2].entity)
}

// TestEnforceBudgetAndRateLimitsReservationRefusalNamesTheKeyLimit covers the
// pre-request reservation path, which refuses on its own counters before the
// DB snapshot sees the spend: its 402 must carry the same per-level text as
// the auth-time check.
func TestEnforceBudgetAndRateLimitsReservationRefusalNamesTheKeyLimit(t *testing.T) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	client, err := ratelimit.NewValkeyClient(config.RedisConfig{InitAddresses: []string{addr}, ForceSingleClient: true})
	require.NoError(t, err)
	t.Cleanup(client.Close)

	tests := []struct {
		name string
		info *dbmodels.TokenInfo
		want string
	}{
		{
			name: "key limit",
			info: &dbmodels.TokenInfo{Token: "tok-key", MaxBudget: pointerTo(1.0), Spend: 0.99},
			want: "Key budget exceeded",
		},
		{
			name: "user budget",
			info: &dbmodels.TokenInfo{Token: "tok-user", UserID: "user-1", UserMaxBudget: pointerTo(1.0), UserSpend: pointerTo(0.99)},
			want: "Budget exceeded",
		},
		{
			name: "key window",
			info: &dbmodels.TokenInfo{Token: "tok-window", BudgetWindows: []dbmodels.BudgetWindow{{
				BudgetLimit: dbmodels.BudgetLimit{Duration: "7d", MaxBudget: 1},
				Start:       time.Date(2026, 3, 8, 21, 0, 0, 0, time.UTC),
				Spend:       0.99,
			}}},
			want: "Key 7d budget exceeded",
		},
		{
			// The key has room and is reserved first; the user level then refuses.
			name: "user budget after the key is reserved",
			info: &dbmodels.TokenInfo{
				Token: "tok-key-then-user", MaxBudget: pointerTo(1000.0),
				UserID: "user-2", UserMaxBudget: pointerTo(1.0), UserSpend: pointerTo(0.99),
			},
			want: "Budget exceeded",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().Build()
			prx.budgetReservationEnabled = true
			prefix := "test-budget-level-" + strconv.FormatInt(time.Now().UnixNano(), 10) + ":"
			prx.budgetReserver = budget.New(client, prefix, time.Minute)
			setTestModelPrice(prx, "model-a", &routermodels.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 1})
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			logCtx := testLogCtx(t)
			logCtx.TokenInfo = tt.info

			allowed := prx.enforceBudgetAndRateLimits(
				recorder, request, logCtx, "model-a", "model-a", []byte(`{"max_tokens":10,"messages":[]}`),
			)

			require.False(t, allowed)
			assertAuthErrorShape(t, recorder, http.StatusPaymentRequired)
			assert.Contains(t, recorder.Body.String(), `"message":"`+tt.want+`"`)
			assert.Equal(t, http.StatusPaymentRequired, logCtx.HTTPStatus)
			assert.Empty(t, logCtx.reservedEntities)

			// Whatever was reserved before the refusal is released: the key's
			// counter is back at its DB spend, so a reservation of the key's
			// whole remaining budget still fits.
			if tt.info.MaxBudget != nil {
				remaining := *tt.info.MaxBudget - tt.info.Spend
				allowed, err := prx.budgetReserver.TryReserve(request.Context(), "token:"+tt.info.Token, tt.info.Spend, remaining, *tt.info.MaxBudget)
				require.NoError(t, err)
				assert.True(t, allowed, "a reservation made before the refusal leaked into the key's counter")
			}
		})
	}
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

// TestEstimateRequestCostSkipsPromptTokenEstimateWhenTiktokenDisabled is a
// regression test: estimateRequestCost used to call
// estimatePromptTokensForModel (the full tiktoken BPE pass) unconditionally,
// with no check against Proxy.tiktokenEnabled — the only other call site of
// that function is correctly gated. tiktoken_enabled=false must skip the
// prompt-token estimate (PromptTokens: 0) while still leaving budget
// reservation active (costKnown stays true; only the completion-token
// portion of the estimate is used).
func TestEstimateRequestCostSkipsPromptTokenEstimateWhenTiktokenDisabled(t *testing.T) {
	registry := routermodels.NewModelPriceRegistry()
	registry.Update(map[string]*routermodels.ModelPrice{
		"gpt-5.2-chat": {InputCostPerToken: 0.01, OutputCostPerToken: 0.02},
	})
	body := []byte(`{"messages":[{"role":"user","content":"this is a real prompt with several words in it"}],"max_completion_tokens":10}`)

	t.Run("disabled: cost reflects only the completion-token estimate", func(t *testing.T) {
		prx := NewTestProxyBuilder().WithTiktokenEnabled(false).Build()
		prx.priceRegistry = registry

		cost, ok := prx.estimateRequestCost(&RequestLogContext{}, "gpt-5.2-chat", "gpt-5.2-chat", "gpt-5.2-chat", body)
		require.True(t, ok, "costKnown must stay true even with tiktoken_enabled=false")

		_, modelPrice := prx.resolveBillingPrice(&RequestLogContext{}, "gpt-5.2-chat", "gpt-5.2-chat", "gpt-5.2-chat")
		require.NotNil(t, modelPrice)
		expected := modelPrice.CalculateCost(&converter.TokenUsage{
			CompletionTokens: prx.estimateCompletionTokens(body),
		})
		assert.InDelta(t, expected, cost, 0.000000001)
	})

	t.Run("enabled: cost also includes a non-zero prompt-token contribution", func(t *testing.T) {
		prx := NewTestProxyBuilder().WithTiktokenEnabled(true).Build()
		prx.priceRegistry = registry

		cost, ok := prx.estimateRequestCost(&RequestLogContext{}, "gpt-5.2-chat", "gpt-5.2-chat", "gpt-5.2-chat", body)
		require.True(t, ok)

		promptOnlyTokens := estimatePromptTokensForModel(body, "gpt-5.2-chat")
		require.Greater(t, promptOnlyTokens, 0, "test setup: body must yield a non-zero prompt-token estimate")
		assert.Greater(t, cost, modelPriceCompletionOnlyCost(t, registry.GetPrice("gpt-5.2-chat"), prx.estimateCompletionTokens(body)))
	})
}

func modelPriceCompletionOnlyCost(t *testing.T, modelPrice *routermodels.ModelPrice, completionTokens int) float64 {
	t.Helper()
	require.NotNil(t, modelPrice)
	return modelPrice.CalculateCost(&converter.TokenUsage{CompletionTokens: completionTokens})
}
