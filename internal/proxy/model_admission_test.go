package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newAdmissionTestProxy(
	t *testing.T,
	credentials []config.CredentialConfig,
	modelConfigs []config.ModelRPMConfig,
	prices map[string]*routermodels.ModelPrice,
) *Proxy {
	t.Helper()
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, modelConfigs)
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)
	builder := NewTestProxyBuilder().WithCredentials(credentials...).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	prx := builder.Build()
	prx.kafkaLog = &stubKafkaManager{enabled: true}
	prx.priceRegistry = routermodels.NewModelPriceRegistry()
	prx.priceRegistry.ReplaceFilePrices(prices)
	return prx
}

func TestModelAdmissionKeepsCredentialPriceGranularityAcrossPriorityAndFallback(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "primary-unpriced", Type: config.ProviderTypeOpenAI, Priority: 1, RPM: -1},
		{Name: "primary-priced", Type: config.ProviderTypeOpenAI, Priority: 2, RPM: -1},
		{Name: "fallback-unpriced", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: -1},
		{Name: "fallback-priced", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: -1},
	}
	configs := []config.ModelRPMConfig{
		{Name: "shared", Model: "missing-a", Credential: "primary-unpriced"},
		{Name: "shared", Model: "priced-a", Credential: "primary-priced"},
		{Name: "shared", Model: "missing-b", Credential: "fallback-unpriced"},
		{Name: "shared", Model: "priced-b", Credential: "fallback-priced"},
	}
	prx := newAdmissionTestProxy(t, credentials, configs, map[string]*routermodels.ModelPrice{
		"priced-a": {}, // An explicit zero tariff is still a valid price row.
		"priced-b": {},
	})

	admission, err := prx.resolveModelAdmission(
		&dbmodels.TokenInfo{IsMasterKey: true}, scope.AdminContext(), nil, "shared", true,
	)
	require.NoError(t, err)
	assert.Equal(t, map[string]bool{
		"primary-unpriced":  true,
		"fallback-unpriced": true,
	}, admission.excluded)

	selected, ok := prx.selectCredentialForModel(
		httptest.NewRecorder(), "shared", "", "", admission.excluded,
		&RequestLogContext{Scope: scope.AdminContext()},
	)
	require.True(t, ok)
	assert.Equal(t, "primary-priced", selected.Name)

	prx.priceRegistry.ReplaceFilePrices(map[string]*routermodels.ModelPrice{"shared": {}})
	admission, err = prx.resolveModelAdmission(
		&dbmodels.TokenInfo{IsMasterKey: true}, scope.AdminContext(), nil, "shared", true,
	)
	require.NoError(t, err)
	assert.Empty(t, admission.excluded, "a logical route price activates every serving credential")
}

func TestModelAdmissionCanSelectPricedFallbackWhenPrimaryIsUnpriced(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "primary", Type: config.ProviderTypeOpenAI, RPM: -1},
		{Name: "fallback", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: -1},
	}
	prx := newAdmissionTestProxy(t, credentials, []config.ModelRPMConfig{
		{Name: "shared", Model: "missing", Credential: "primary"},
		{Name: "shared", Model: "priced", Credential: "fallback"},
	}, map[string]*routermodels.ModelPrice{"priced": {}})

	admission, err := prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "shared", false)
	require.NoError(t, err)
	selected, ok := prx.selectCredentialForModel(
		httptest.NewRecorder(), "shared", "", "", admission.excluded,
		&RequestLogContext{Scope: scope.PublicContext()},
	)
	require.True(t, ok)
	assert.Equal(t, "fallback", selected.Name)
}

func TestModelAdmissionPublicPriceIsSufficient(t *testing.T) {
	credential := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1}
	prx := newAdmissionTestProxy(t, []config.CredentialConfig{credential}, []config.ModelRPMConfig{
		{Name: "route", Model: "provider-real", Credential: credential.Name},
	}, map[string]*routermodels.ModelPrice{"public-alias": {}})
	prx.modelManager.SetClientModelIDs([]string{"canonical"})
	prx.modelManager.SetModelAliases(map[string]string{"canonical": "route"})
	prx.modelManager.SetAcceptedModelAliases(map[string]string{"public-alias": "canonical"})

	admission, err := prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "public-alias", false)
	require.NoError(t, err)
	assert.Empty(t, admission.excluded)

	listed, err := prx.ListModelsForToken(nil, scope.PublicContext())
	require.NoError(t, err)
	require.Len(t, listed.Data, 1)
	assert.Equal(t, "public-alias", listed.Data[0].ID)

	prx.priceRegistry.ReplaceFilePrices(map[string]*routermodels.ModelPrice{
		"public-alias": {}, "provider-real": {},
	})
	_, err = prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "public-alias", false)
	require.NoError(t, err)

	prx.priceRegistry.ReplaceFilePrices(nil)
	_, err = prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "public-alias", false)
	assert.ErrorIs(t, err, errModelNotAvailable)
	listed, err = prx.ListModelsForToken(nil, scope.PublicContext())
	require.NoError(t, err)
	assert.Empty(t, listed.Data)
}

func TestModelAdmissionProxyUsesForwardedLogicalModelForBillingPrice(t *testing.T) {
	credential := config.CredentialConfig{Name: "downstream", Type: config.ProviderTypeAIR, RPM: -1}
	prx := newAdmissionTestProxy(t, []config.CredentialConfig{credential}, []config.ModelRPMConfig{
		{Name: "logical", Model: "direct-only-real", Credential: credential.Name},
	}, map[string]*routermodels.ModelPrice{"direct-only-real": {}})

	_, err := prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "logical", false)
	assert.ErrorIs(t, err, errModelNotAvailable)

	prx.priceRegistry.ReplaceFilePrices(map[string]*routermodels.ModelPrice{"logical": {}})
	_, err = prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "logical", false)
	require.NoError(t, err)

	prx = newAdmissionTestProxy(t, []config.CredentialConfig{credential}, []config.ModelRPMConfig{
		{Name: "logical", Model: "global-real"},
	}, map[string]*routermodels.ModelPrice{"global-real": {}})
	_, err = prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "logical", false)
	require.NoError(t, err, "a global real-model price applies because proxy credentials forward the shared logical mapping")
}

func TestModelAdmissionDoesNotRequirePriceWhenSpendTrackingIsDisabled(t *testing.T) {
	credential := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1}
	prx := newAdmissionTestProxy(t, []config.CredentialConfig{credential}, []config.ModelRPMConfig{
		{Name: "route", Credential: credential.Name},
	}, nil)
	prx.kafkaLog = nil

	_, err := prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "route", false)
	require.NoError(t, err)
}

func TestModelAdmissionEmptyTopologyIsClosedWorld(t *testing.T) {
	prx := newAdmissionTestProxy(t, nil, nil, nil)

	_, err := prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "unknown", false)
	assert.ErrorIs(t, err, errModelNotAvailable)
	listed, err := prx.ListModelsForToken(nil, scope.PublicContext())
	require.NoError(t, err)
	assert.Empty(t, listed.Data)
}

func TestProxyConstructionSynchronizesCatalogCredentialInventory(t *testing.T) {
	credential := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1}
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "route", Credential: credential.Name},
	})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	prx := builder.Build()

	listed, err := prx.ListModelsForToken(nil, scope.PublicContext())
	require.NoError(t, err)
	require.Len(t, listed.Data, 1)
	assert.Equal(t, "route", listed.Data[0].ID)
	_, err = prx.resolveModelAdmission(nil, scope.PublicContext(), nil, "route", false)
	require.NoError(t, err)
}

func TestModelListingUsesInferenceAdmissionAndIgnoresTransientAvailability(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "a", Type: config.ProviderTypeOpenAI, RPM: -1},
		{Name: "b", Type: config.ProviderTypeOpenAI, RPM: -1},
	}
	prx := newAdmissionTestProxy(t, credentials, []config.ModelRPMConfig{
		{Name: "model-a", Credential: "a"},
		{Name: "model-b", Credential: "b"},
	}, map[string]*routermodels.ModelPrice{"model-a": {}, "model-b": {}})
	prx.strictAllTeamModelsACL = true
	token := &dbmodels.TokenInfo{Models: []string{"model-a"}}

	listed, err := prx.ListModelsForToken(token, scope.PublicContext())
	require.NoError(t, err)
	require.Len(t, listed.Data, 1)
	assert.Equal(t, "model-a", listed.Data[0].ID)
	_, err = prx.resolveModelAdmission(token, scope.PublicContext(), nil, "model-b", false)
	assert.ErrorIs(t, err, errModelNotAllowed)

	prx.balancer.BanUntil("a", "model-a", http.StatusTooManyRequests, testFutureTime(), "test")
	listed, err = prx.ListModelsForToken(token, scope.PublicContext())
	require.NoError(t, err)
	require.Len(t, listed.Data, 1, "a transient ban must not make /v1/models flicker")
}

func TestOrganizationBillingPriceSurvivesBasePriceRemoval(t *testing.T) {
	credential := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1}
	prx := newAdmissionTestProxy(t, []config.CredentialConfig{credential}, []config.ModelRPMConfig{
		{Name: "route", Credential: credential.Name},
	}, map[string]*routermodels.ModelPrice{"route": {}})
	logCtx := &RequestLogContext{
		Credential:           &credential,
		OrganizationPolicy:   &routermodels.OrganizationPolicy{OrganizationID: "org"},
		billingPriceResolved: true,
		billingPriceModelID:  "org/route",
		billingPrice:         &routermodels.ModelPrice{},
	}
	prx.priceRegistry.ReplaceFilePrices(nil)
	w := httptest.NewRecorder()
	ok := prx.checkModelPriceAvailable(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), logCtx, "route", "route")

	assert.True(t, ok)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "org/route", logCtx.PriceModelID)
	assert.Same(t, logCtx.billingPrice, logCtx.ModelPrice)
}

func TestProxyRetryRechecksBillingPrice(t *testing.T) {
	var prx *Proxy
	var firstCalls, secondCalls int
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		firstCalls++
		prx.priceRegistry.ReplaceFilePrices(nil)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"retry"}}`))
	}))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		secondCalls++
		w.WriteHeader(http.StatusOK)
	}))
	defer second.Close()

	credentials := []config.CredentialConfig{
		{Name: "first", Type: config.ProviderTypeProxy, BaseURL: first.URL, APIKey: "key", Priority: 1, RPM: -1},
		{Name: "second", Type: config.ProviderTypeProxy, BaseURL: second.URL, APIKey: "key", Priority: 2, RPM: -1},
	}
	prx = newAdmissionTestProxy(t, credentials, []config.ModelRPMConfig{
		{Name: "route", Credential: "first"},
		{Name: "route", Credential: "second"},
	}, map[string]*routermodels.ModelPrice{"route": {}})
	prx.maxProviderRetries = 1

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"route","messages":[]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, 1, firstCalls)
	assert.Zero(t, secondCalls)
}

func TestFallbackProxyRechecksBillingPrice(t *testing.T) {
	var prx *Proxy
	var primaryCalls, fallbackCalls int
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls++
		prx.priceRegistry.ReplaceFilePrices(nil)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "retry"}})
	}))
	defer primary.Close()
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls++
		w.WriteHeader(http.StatusOK)
	}))
	defer fallback.Close()

	credentials := []config.CredentialConfig{
		{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: primary.URL, APIKey: "key", RPM: -1},
		{Name: "fallback", Type: config.ProviderTypeProxy, BaseURL: fallback.URL, APIKey: "key", IsFallback: true, RPM: -1},
	}
	prx = newAdmissionTestProxy(t, credentials, []config.ModelRPMConfig{
		{Name: "route", Credential: "primary"},
		{Name: "route", Credential: "fallback"},
	}, map[string]*routermodels.ModelPrice{"route": {}})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(`{"model":"route","messages":[]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, 1, primaryCalls)
	assert.Zero(t, fallbackCalls)
}

func TestFallbackProxyReplacesInitialCredentialBillingMetadata(t *testing.T) {
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"fallback","choices":[],"usage":{"total_tokens":1}}`))
	}))
	defer fallback.Close()

	credentials := []config.CredentialConfig{
		{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: "http://primary.invalid", APIKey: "key", RPM: -1},
		{Name: "fallback", Type: config.ProviderTypeProxy, BaseURL: fallback.URL, APIKey: "key", IsFallback: true, RPM: -1},
	}
	directPrice := &routermodels.ModelPrice{InputCostPerToken: 1}
	fallbackPrice := &routermodels.ModelPrice{InputCostPerToken: 2}
	prx := newAdmissionTestProxy(t, credentials, []config.ModelRPMConfig{
		{Name: "route", Model: "global-real"},
		{Name: "route", Model: "direct-real", Credential: "primary"},
	}, map[string]*routermodels.ModelPrice{
		"direct-real": directPrice,
		"global-real": fallbackPrice,
	})
	logCtx := &RequestLogContext{
		Credential:           &credentials[0],
		PublicModelID:        "route",
		ModelID:              "route",
		RealModelID:          "direct-real",
		PriceModelID:         "direct-real",
		ModelPrice:           directPrice,
		billingPriceResolved: true,
		billingPriceModelID:  "direct-real",
		billingPrice:         directPrice,
		Scope:                scope.PublicContext(),
	}
	body := []byte(`{"model":"route","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	success, reason := prx.TryFallbackProxy(
		w, req, "route", "primary", http.StatusInternalServerError,
		RetryReasonServerErr, body, time.Now(), logCtx,
	)

	require.True(t, success)
	assert.Empty(t, reason)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "fallback", logCtx.Credential.Name)
	assert.Equal(t, "global-real", logCtx.RealModelID)
	assert.Equal(t, "global-real", logCtx.PriceModelID)
	assert.Same(t, fallbackPrice, logCtx.ModelPrice)
}

func testFutureTime() time.Time { return time.Now().Add(time.Minute) }
