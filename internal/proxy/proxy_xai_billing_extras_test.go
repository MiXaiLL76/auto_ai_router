package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// An organization billed by its own tariff pays for tool images at that
// tariff's image row, never at the default price list's.
func TestOrganizationTariff_ImageToolPricedFromSameTariff(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed","model":"route-a",
			"output":[{"type":"image_generation_call","id":"ig_1","status":"completed","result":"AAAA"},
				{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[]}],
			"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,
				"server_side_tool_usage_details":{"image_generation_calls":1}}}`))
	}))
	defer upstream.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", OrganizationID: "org-1"},
	}}
	prx := newOrganizationPolicyProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID: "org-1",
		PriceProfileID: "profile-1",
		ModelPricesLink: writeProxyPolicyPrices(t, `{
			"public/shared": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002,
				"image_generation_tool_model": "org-image"},
			"org-image": {"output_cost_per_image": 0.5}
		}`),
	}})
	defaultPrices := routermodels.NewModelPriceRegistry()
	defaultPrices.Update(map[string]*routermodels.ModelPrice{"org-image": {OutputCostPerImage: 99}})
	prx.priceRegistry = defaultPrices

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", stringsReader(`{"model":"public/shared","input":"draw","tools":[{"type":"image_generation"}]}`))
	req.Header.Set("Authorization", "Bearer token")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, db.logs, 1)
	assert.InDelta(t, 10*0.001+5*0.002+0.5, db.logs[0].Spend, 1e-12)
}

func TestSanitizeJSONRequestBody_StripsClientControlledBillingParams(t *testing.T) {
	body := []byte(`{"model":"grok-4.7","deferred":true,"service_tier":"priority",
		"search_parameters":{"mode":"on","sources":[{"type":"x"}]},
		"extra_body":{"deferred":true,"search_parameters":{"mode":"auto"},"keep":1},
		"metadata":{"deferred":"user data","search_parameters":"user data"},
		"messages":[{"role":"user","content":"deferred search_parameters"}]}`)

	result, err := sanitizeAndExtractRequestBody(body, "application/json", false)

	require.NoError(t, err)
	require.True(t, result.Changed)
	assert.Equal(t, "grok-4.7", result.ModelID)
	var got map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(result.Body, &got))
	assert.NotContains(t, got, "deferred")
	assert.NotContains(t, got, "service_tier")
	assert.NotContains(t, got, "search_parameters")
	assert.JSONEq(t, `{"keep":1}`, string(got["extra_body"]))
	assert.JSONEq(t, `{"deferred":"user data","search_parameters":"user data"}`, string(got["metadata"]),
		"the same names nested in user data are preserved")
	assert.Contains(t, string(got["messages"]), "deferred search_parameters")
}

func TestSanitizeMultipartRequestBody_StripsDeferredAndSearchParameters(t *testing.T) {
	body, contentType := buildMultipartTestBody(t, "air-xai-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("grok-4.7")},
		multipartTestPart{disposition: `form-data; name="deferred"`, data: []byte("true")},
		multipartTestPart{disposition: `form-data; name="extra_body[search_parameters]"`, data: []byte(`{"mode":"on"}`)},
		multipartTestPart{disposition: `form-data; name="extra_body"`, data: []byte(`{"deferred":true,"keep":"yes"}`)},
		multipartTestPart{disposition: `form-data; name="prompt"`, data: []byte("keep")},
	)

	result, err := sanitizeAndExtractRequestBody(body, contentType, false)

	require.NoError(t, err)
	require.True(t, result.Changed)
	_, parts := parseMultipartTestBody(t, result.Body, contentType)
	require.Equal(t, []string{"model", "extra_body", "prompt"}, multipartPartNames(parts))
	assert.JSONEq(t, `{"keep":"yes"}`, string(parts[1].data))
}

func TestBuildKafkaSpendEvent_ServerSideToolFields(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.TokenUsage = &converter.TokenUsage{
		PromptTokens:            100,
		CompletionTokens:        10,
		ServerToolUsageReported: true,
		WebSearchRequests:       2,
		XSearchCalls:            1,
		XSearchPosts:            44,
		XSearchProfiles:         3,
		CodeExecutionCalls:      1,
		AttachmentSearchCalls:   2,
		CollectionsSearchCalls:  3,
		MCPCalls:                4,
		ImageToolGenerations:    5,
		ImageToolEdits:          6,
		ProviderCostUSD:         0.123,
	}
	costs := &converter.TokenCosts{
		WebSearchCost:           0.1,
		XSearchCost:             0.2,
		CodeExecutionCost:       0.3,
		AttachmentSearchCost:    0.4,
		CollectionsSearchCost:   0.5,
		ImageGenerationToolCost: 0.6,
		ToolUsageCost:           2.1,
		TotalCost:               3,
	}

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.x.ai", "success", 3, costs, 0, logCtx.StartTime)

	assert.Equal(t, 1, event.XSearchCalls)
	assert.Equal(t, 44, event.XSearchPosts)
	assert.Equal(t, 3, event.XSearchProfiles)
	assert.Equal(t, 1, event.CodeExecutionCalls)
	assert.Equal(t, 2, event.AttachmentSearchCalls)
	assert.Equal(t, 3, event.CollectionsSearchCalls)
	assert.Equal(t, 4, event.MCPCalls)
	assert.Equal(t, 5, event.ImageToolGenerations)
	assert.Equal(t, 6, event.ImageToolEdits)
	assert.Equal(t, 0.2, event.XSearchCost)
	assert.Equal(t, 0.3, event.CodeExecutionCost)
	assert.Equal(t, 0.4, event.AttachmentSearchCost)
	assert.Equal(t, 0.5, event.CollectionsSearchCost)
	assert.Equal(t, 0.6, event.ImageToolCost)
	assert.Equal(t, 2.1, event.ToolUsageCost)
	require.NotNil(t, event.ProviderReportedCost)
	assert.Equal(t, 0.123, *event.ProviderReportedCost)

	logCtx.TokenUsage.ProviderCostUSD = 0
	event = prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.x.ai", "success", 3, costs, 0, logCtx.StartTime)
	assert.Nil(t, event.ProviderReportedCost, "no provider figure is NULL, not a zero cost")
}

func TestBuildMetadata_ToolUsageCostIsTheSumOfToolCharges(t *testing.T) {
	usage := &converter.TokenUsage{PromptTokens: 1, WebSearchRequests: 1}
	costs := &converter.TokenCosts{WebSearchCost: 0.1, XSearchCost: 0.2, ToolUsageCost: 0.3, TotalCost: 0.5}

	metadata := decodeMetadata(t, buildMetadata("hash", nil, "", 200, usage, "127.0.0.1", costs, "grok-4.7", 0, 0, ""))

	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, 0.3, costBreakdown["tool_usage_cost"])
	assert.Equal(t, 0.1, costBreakdown["web_search_cost"])
	assert.Equal(t, 0.2, costBreakdown["x_search_cost"])
	serverToolUse := metadata["usage_object"].(map[string]interface{})["server_tool_use"].(map[string]interface{})
	assert.NotContains(t, serverToolUse, "x_posts_fetched", "per-tool counters are logged only when tools other than web search ran")
	assert.NotContains(t, metadata, "provider_reported_cost")
}
