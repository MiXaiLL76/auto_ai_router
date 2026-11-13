package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
)

func costLogCtx(mode config.ResponseCostMode, total float64, breakdown []byte) *RequestLogContext {
	logCtx := &RequestLogContext{responseCostUSD: total, responseCostKnown: true, responseCostBreakdown: breakdown}
	if mode != "" {
		logCtx.OrganizationPolicy = &routermodels.OrganizationPolicy{ResponseCostMode: mode}
	}
	return logCtx
}

// The cost headers are an opt-in per organization, gated by response_cost_mode:
// off emits nothing, total emits X-Total-Cost, breakdown emits both.
func TestSetResponseCostHeader_GatedByMode(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	breakdown, _ := json.Marshal(map[string]any{"input_cost": 0.0001, "output_cost": 0.0001, "total_cost": 0.0002})

	t.Run("off emits nothing", func(t *testing.T) {
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, costLogCtx(config.ResponseCostModeOff, 0.0002, breakdown))
		assert.Empty(t, w.Header().Get(HeaderResponseCost))
		assert.Empty(t, w.Header().Get(HeaderCostBreakdown))
	})

	t.Run("total emits only X-Total-Cost", func(t *testing.T) {
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, costLogCtx(config.ResponseCostModeTotal, 0.0002, breakdown))
		assert.Equal(t, "0.0002", w.Header().Get(HeaderResponseCost))
		assert.Empty(t, w.Header().Get(HeaderCostBreakdown), "total mode must not leak the breakdown")
	})

	t.Run("breakdown emits both", func(t *testing.T) {
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, costLogCtx(config.ResponseCostModeBreakdown, 0.0002, breakdown))
		assert.Equal(t, "0.0002", w.Header().Get(HeaderResponseCost))
		var got map[string]any
		assert.NoError(t, json.Unmarshal([]byte(w.Header().Get(HeaderCostBreakdown)), &got))
		assert.Equal(t, 0.0001, got["input_cost"])
		assert.Equal(t, 0.0002, got["total_cost"])
	})

	t.Run("breakdown mode without a recorded breakdown emits only the total", func(t *testing.T) {
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, costLogCtx(config.ResponseCostModeBreakdown, 0.0002, nil))
		assert.Equal(t, "0.0002", w.Header().Get(HeaderResponseCost))
		assert.Empty(t, w.Header().Get(HeaderCostBreakdown))
	})

	t.Run("no organization policy emits nothing", func(t *testing.T) {
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, costLogCtx("", 0.0002, breakdown))
		assert.Empty(t, w.Header().Get(HeaderResponseCost))
		assert.Empty(t, w.Header().Get(HeaderCostBreakdown))
	})

	t.Run("cost unknown emits nothing even in breakdown mode", func(t *testing.T) {
		logCtx := &RequestLogContext{responseCostKnown: false}
		logCtx.OrganizationPolicy = &routermodels.OrganizationPolicy{ResponseCostMode: config.ResponseCostModeBreakdown}
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, logCtx)
		assert.Empty(t, w.Header().Get(HeaderResponseCost))
		assert.Empty(t, w.Header().Get(HeaderCostBreakdown))
	})

	t.Run("nil logCtx is a no-op", func(t *testing.T) {
		w := httptest.NewRecorder()
		prx.setResponseCostHeader(w, nil)
		assert.Empty(t, w.Header().Get(HeaderResponseCost))
	})
}

func TestSetResponseCostHeader_PlainDecimal(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()
	prx.setResponseCostHeader(w, costLogCtx(config.ResponseCostModeTotal, 1e-07, nil))
	assert.Equal(t, "0.0000001", w.Header().Get(HeaderResponseCost))
	assert.NotContains(t, w.Header().Get(HeaderResponseCost), "e")
}

// A provider's own cost headers must not leak to the client.
func TestResponseCostHeader_ProviderValuesDoNotLeak(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	src := http.Header{
		"Content-Type":     []string{"application/json"},
		"X-Total-Cost":     []string{"999"},
		"X-Cost-Breakdown": []string{`{"input_cost":999}`},
	}
	w := httptest.NewRecorder()
	prx.copyResponseHeaders(w, src, &config.CredentialConfig{Name: "c", Type: config.ProviderTypeOpenAI}, false)
	assert.Empty(t, w.Header().Get(HeaderResponseCost))
	assert.Empty(t, w.Header().Get(HeaderCostBreakdown))
}
