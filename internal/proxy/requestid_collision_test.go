package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_SharedRequestIDGetsDistinctAirEventID reproduces the
// scenario behind request_id.go's collision-resolution fallback: when otel
// trusts an inbound traceparent (config.OtelConfig.TrustIncomingTraceparent,
// default true), requestid.Middleware derives RequestID from that
// client-supplied header, so two entirely distinct requests (retry replay, or
// a malicious caller) can end up with the identical RequestID — which is also
// the spend-log primary key. The upstream here omits a response "id" field,
// so ClientResponseID stays empty and spendRequestID() falls back to the
// (here, colliding) RequestID for both requests, matching the scenario where
// the DB-level collision-resolution in insertSpendRowsCollisionSafe actually
// needs to kick in.
//
// AirEventID must still be distinct per logical request in this scenario, or
// the second request's spend row is silently dropped instead of being
// resolved via the AIR event ID fallback path.
func TestProxyRequest_SharedRequestIDGetsDistinctAirEventID(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"chat.completion","created":1,"model":"gpt-collide","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithSingleCredential("openai_primary", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-collide", &pricing.ModelPrice{
		InputCostPerToken: 0.000001, OutputCostPerToken: 0.000002,
	})

	newReq := func() *http.Request {
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
			"model": "gpt-collide",
			"messages": [{"role": "user", "content": "hello"}]
		}`))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", "application/json")
		// Simulates requestid.Middleware having already adopted the same
		// trusted inbound traceparent's trace ID for two distinct requests.
		ctx := requestid.WithID(req.Context(), "shared-trace-id")
		return req.WithContext(ctx)
	}

	for i := 0; i < 2; i++ {
		w := httptest.NewRecorder()
		prx.ProxyRequest(w, newReq())
		require.Equal(t, http.StatusOK, w.Code)
	}

	require.Len(t, dbStub.loggedEntries, 2)
	first, second := dbStub.loggedEntries[0], dbStub.loggedEntries[1]

	require.Equal(t, "shared-trace-id", first.RequestID, "precondition: both requests collide on the spend-log primary key")
	require.Equal(t, "shared-trace-id", second.RequestID, "precondition: both requests collide on the spend-log primary key")

	assert.NotEmpty(t, first.AirEventID)
	assert.NotEmpty(t, second.AirEventID)
	assert.NotEqual(t, first.AirEventID, second.AirEventID,
		"AirEventID must stay distinct per request even when RequestID collides, or the collision-resolution fallback in insertSpendRowsCollisionSafe has nothing to distinguish the two events by")
}
