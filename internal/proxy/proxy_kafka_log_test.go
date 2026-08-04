package proxy

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	pricingmodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stubKafkaManager is a minimal kafkalog.Manager test double that records
// every event passed to LogSpend, optionally failing with a fixed error.
type stubKafkaManager struct {
	events  []*kafkalog.SpendEvent
	err     error
	enabled bool
}

func (s *stubKafkaManager) LogSpend(event *kafkalog.SpendEvent) error {
	s.events = append(s.events, event)
	return s.err
}
func (s *stubKafkaManager) IsEnabled() bool       { return s.enabled }
func (s *stubKafkaManager) IsHealthy() bool       { return true }
func (s *stubKafkaManager) Stats() kafkalog.Stats { return kafkalog.Stats{} }
func (s *stubKafkaManager) Shutdown(context.Context) error {
	return nil
}

var _ kafkalog.Manager = (*stubKafkaManager)(nil)

func testLogCtx(t *testing.T) *RequestLogContext {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	return &RequestLogContext{
		RequestID: "req-123",
		StartTime: time.Date(2026, 7, 15, 10, 0, 0, 0, time.UTC),
		Request:   req,
		Token:     "sk-test",
		ModelID:   "gpt-4o-mini",
		Credential: &config.CredentialConfig{
			Name:    "openai_primary",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: "https://api.openai.com/v1",
		},
		TokenUsage: &converter.TokenUsage{
			PromptTokens:         100,
			CompletionTokens:     50,
			WebSearchRequests:    2,
			WebSearchContextSize: "high",
		},
		SessionID: "session-1",
	}
}

func setTestModelPrice(prx *Proxy, modelID string, price *pricingmodels.ModelPrice) {
	registry := pricingmodels.NewModelPriceRegistry()
	registry.Update(map[string]*pricingmodels.ModelPrice{modelID: price})
	prx.priceRegistry = registry
}

func TestBuildKafkaSpendEvent_BasicMapping(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	endTime := logCtx.StartTime.Add(1230 * time.Millisecond)

	event := prx.buildKafkaSpendEvent(logCtx, "openai_primary", "openai_primary:gpt-4o-mini", "hashed-token",
		"user-1", "team-1", "org-1", "end-user@example.com", "api.openai.com", "success",
		0.00057, nil, 4.2, endTime)

	require.NotNil(t, event)
	assert.Equal(t, "req-123", event.RequestID)
	assert.Equal(t, logCtx.StartTime, event.StartTime)
	assert.Equal(t, endTime, event.EndTime)
	assert.Equal(t, int64(1230), event.DurationMs)
	assert.Equal(t, "acompletion", event.CallType)
	assert.Equal(t, "api.openai.com", event.APIBase)
	assert.Equal(t, "success", event.Status)
	assert.Equal(t, "gpt-4o-mini", event.Model)
	assert.Equal(t, "gpt-4o-mini", event.RealModel, "RealModel should fall back to ModelID when RealModelID is empty")
	assert.Equal(t, "openai_primary:gpt-4o-mini", event.ModelID)
	assert.Equal(t, "openai_primary", event.CredentialName)
	assert.Equal(t, string(config.ProviderTypeOpenAI), event.CredentialType)
	assert.Equal(t, "https://api.openai.com/v1", event.CredentialBaseURL)
	assert.Equal(t, "test-version", event.ServerVersion)
	assert.Equal(t, "test-commit", event.ServerCommit)
	assert.Equal(t, 100, event.PromptTokens)
	assert.Equal(t, 50, event.CompletionTokens)
	assert.Equal(t, 150, event.TotalTokens)
	assert.Equal(t, 2, event.WebSearchRequests)
	assert.Equal(t, "high", event.WebSearchContextSize)
	assert.Equal(t, 0.00057, event.TotalCost)
	assert.Equal(t, "hashed-token", event.APIKeyHash)
	assert.Equal(t, "user-1", event.UserID)
	assert.Equal(t, "session-1", event.SessionID)
	assert.Equal(t, 4.2, event.OverheadMs)

	// Not streamed: TTFT fields must stay nil.
	assert.Nil(t, event.CompletionStartTime)
	assert.Nil(t, event.TTFTMs)

	// Body capture is explicitly out of scope: always the zero-value placeholder.
	assert.False(t, event.BodyCaptured)
	assert.Equal(t, 0, event.BodyRequestBytes)
	assert.Equal(t, 0, event.BodyResponseBytes)
}

func TestBuildKafkaSpendEvent_UsesClientResponseID(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.ClientResponseID = "chatcmpl-client-123"

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)

	assert.Equal(t, "chatcmpl-client-123", event.RequestID)
}

func TestBuildKafkaSpendEvent_NilTokenUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.TokenUsage = nil

	assert.NotPanics(t, func() {
		event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
			"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)
		assert.Equal(t, 0, event.PromptTokens)
		assert.Equal(t, 0, event.TotalTokens)
	})
}

func TestBuildKafkaSpendEvent_NormalizesTokenUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.TokenUsage = &converter.TokenUsage{
		PromptTokens:           -100,
		CompletionTokens:       50,
		AudioInputTokens:       -10,
		CachedInputTokens:      -80,
		CachedAudioInputTokens: 40,
		CacheCreationTokens:    10,
		CacheCreation5mTokens:  8,
		CacheCreation1hTokens:  8,
	}

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)

	assert.Equal(t, 0, event.PromptTokens)
	assert.Equal(t, 50, event.CompletionTokens)
	assert.Equal(t, 50, event.TotalTokens)
	assert.Equal(t, 0, event.AudioInputTokens)
	assert.Equal(t, 0, event.CachedInputTokens)
	assert.Equal(t, 0, event.CachedAudioInputTokens)
	assert.Equal(t, 10, event.CacheCreationTokens)
	assert.Equal(t, 8, event.CacheCreation5mTokens)
	assert.Equal(t, 2, event.CacheCreation1hTokens)
}

func TestBuildKafkaSpendEvent_CacheBreakdownMapped(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.TokenUsage = &converter.TokenUsage{
		PromptTokens:           200,
		CompletionTokens:       50,
		CachedInputTokens:      80,
		CachedAudioInputTokens: 40,
		CacheCreationTokens:    30,
		CacheCreation5mTokens:  10,
		CacheCreation1hTokens:  20,
	}

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)

	assert.Equal(t, 80, event.CachedInputTokens)
	assert.Equal(t, 40, event.CachedAudioInputTokens)
	assert.Equal(t, 30, event.CacheCreationTokens)
	assert.Equal(t, 10, event.CacheCreation5mTokens)
	assert.Equal(t, 20, event.CacheCreation1hTokens)
}

func TestBuildKafkaSpendEvent_TTFTComputedWhenStreamed(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.CompletionStartTime = logCtx.StartTime.Add(310 * time.Millisecond)
	endTime := logCtx.StartTime.Add(1230 * time.Millisecond)

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, endTime)

	require.NotNil(t, event.CompletionStartTime)
	assert.Equal(t, logCtx.CompletionStartTime, *event.CompletionStartTime)
	require.NotNil(t, event.TTFTMs)
	assert.Equal(t, int64(310), *event.TTFTMs)
}

func TestBuildKafkaSpendEvent_TokenCostsMapped(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	costs := &converter.TokenCosts{
		InputCost:     0.0003,
		OutputCost:    0.00027,
		WebSearchCost: 0.00011,
		TotalCost:     0.00068,
	}

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", costs.TotalCost, costs, 0, logCtx.StartTime)

	assert.Equal(t, 0.0003, event.InputCost)
	assert.Equal(t, 0.00027, event.OutputCost)
	assert.Equal(t, 0.00011, event.WebSearchCost)
	assert.Equal(t, 0.00068, event.TotalCost)
}

func TestBuildKafkaSpendEvent_ErrorClassOnlyOnFailure(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	logCtx := testLogCtx(t)
	logCtx.HTTPStatus = 429
	eventFail := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "failure", 0, nil, 0, logCtx.StartTime)
	assert.Equal(t, "RateLimitError", eventFail.ErrorClass)

	logCtx2 := testLogCtx(t)
	logCtx2.HTTPStatus = 200
	eventOK := prx.buildKafkaSpendEvent(logCtx2, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx2.StartTime)
	assert.Empty(t, eventOK.ErrorClass)
}

func TestBuildKafkaSpendEvent_RealModelIDPreserved(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.RealModelID = "gpt-4o-mini-2024-07-18"

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)

	assert.Equal(t, "gpt-4o-mini-2024-07-18", event.RealModel)
}

func TestBuildKafkaSpendEvent_KeyAliasesFromTokenInfo(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.TokenInfo = &litellmdb.TokenInfo{
		KeyAlias:  "my-key",
		UserAlias: "my-user",
		TeamAlias: "my-team",
	}

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:model", "hash",
		"", "", "", "", "api.openai.com", "success", 0, nil, 0, logCtx.StartTime)

	assert.Equal(t, "my-key", event.KeyAlias)
	assert.Equal(t, "my-user", event.UserAlias)
	assert.Equal(t, "my-team", event.TeamAlias)
}

func TestLogSpendToKafka_PublishesBuiltEvent(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	stub := &stubKafkaManager{enabled: true}
	prx.kafkaLog = stub

	logCtx := testLogCtx(t)
	err := prx.logSpendToKafka(logCtx, "cred", "cred:model", "hash",
		"user-1", "team-1", "org-1", "end-user", "api.openai.com", "success",
		0.001, nil, 1.0, logCtx.StartTime.Add(time.Second))

	require.NoError(t, err)
	require.Len(t, stub.events, 1)
	assert.Equal(t, "req-123", stub.events[0].RequestID)
}

func TestLogSpendToKafka_PublishFailureDoesNotPanic(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	stub := &stubKafkaManager{enabled: true, err: assert.AnError}
	prx.kafkaLog = stub

	logCtx := testLogCtx(t)
	var err error
	assert.NotPanics(t, func() {
		err = prx.logSpendToKafka(logCtx, "cred", "cred:model", "hash",
			"", "", "", "", "api.openai.com", "success",
			0, nil, 0, logCtx.StartTime)
	})
	assert.ErrorIs(t, err, assert.AnError, "the manager's error should be surfaced to the caller")
	assert.Len(t, stub.events, 1, "event should still be attempted even though the manager returns an error")
}
