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

// stubKafkaRawBodyManager mirrors stubKafkaManager for the separate
// raw-bodies write-path (kafkalog.RawBodyManager).
type stubKafkaRawBodyManager struct {
	events  []*kafkalog.RawBodyEvent
	err     error
	enabled bool
}

func (s *stubKafkaRawBodyManager) LogRawBody(event *kafkalog.RawBodyEvent) error {
	s.events = append(s.events, event)
	return s.err
}
func (s *stubKafkaRawBodyManager) IsEnabled() bool       { return s.enabled }
func (s *stubKafkaRawBodyManager) IsHealthy() bool       { return true }
func (s *stubKafkaRawBodyManager) Stats() kafkalog.Stats { return kafkalog.Stats{} }
func (s *stubKafkaRawBodyManager) Shutdown(context.Context) error {
	return nil
}

var _ kafkalog.RawBodyManager = (*stubKafkaRawBodyManager)(nil)

func testLogCtx(t *testing.T) *RequestLogContext {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	return &RequestLogContext{
		RequestID: "req-123",
		EventID:   "req-123",
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

// TestBuildRawBodyEvent_MapsRawBodies checks that buildRawBodyEvent (the
// separate raw-bodies write-path, see kafkalog.RawBodyEvent) carries the
// raw response body through untouched, keyed on the same
// request_id/server_router_id a matching air.errors row would have so the
// two can be joined. RequestBody stays empty here since
// rawBodyStoreRawBody defaults to false on a bare NewTestProxyBuilder --
// see TestBuildRawBodyEvent_RequestBody{Included,Omitted} for that toggle.
func TestBuildRawBodyEvent_MapsRawBodies(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	rawResponse := `{"error":{"message":"the model produced invalid content","type":"invalid_request_error"}}`

	logCtx := testLogCtx(t)
	logCtx.HTTPStatus = 400
	logCtx.ErrorBodyRaw = rawResponse
	logCtx.RequestBodyRaw = "some prompt that should not leak out by default"

	event := prx.buildRawBodyEvent(logCtx, "failure")

	assert.Equal(t, logCtx.spendRequestID(), event.RequestID)
	assert.Equal(t, prx.routerID, event.ServerRouterID)
	assert.Equal(t, "BadRequestError", event.ErrorClass)
	assert.Equal(t, rawResponse, event.ResponseBody)
	assert.Empty(t, event.RequestBody, "RequestBody must stay empty when rawBodyStoreRawBody is off, even if logCtx captured one")
}

// TestBuildRawBodyEvent_RequestBodyIncludedWhenStoreRawBodyEnabled verifies
// the opt-in: kafka.raw_bodies.store_raw_body=true is the only thing that
// lets RequestBody reach the event.
func TestBuildRawBodyEvent_RequestBodyIncludedWhenStoreRawBodyEnabled(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	prx.rawBodyStoreRawBody = true

	logCtx := testLogCtx(t)
	logCtx.HTTPStatus = 400
	logCtx.RequestBodyRaw = `{"messages":[{"role":"user","content":"hello"}]}`

	event := prx.buildRawBodyEvent(logCtx, "failure")
	assert.Equal(t, logCtx.RequestBodyRaw, event.RequestBody)
}

// TestBuildRawBodyEvent_NoErrorClassOnSuccess guards the store_only_errors
// gating (proxy_log.go): once that toggle is disabled, buildRawBodyEvent
// also runs for successful (2xx) requests, and a "success" status must not
// get a misleading ErrorClass derived from mapHTTPStatusToErrorClass.
func TestBuildRawBodyEvent_NoErrorClassOnSuccess(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	logCtx := testLogCtx(t)
	logCtx.HTTPStatus = 200

	event := prx.buildRawBodyEvent(logCtx, "success")
	assert.Empty(t, event.ErrorClass)
}

// TestBuildRawBodyEvent_ErrorClassSetOnMidStreamFailureWithHTTP2xx guards a
// real bug: a mid-stream SSE error (provider returns HTTP 200, then sends an
// error event inside the stream -- see stream.go's finalizeStreamingLog)
// sets logCtx.Status = "failure" without touching HTTPStatus, which stays
// 2xx. Gating ErrorClass on a raw "HTTPStatus >= 400" check (instead of the
// canonical status, like buildKafkaSpendEvent does) left this row's
// ErrorClass empty despite ResponseBody/ClientResponseBody being populated
// and the row actually getting published -- so air.raw_bodies.error_class
// silently disagreed with air.spend_logs.error_class for the exact same
// request.
func TestBuildRawBodyEvent_ErrorClassSetOnMidStreamFailureWithHTTP2xx(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	logCtx := testLogCtx(t)
	logCtx.HTTPStatus = 200 // never updated by the mid-stream-error branch in stream.go
	logCtx.ErrorBodyRaw = `{"error":{"message":"content filtered"}}`
	logCtx.ClientResponseBody = logCtx.ErrorBodyRaw

	event := prx.buildRawBodyEvent(logCtx, "failure")
	assert.NotEmpty(t, event.ErrorClass, "a canonical-failure row must get a non-empty ErrorClass even with a 2xx HTTPStatus")
}

// TestLogRawBodyToKafka_OnlyCalledOnFailure guards the gate in
// logSpendToLiteLLMDB: the raw-bodies write-path must never publish for a
// successful request, even when logCtx.ErrorBodyRaw is a non-empty leftover
// from an earlier failed attempt on a retried request that ultimately
// succeeded (mirrors TestBuildKafkaSpendEvent_ErrorClassOnlyOnFailure's
// stale-retry concern, but at the call-site gate instead of inside the
// builder, since RawBodyEvent has no "status" field of its own to gate on).
func TestLogRawBodyToKafka_OnlyCalledOnFailure(t *testing.T) {
	stub := &stubKafkaRawBodyManager{enabled: true}
	prx := NewTestProxyBuilder().Build()
	prx.rawBodyLog = stub

	logCtx := testLogCtx(t)
	logCtx.HTTPStatus = 400
	logCtx.ErrorBodyRaw = `{"error":"bad request"}`

	prx.logRawBodyToKafka(logCtx, "failure") // simulates what logSpendToLiteLLMDB does when status == "failure"
	require.Len(t, stub.events, 1)
	assert.Equal(t, `{"error":"bad request"}`, stub.events[0].ResponseBody)
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
