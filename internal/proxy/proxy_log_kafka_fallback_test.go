package proxy

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stubLiteLLMManager is a minimal litellmdb.Manager test double that records
// every entry passed to LogSpend. Embeds NoopManager so it only needs to
// override what these tests actually exercise.
type stubLiteLLMManager struct {
	litellmdb.NoopManager
	loggedEntries []*dbmodels.SpendLogEntry
}

func (s *stubLiteLLMManager) IsEnabled() bool { return true }

func (s *stubLiteLLMManager) SpendLoggingEnabled() bool { return true }

func (s *stubLiteLLMManager) LogSpend(entry *dbmodels.SpendLogEntry) error {
	s.loggedEntries = append(s.loggedEntries, entry)
	return nil
}

var _ litellmdb.Manager = (*stubLiteLLMManager)(nil)

// TestLogSpendToLiteLLMDB_FlagsKafkaFallbackOnQueueFull verifies the review
// fix for the Kafka queue-overflow finding: when publishing to Kafka fails
// (kafkalog.ErrQueueFull, i.e. the queue was full and the 5s backpressure
// wait timed out), the row that's about to be written to LiteLLM_SpendLogs
// anyway gets flagged in its metadata so a background job can find it later
// and re-publish it, instead of the event being lost entirely.
func TestLogSpendToLiteLLMDB_FlagsKafkaFallbackOnQueueFull(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	kafkaStub := &stubKafkaManager{enabled: true, err: kafkalog.ErrQueueFull}
	prx.kafkaLog = kafkaStub

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-4o-mini", &routermodels.ModelPrice{
		InputCostPerToken: 0.000001, OutputCostPerToken: 0.000002,
	})

	logCtx := testLogCtx(t)

	err := prx.logSpendToLiteLLMDB(logCtx)
	require.NoError(t, err)

	require.Len(t, dbStub.loggedEntries, 1)
	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(dbStub.loggedEntries[0].Metadata), &metadata))

	assert.Equal(t, true, metadata["kafka_fallback"])
	assert.Equal(t, "queue_full", metadata["kafka_fallback_reason"])
}

// TestLogSpendToLiteLLMDB_NoKafkaFallbackFlagOnSuccess verifies the flag is
// absent when Kafka publishing succeeds, so successful rows aren't picked up
// by the resend job.
func TestLogSpendToLiteLLMDB_NoKafkaFallbackFlagOnSuccess(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	kafkaStub := &stubKafkaManager{enabled: true}
	prx.kafkaLog = kafkaStub

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-4o-mini", &routermodels.ModelPrice{
		InputCostPerToken: 0.000001, OutputCostPerToken: 0.000002,
	})

	logCtx := testLogCtx(t)

	err := prx.logSpendToLiteLLMDB(logCtx)
	require.NoError(t, err)

	require.Len(t, dbStub.loggedEntries, 1)
	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(dbStub.loggedEntries[0].Metadata), &metadata))

	_, hasFlag := metadata["kafka_fallback"]
	assert.False(t, hasFlag)
}

func TestLogSpendToLiteLLMDB_NormalizesTokenUsageBeforePersisting(t *testing.T) {
	prx := NewTestProxyBuilder().Build()

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-4o-mini", &routermodels.ModelPrice{
		InputCostPerToken: 0.000001, OutputCostPerToken: 0.000002,
	})

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

	err := prx.logSpendToLiteLLMDB(logCtx)
	require.NoError(t, err)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 0, entry.PromptTokens)
	assert.Equal(t, 50, entry.CompletionTokens)
	assert.Equal(t, 50, entry.TotalTokens)

	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(entry.Metadata), &metadata))
	usageObject := metadata["usage_object"].(map[string]interface{})
	promptDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(0), promptDetails["audio_tokens"])
	assert.Equal(t, float64(0), promptDetails["cached_tokens"])
	assert.Equal(t, float64(0), promptDetails["cached_audio_tokens"])
	ttlDetails := promptDetails["cache_creation_token_details"].(map[string]interface{})
	assert.Equal(t, float64(8), ttlDetails["ephemeral_5m_input_tokens"])
	assert.Equal(t, float64(2), ttlDetails["ephemeral_1h_input_tokens"])
}

func TestLogSpendToLiteLLMDB_PreservesTeamID(t *testing.T) {
	tests := []struct {
		name       string
		teamID     string
		expectedID string
	}{
		{name: "no team", expectedID: ""},
		{name: "token team", teamID: "team-1", expectedID: "team-1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().Build()
			dbStub := &stubLiteLLMManager{}
			prx.LiteLLMDB = dbStub
			setTestModelPrice(prx, "gpt-4o-mini", &routermodels.ModelPrice{
				InputCostPerToken: 0.000001, OutputCostPerToken: 0.000002,
			})

			logCtx := testLogCtx(t)
			logCtx.TokenInfo = &litellmdb.TokenInfo{TeamID: tt.teamID}

			require.NoError(t, prx.logSpendToLiteLLMDB(logCtx))
			require.Len(t, dbStub.loggedEntries, 1)
			assert.Equal(t, tt.expectedID, dbStub.loggedEntries[0].TeamID)
		})
	}
}

func TestLogSpendToLiteLLMDB_UsesClientResponseID(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub

	logCtx := testLogCtx(t)
	logCtx.ClientResponseID = "chatcmpl-client-123"

	require.NoError(t, prx.logSpendToLiteLLMDB(logCtx))
	require.Len(t, dbStub.loggedEntries, 1)
	assert.Equal(t, "chatcmpl-client-123", dbStub.loggedEntries[0].RequestID)
}

func TestLogSpendToLiteLLMDB_BillsAliasPriceBeforeRealModelPrice(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	registry := routermodels.NewModelPriceRegistry()
	registry.Update(map[string]*routermodels.ModelPrice{
		"gpt-5.2-chat": {
			InputCostPerToken:       0.000001575,
			OutputCostPerToken:      0.0000126,
			CacheReadInputTokenCost: 0.0000001575,
		},
		"gpt-chat-latest": {
			InputCostPerToken:  0.0000045,
			OutputCostPerToken: 0.000027,
		},
	})
	prx.priceRegistry = registry

	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub

	logCtx := testLogCtx(t)
	logCtx.ModelID = "gpt-5.2-chat"
	logCtx.RealModelID = "gpt-chat-latest"
	logCtx.TokenUsage.PromptTokens = 25869
	logCtx.TokenUsage.CompletionTokens = 318
	logCtx.TokenUsage.ReasoningTokens = 64

	err := prx.logSpendToLiteLLMDB(logCtx)
	require.NoError(t, err)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.InDelta(t, 0.044750475, entry.Spend, 0.000000001)

	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(entry.Metadata), &metadata))
	costBreakdown, ok := metadata["cost_breakdown"].(map[string]interface{})
	require.True(t, ok)
	assert.InDelta(t, 0.040743675, costBreakdown["input_cost"].(float64), 0.000000001)
	assert.InDelta(t, 0.0032004, costBreakdown["output_cost"].(float64), 0.000000001)
	assert.InDelta(t, 0.0008064, costBreakdown["reasoning_cost"].(float64), 0.000000001)
	assert.InDelta(t, entry.Spend, costBreakdown["total_cost"].(float64), 0.000000001)
}

func TestLogSpendToLiteLLMDB_RejectsUnknownPrice(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub

	err := prx.logSpendToLiteLLMDB(testLogCtx(t))

	require.ErrorContains(t, err, "model price unavailable")
	assert.Empty(t, dbStub.loggedEntries)
}

// TestLogSpendToLiteLLMDB_WritesZeroCostRowWhenNoUsageAndPriceUnavailable
// guards against a regression of PR #96/#115 follow-up review TODO 2: rows
// for requests rejected before any provider was contacted (e.g. the
// "no credentials available" 429 path, which never resolves ModelPrice) used
// to be silently dropped once price-lookup failure became a hard error. Since
// no usage was ever incurred, a $0 row must still be written — this is
// distinct from TestLogSpendToLiteLLMDB_RejectsUnknownPrice, which covers the
// case where real, non-zero usage exists and pricing is genuinely missing.
func TestLogSpendToLiteLLMDB_WritesZeroCostRowWhenNoUsageAndPriceUnavailable(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub

	logCtx := testLogCtx(t)
	logCtx.TokenUsage = nil
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusTooManyRequests

	err := prx.logSpendToLiteLLMDB(logCtx)
	require.NoError(t, err)

	require.Len(t, dbStub.loggedEntries, 1)
	assert.Zero(t, dbStub.loggedEntries[0].Spend)
}

func TestLogSpendToLiteLLMDB_ChargesImagesWithoutProviderUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-4o-mini", &routermodels.ModelPrice{OutputCostPerImage: 0.04})
	logCtx := testLogCtx(t)
	logCtx.IsImageGeneration = true
	logCtx.ImageCount = 2
	logCtx.TokenUsage = nil

	require.NoError(t, prx.logSpendToLiteLLMDB(logCtx))

	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, 0.08, dbStub.loggedEntries[0].Spend, 1e-12)
}

func TestLogSpendToLiteLLMDB_DoesNotChargeFailedImageRequest(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-4o-mini", &routermodels.ModelPrice{OutputCostPerImage: 0.04})
	logCtx := testLogCtx(t)
	logCtx.IsImageGeneration = true
	logCtx.ImageCount = 2
	logCtx.TokenUsage = nil
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusBadRequest

	require.NoError(t, prx.logSpendToLiteLLMDB(logCtx))

	require.Len(t, dbStub.loggedEntries, 1)
	assert.Zero(t, dbStub.loggedEntries[0].Spend)
}
