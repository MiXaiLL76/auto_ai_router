package proxy

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stubLiteLLMManager is a minimal litellmdb.Manager test double that records
// every entry passed to LogSpend. Embeds NoopManager so it only needs to
// override what these tests actually exercise.
type stubLiteLLMManager struct {
	litellmdb.NoopManager
	loggedEntries []*models.SpendLogEntry
}

func (s *stubLiteLLMManager) IsEnabled() bool { return true }

func (s *stubLiteLLMManager) LogSpend(entry *models.SpendLogEntry) error {
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

	logCtx := testLogCtx(t)

	err := prx.logSpendToLiteLLMDB(logCtx)
	require.NoError(t, err)

	require.Len(t, dbStub.loggedEntries, 1)
	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(dbStub.loggedEntries[0].Metadata), &metadata))

	_, hasFlag := metadata["kafka_fallback"]
	assert.False(t, hasFlag)
}
