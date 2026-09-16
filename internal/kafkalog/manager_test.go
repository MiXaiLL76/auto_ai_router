package kafkalog

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNoopManager(t *testing.T) {
	m := NewNoopManager()

	assert.False(t, m.IsEnabled())
	assert.False(t, m.IsHealthy())
	assert.Equal(t, Stats{}, m.Stats())
	assert.NoError(t, m.LogSpend(&SpendEvent{RequestID: "req-1"}))
	assert.NoError(t, m.LogSpend(nil))
	assert.NoError(t, m.Shutdown(context.Background()))
}

func TestNoopRawBodyManager(t *testing.T) {
	m := NewNoopRawBodyManager()

	assert.False(t, m.IsEnabled())
	assert.False(t, m.IsHealthy())
	assert.Equal(t, Stats{}, m.Stats())
	assert.NoError(t, m.LogRawBody(&RawBodyEvent{RequestID: "req-1"}))
	assert.NoError(t, m.LogRawBody(nil))
	assert.NoError(t, m.Shutdown(context.Background()))
}

// TestDefaultManager_LogSpend_NilEvent guards the nil-check that moved from
// Logger[T].Log (which can no longer safely compare a generic T to nil, see
// Logger.Log's doc comment) up to this wrapper, which knows the concrete
// *SpendEvent type.
func TestDefaultManager_LogSpend_NilEvent(t *testing.T) {
	m := &DefaultManager{logger: newTestLogger(10)}
	assert.NoError(t, m.LogSpend(nil))
	assert.Equal(t, 0, m.logger.Stats().QueueLen)
}

// TestDefaultRawBodyManager_LogRawBody_NilEvent mirrors
// TestDefaultManager_LogSpend_NilEvent for the raw-body write-path.
func TestDefaultRawBodyManager_LogRawBody_NilEvent(t *testing.T) {
	m := &DefaultRawBodyManager{logger: newTestRawBodyLogger(10)}
	assert.NoError(t, m.LogRawBody(nil))
	assert.Equal(t, 0, m.logger.Stats().QueueLen)
}
