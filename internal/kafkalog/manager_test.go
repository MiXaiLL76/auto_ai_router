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

func TestNoopErrorBodyManager(t *testing.T) {
	m := NewNoopErrorBodyManager()

	assert.False(t, m.IsEnabled())
	assert.False(t, m.IsHealthy())
	assert.Equal(t, Stats{}, m.Stats())
	assert.NoError(t, m.LogErrorBody(&ErrorBodyEvent{RequestID: "req-1"}))
	assert.NoError(t, m.LogErrorBody(nil))
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

// TestDefaultErrorBodyManager_LogErrorBody_NilEvent mirrors
// TestDefaultManager_LogSpend_NilEvent for the error-body write-path.
func TestDefaultErrorBodyManager_LogErrorBody_NilEvent(t *testing.T) {
	m := &DefaultErrorBodyManager{logger: newTestErrorBodyLogger(10)}
	assert.NoError(t, m.LogErrorBody(nil))
	assert.Equal(t, 0, m.logger.Stats().QueueLen)
}
