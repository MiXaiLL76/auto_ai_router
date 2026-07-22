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
