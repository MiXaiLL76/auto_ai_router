package health

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDBHealthChecker_NewAndDefaults(t *testing.T) {
	hc := NewDBHealthChecker()

	assert.True(t, hc.IsHealthy(), "new checker should start healthy")
	assert.True(t, hc.IsDBHealthy(), "IsDBHealthy should be alias for IsHealthy")
	assert.NotNil(t, hc.GetHealthValue(), "health value pointer should not be nil")
}

func TestDBHealthChecker_SetHealthy(t *testing.T) {
	hc := NewDBHealthChecker()

	// Start healthy
	assert.True(t, hc.IsHealthy())

	// Set unhealthy
	hc.SetHealthy(false)
	assert.False(t, hc.IsHealthy())
	assert.False(t, hc.IsDBHealthy())

	// Set healthy again
	hc.SetHealthy(true)
	assert.True(t, hc.IsHealthy())
	assert.True(t, hc.IsDBHealthy())
}

func TestDBHealthChecker_NilSafety(t *testing.T) {
	t.Run("nil_receiver", func(t *testing.T) {
		var hc *DBHealthChecker
		// Should not panic, default to healthy
		assert.True(t, hc.IsHealthy())
		assert.True(t, hc.IsDBHealthy())
		assert.Nil(t, hc.GetHealthValue())

		// SetHealthy on nil should not panic
		hc.SetHealthy(false)
		hc.SetHealthy(true)
	})

	t.Run("nil_dbHealthy", func(t *testing.T) {
		hc := &DBHealthChecker{dbHealthy: nil}
		assert.True(t, hc.IsHealthy(), "nil dbHealthy should default to healthy")
		// Should not panic
		hc.SetHealthy(false)
	})
}
