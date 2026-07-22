package litellmdb

import (
	"context"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
)

func TestNoopManager(t *testing.T) {
	manager := NewNoopManager()

	t.Run("IsEnabled", func(t *testing.T) {
		assert.False(t, manager.IsEnabled())
	})

	t.Run("IsHealthy", func(t *testing.T) {
		assert.False(t, manager.IsHealthy())
	})

	t.Run("ValidateToken", func(t *testing.T) {
		_, err := manager.ValidateToken(context.Background(), "test-token")
		assert.ErrorIs(t, err, models.ErrModuleDisabled)
	})

	t.Run("ValidateTokenForModel", func(t *testing.T) {
		_, err := manager.ValidateTokenForModel(context.Background(), "test-token", "gpt-4")
		assert.ErrorIs(t, err, models.ErrModuleDisabled)
	})

	t.Run("LogSpend", func(t *testing.T) {
		// Should not panic
		err := manager.LogSpend(&models.SpendLogEntry{RequestID: "test"})
		assert.NoError(t, err)
		err = manager.LogSpend(nil)
		assert.NoError(t, err)
	})

	t.Run("Stats", func(t *testing.T) {
		authStats := manager.AuthCacheStats()
		assert.Equal(t, 0, authStats.Size)

		logStats := manager.SpendLoggerStats()
		assert.Equal(t, 0, logStats.QueueLen)

		connStats := manager.ConnectionStats()
		assert.Nil(t, connStats)
	})

	t.Run("Shutdown", func(t *testing.T) {
		err := manager.Shutdown(context.Background())
		assert.NoError(t, err)
	})
}

func TestDefaultManager_InterfaceCompliance(t *testing.T) {
	// Compile-time check that DefaultManager implements Manager
	var _ Manager = (*DefaultManager)(nil)
	var _ Manager = (*NoopManager)(nil)
}

func TestNew_InvalidConfig(t *testing.T) {
	t.Run("missing database URL", func(t *testing.T) {
		cfg := &models.Config{}
		_, err := New(cfg)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "database_url")
	})

	t.Run("invalid database URL", func(t *testing.T) {
		cfg := &models.Config{
			DatabaseURL: "invalid-url",
		}
		_, err := New(cfg)
		assert.Error(t, err)
	})
}
