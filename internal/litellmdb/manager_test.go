package litellmdb

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// Integration test - requires real DB
func TestDefaultManager_Integration(t *testing.T) {
	dbURL := os.Getenv("LITELLM_DATABASE_URL")
	if dbURL == "" {
		t.Skip("LITELLM_DATABASE_URL not set, skipping integration test")
	}

	cfg := &models.Config{
		DatabaseURL:      dbURL,
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: time.Second,
	}

	manager, err := New(cfg)
	require.NoError(t, err)

	defer func() {
		_ = manager.Shutdown(context.Background())
	}()

	t.Run("IsEnabled", func(t *testing.T) {
		assert.True(t, manager.IsEnabled())
	})

	t.Run("IsHealthy", func(t *testing.T) {
		assert.True(t, manager.IsHealthy())
	})

	t.Run("ValidateToken_NotFound", func(t *testing.T) {
		_, err := manager.ValidateToken(context.Background(), "sk-nonexistent-token")
		assert.ErrorIs(t, err, models.ErrTokenNotFound)
	})

	t.Run("ConnectionStats", func(t *testing.T) {
		stats := manager.ConnectionStats()
		assert.NotNil(t, stats)
		assert.GreaterOrEqual(t, stats.TotalConns(), int32(1))
	})

	t.Run("LogSpend", func(t *testing.T) {
		entry := &models.SpendLogEntry{
			RequestID: "test-" + time.Now().UTC().Format("20060102150405.000"),
			CallType:  "/v1/chat/completions",
			Model:     "gpt-4",
			StartTime: time.Now().UTC(),
			EndTime:   time.Now().UTC(),
			Status:    "success",
		}

		// Should not panic
		err := manager.LogSpend(entry)
		assert.NoError(t, err)

		// Wait for flush
		time.Sleep(2 * time.Second)

		stats := manager.SpendLoggerStats()
		assert.GreaterOrEqual(t, stats.Written, uint64(1))
	})
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
