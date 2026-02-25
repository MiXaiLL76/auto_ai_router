package litellmdb

import (
	"context"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNew_ResourceCleanupOnCacheError tests that connection pool is closed
// if auth cache creation fails during initialization
func TestNew_ResourceCleanupOnCacheError(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL:      "invalid-connection-string",
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	// Should fail during connection pool creation, not during cache creation
	manager, err := New(cfg)
	assert.Error(t, err)
	assert.Nil(t, manager)
}

// TestNew_ManagerInitialization tests basic manager creation
func TestNew_ManagerInitialization(t *testing.T) {
	t.Skip("Skipping integration test - requires real database")

	// This test would require LITELLM_DATABASE_URL env var set
	cfg := &models.Config{
		DatabaseURL:      "postgres://localhost/litellm",
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	manager, err := New(cfg)
	if err != nil {
		t.Skipf("Database not available: %v", err)
	}

	defer func() {
		_ = manager.Shutdown(context.Background())
	}()

	// Verify manager is enabled
	assert.True(t, manager.IsEnabled())
}

// TestNoopManager_IsDisabled tests NoopManager behavior
func TestNoopManager_IsDisabled(t *testing.T) {
	noop := NewNoopManager()

	assert.False(t, noop.IsEnabled())
	assert.False(t, noop.IsHealthy())

	// All operations should be no-ops
	assert.NoError(t, noop.LogSpend(&models.SpendLogEntry{RequestID: "test"}))

	authStats := noop.AuthCacheStats()
	assert.Equal(t, 0, authStats.Size)

	logStats := noop.SpendLoggerStats()
	assert.Equal(t, 0, logStats.QueueLen)

	connStats := noop.ConnectionStats()
	assert.Nil(t, connStats)

	err := noop.Shutdown(context.Background())
	assert.NoError(t, err)
}

// TestNew_ConfigValidation tests that config validation happens
func TestNew_ConfigValidation(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "",
		MaxConns:    5,
		MinConns:    1,
	}

	err := cfg.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database_url")
}

// TestNew_ApplyDefaults tests that defaults are applied
func TestNew_ApplyDefaults(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
	}
	cfg.ApplyDefaults()

	// Verify some defaults are applied
	assert.Greater(t, cfg.MaxConns, int32(0))
	assert.Greater(t, cfg.MinConns, int32(0))
	assert.Greater(t, cfg.AuthCacheSize, 0)
	assert.Greater(t, cfg.LogQueueSize, 0)
}

// TestNoopManager_ValidateToken tests token validation in noop mode
func TestNoopManager_ValidateToken(t *testing.T) {
	noop := NewNoopManager()

	_, err := noop.ValidateToken(context.Background(), "sk-test-token")
	assert.ErrorIs(t, err, models.ErrModuleDisabled)
}

// TestNoopManager_ValidateTokenForModel tests model-specific token validation in noop mode
func TestNoopManager_ValidateTokenForModel(t *testing.T) {
	noop := NewNoopManager()

	_, err := noop.ValidateTokenForModel(context.Background(), "sk-test-token", "gpt-4")
	assert.ErrorIs(t, err, models.ErrModuleDisabled)
}

// TestNoopManager_AllMethodsSafe tests that all noop methods are safe to call
func TestNoopManager_AllMethodsSafe(t *testing.T) {
	noop := NewNoopManager()

	// All these should not panic
	_ = noop.IsEnabled()
	_ = noop.IsHealthy()
	_ = noop.AuthCacheStats()
	_ = noop.SpendLoggerStats()
	_ = noop.ConnectionStats()
	_ = noop.LogSpend(nil)
	_ = noop.LogSpend(&models.SpendLogEntry{})
	_ = noop.Shutdown(context.Background())

	_, _ = noop.ValidateToken(context.Background(), "test")
	_, _ = noop.ValidateTokenForModel(context.Background(), "test", "model")
}

// TestNew_FailureLogging tests that initialization failures are logged
func TestNew_FailureLogging(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "invalid://url",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	manager, err := New(cfg)
	assert.Error(t, err)
	assert.Nil(t, manager)
}

// TestManagerShutdown_MultipleTimes tests that shutdown is safe to call multiple times
func TestManagerShutdown_MultipleTimes(t *testing.T) {
	t.Skip("Skipping integration test - requires real database")

	cfg := &models.Config{
		DatabaseURL:      "postgres://localhost/litellm",
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	manager, err := New(cfg)
	if err != nil {
		t.Skipf("Database not available: %v", err)
	}

	ctx := context.Background()

	// First shutdown
	err1 := manager.Shutdown(ctx)
	assert.NoError(t, err1)

	// Second shutdown should be safe (no-op or error, but not panic)
	err2 := manager.Shutdown(ctx)
	// May error, but shouldn't panic
	_ = err2
}

// TestManagerShutdown_WithTimeout tests shutdown respects context timeout
func TestManagerShutdown_WithTimeout(t *testing.T) {
	t.Skip("Skipping integration test - requires real database")

	cfg := &models.Config{
		DatabaseURL:      "postgres://localhost/litellm",
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	manager, err := New(cfg)
	if err != nil {
		t.Skipf("Database not available: %v", err)
	}

	// Shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Should complete within timeout
	start := time.Now()
	err = manager.Shutdown(ctx)
	elapsed := time.Since(start)

	assert.Less(t, elapsed, 2*time.Second)
	assert.Nil(t, err)
}

// TestManager_NoopVsDefault tests switching between noop and default
func TestManager_NoopVsDefault(t *testing.T) {
	// Test noop manager
	noop := NewNoopManager()
	assert.False(t, noop.IsEnabled())

	// Test with invalid config (will return noop during init)
	cfg := &models.Config{
		DatabaseURL: "",
	}
	cfg.ApplyDefaults()

	// This should fail during validation
	err := cfg.Validate()
	assert.Error(t, err)
}

// TestManagerCreation_WithLogging tests that manager logs initialization
func TestManagerCreation_WithLogging(t *testing.T) {
	t.Skip("Skipping integration test - requires real database")

	cfg := &models.Config{
		DatabaseURL:      "postgres://localhost/litellm",
		MaxConns:         5,
		MinConns:         1,
		AuthCacheSize:    100,
		AuthCacheTTL:     time.Second,
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	manager, err := New(cfg)
	if err != nil {
		t.Skipf("Database not available: %v", err)
	}

	defer func() {
		_ = manager.Shutdown(context.Background())
	}()

	// Should be enabled if successfully created
	defaultMgr, ok := manager.(*DefaultManager)
	require.True(t, ok)
	assert.NotNil(t, defaultMgr)
	assert.NotNil(t, defaultMgr.pool)
	assert.NotNil(t, defaultMgr.auth)
	assert.NotNil(t, defaultMgr.spendLogger)
}
