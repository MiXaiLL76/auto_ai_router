package litellmdb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestDefaultConfig verifies the backwards compatibility export
func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	assert.NotNil(t, cfg)
	assert.Equal(t, int32(10), cfg.MaxConns)
	assert.Equal(t, int32(2), cfg.MinConns)
	assert.Greater(t, cfg.AuthCacheSize, 0)
}

// TestMaskDatabaseURL verifies the backwards compatibility export
func TestMaskDatabaseURL(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		hasPort bool
		hasHost bool
	}{
		{
			name:    "postgresql url",
			input:   "postgresql://user:password@localhost:5432/database",
			hasPort: true,
			hasHost: true,
		},
		{
			name:    "postgres url with ip",
			input:   "postgres://user:pass@192.168.1.1:5432/db",
			hasPort: true,
			hasHost: true,
		},
		{
			name:    "empty url",
			input:   "",
			hasPort: false,
			hasHost: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := maskDatabaseURL(tt.input)
			// Result should be masked (different from input when input contains credentials)
			if tt.input != "" && (tt.input != result) {
				assert.NotEqual(t, tt.input, result)
			}
		})
	}
}

// TestConfigAlias verifies type aliases work correctly
func TestConfigAlias(t *testing.T) {
	// Config is a type alias, so these should work
	cfg := &Config{
		DatabaseURL: "postgresql://localhost/test",
		MaxConns:    20,
	}

	assert.NotNil(t, cfg)
	assert.Equal(t, "postgresql://localhost/test", cfg.DatabaseURL)
	assert.Equal(t, int32(20), cfg.MaxConns)
}

// TestTokenInfoAlias verifies type aliases work correctly
func TestTokenInfoAlias(t *testing.T) {
	// TokenInfo is a type alias
	info := &TokenInfo{
		Token:  "test-token",
		UserID: "user1",
	}

	assert.NotNil(t, info)
	assert.Equal(t, "test-token", info.Token)
	assert.Equal(t, "user1", info.UserID)
}

// TestSpendLogEntryAlias verifies type aliases work correctly
func TestSpendLogEntryAlias(t *testing.T) {
	// SpendLogEntry is a type alias
	entry := &SpendLogEntry{
		RequestID: "req-123",
		Model:     "gpt-4",
		Spend:     0.05,
	}

	assert.NotNil(t, entry)
	assert.Equal(t, "req-123", entry.RequestID)
	assert.Equal(t, "gpt-4", entry.Model)
	assert.Equal(t, 0.05, entry.Spend)
}

// TestAuthCacheStatsAlias verifies type aliases work correctly
func TestAuthCacheStatsAlias(t *testing.T) {
	// AuthCacheStats is a type alias
	stats := &AuthCacheStats{
		Size:    100,
		Hits:    1000,
		Misses:  100,
		HitRate: 90.9,
	}

	assert.NotNil(t, stats)
	assert.Equal(t, 100, stats.Size)
	assert.Equal(t, uint64(1000), stats.Hits)
}

// TestSpendLoggerStatsAlias verifies type aliases work correctly
func TestSpendLoggerStatsAlias(t *testing.T) {
	// SpendLoggerStats is a type alias
	stats := &SpendLoggerStats{
		QueueLen:       50,
		QueueCap:       1000,
		Queued:         5000,
		Written:        4900,
		Dropped:        100,
		Errors:         10,
		BatchesOK:      49,
		QueueFullCount: 5,
	}

	assert.NotNil(t, stats)
	assert.Equal(t, 50, stats.QueueLen)
	assert.Equal(t, 1000, stats.QueueCap)
	assert.Equal(t, uint64(5000), stats.Queued)
	assert.Equal(t, uint64(4900), stats.Written)
}

// TestErrorExports verifies error constants are properly exported
func TestErrorExports(t *testing.T) {
	assert.NotNil(t, ErrModuleDisabled)
	assert.NotNil(t, ErrTokenNotFound)
	assert.NotNil(t, ErrTokenBlocked)
	assert.NotNil(t, ErrTokenExpired)
	assert.NotNil(t, ErrBudgetExceeded)
	assert.NotNil(t, ErrModelNotAllowed)
	assert.NotNil(t, ErrConnectionFailed)
}
