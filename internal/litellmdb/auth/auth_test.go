package auth

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHashToken(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "sk- prefixed token",
			input:    "sk-iq0apw_l6s9IJRu2PBVu-g",
			expected: "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
		},
		{
			name:     "already hashed token",
			input:    "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
			expected: "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
		},
		{
			name:     "non sk- token",
			input:    "some-other-token",
			expected: "some-other-token",
		},
		{
			name:     "empty token",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HashToken(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMaskToken(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "empty token",
			input:    "",
			expected: "",
		},
		{
			name:     "1 char - too short",
			input:    "1",
			expected: "***",
		},
		{
			name:     "4 chars - exactly at limit",
			input:    "1234",
			expected: "***",
		},
		{
			name:     "5 chars - first masked",
			input:    "12345",
			expected: "1234...",
		},
		{
			name:     "short token",
			input:    "short",
			expected: "shor...",
		},
		{
			name:     "hashed token - sha256",
			input:    "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
			expected: "f3d2...",
		},
		{
			name:     "typical hashed token",
			input:    "abc123def456ghi789jkl012mno345pqr",
			expected: "abc1...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := security.MaskToken(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestTokenInfo_IsExpired(t *testing.T) {
	t.Run("nil expires - not expired", func(t *testing.T) {
		info := &models.TokenInfo{Expires: nil}
		assert.False(t, info.IsExpired())
	})

	t.Run("future expires - not expired", func(t *testing.T) {
		future := time.Now().UTC().Add(time.Hour)
		info := &models.TokenInfo{Expires: &future}
		assert.False(t, info.IsExpired())
	})

	t.Run("past expires - expired", func(t *testing.T) {
		past := time.Now().UTC().Add(-time.Hour)
		info := &models.TokenInfo{Expires: &past}
		assert.True(t, info.IsExpired())
	})
}

func TestTokenInfo_IsBudgetExceeded(t *testing.T) {
	t.Run("nil max_budget - not exceeded", func(t *testing.T) {
		info := &models.TokenInfo{Spend: 1000, MaxBudget: nil}
		assert.False(t, info.IsBudgetExceeded())
	})

	t.Run("spend < max_budget - not exceeded", func(t *testing.T) {
		maxBudget := 100.0
		info := &models.TokenInfo{Spend: 50, MaxBudget: &maxBudget}
		assert.False(t, info.IsBudgetExceeded())
	})

	t.Run("spend == max_budget - not exceeded (embedded uses >)", func(t *testing.T) {
		maxBudget := 100.0
		info := &models.TokenInfo{Spend: 100, MaxBudget: &maxBudget}
		assert.False(t, info.IsBudgetExceeded())
	})

	t.Run("spend > max_budget - exceeded", func(t *testing.T) {
		maxBudget := 100.0
		info := &models.TokenInfo{Spend: 150, MaxBudget: &maxBudget}
		assert.True(t, info.IsBudgetExceeded())
	})
}

func TestTokenInfo_IsModelAllowed(t *testing.T) {
	t.Run("empty models list - all allowed", func(t *testing.T) {
		info := &models.TokenInfo{Models: nil}
		assert.True(t, info.IsModelAllowed("gpt-4"))
		assert.True(t, info.IsModelAllowed("claude-3"))
	})

	t.Run("model in list - allowed", func(t *testing.T) {
		info := &models.TokenInfo{Models: []string{"gpt-4", "gpt-3.5-turbo"}}
		assert.True(t, info.IsModelAllowed("gpt-4"))
	})

	t.Run("model not in list - not allowed", func(t *testing.T) {
		info := &models.TokenInfo{Models: []string{"gpt-4", "gpt-3.5-turbo"}}
		assert.False(t, info.IsModelAllowed("claude-3"))
	})
}

func TestTokenInfo_Validate(t *testing.T) {
	t.Run("valid token", func(t *testing.T) {
		info := &models.TokenInfo{
			Token:   "test",
			Blocked: false,
		}
		err := info.Validate("")
		assert.NoError(t, err)
	})

	t.Run("blocked token", func(t *testing.T) {
		info := &models.TokenInfo{Blocked: true}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrTokenBlocked)
	})

	t.Run("expired token", func(t *testing.T) {
		past := time.Now().UTC().Add(-time.Hour)
		info := &models.TokenInfo{Expires: &past}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrTokenExpired)
	})

	t.Run("token budget exceeded", func(t *testing.T) {
		maxBudget := 100.0
		info := &models.TokenInfo{Spend: 150, MaxBudget: &maxBudget}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("team budget exceeded (embedded, >)", func(t *testing.T) {
		teamBudget := 100.0
		teamSpend := 150.0
		info := &models.TokenInfo{
			TeamMaxBudget: &teamBudget,
			TeamSpend:     &teamSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("team budget at limit - not exceeded (embedded uses >)", func(t *testing.T) {
		teamBudget := 100.0
		teamSpend := 100.0
		info := &models.TokenInfo{
			TeamMaxBudget: &teamBudget,
			TeamSpend:     &teamSpend,
		}
		err := info.Validate("")
		assert.NoError(t, err)
	})

	t.Run("team member budget exceeded (external, >=)", func(t *testing.T) {
		memberBudget := 100.0
		memberSpend := 100.0 // >= trigger
		info := &models.TokenInfo{
			UserID:              "user1",
			TeamID:              "team1",
			TeamMemberMaxBudget: &memberBudget,
			TeamMemberSpend:     &memberSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("organization budget exceeded (external, >=)", func(t *testing.T) {
		orgBudget := 100.0
		orgSpend := 100.0 // >= trigger
		info := &models.TokenInfo{
			OrganizationID: "org1",
			OrgMaxBudget:   &orgBudget,
			OrgSpend:       &orgSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("user budget exceeded (personal key, embedded, >)", func(t *testing.T) {
		userBudget := 100.0
		userSpend := 150.0
		info := &models.TokenInfo{
			UserID:        "user1",
			TeamID:        "", // Personal key - no team
			UserMaxBudget: &userBudget,
			UserSpend:     &userSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("org member budget exceeded (external, >=)", func(t *testing.T) {
		memberBudget := 100.0
		memberSpend := 100.0
		info := &models.TokenInfo{
			UserID:             "user1",
			OrganizationID:     "org1",
			OrgMemberMaxBudget: &memberBudget,
			OrgMemberSpend:     &memberSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("model not allowed", func(t *testing.T) {
		info := &models.TokenInfo{Models: []string{"gpt-4"}}
		err := info.Validate("claude-3")
		assert.ErrorIs(t, err, models.ErrModelNotAllowed)
	})

	t.Run("model allowed with empty check", func(t *testing.T) {
		info := &models.TokenInfo{Models: []string{"gpt-4"}}
		err := info.Validate("") // Empty model - skip check
		assert.NoError(t, err)
	})
}

func TestCache_Auth_SetGet(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	info := &models.TokenInfo{Token: "test123", UserID: "user1"}
	cache.Set("hash123", info)

	got, ok := cache.Get("hash123")
	assert.True(t, ok)
	assert.Equal(t, "user1", got.UserID)
}

func TestCache_Auth_NotFound(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	_, ok := cache.Get("nonexistent")
	assert.False(t, ok)
}

func TestCache_Auth_TTLExpired(t *testing.T) {
	cache, err := NewCache(100, 10*time.Millisecond)
	require.NoError(t, err)

	cache.Set("hash123", &models.TokenInfo{UserID: "user1"})
	time.Sleep(20 * time.Millisecond)

	_, ok := cache.Get("hash123")
	assert.False(t, ok)
}

func TestCache_Auth_Invalidate(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	cache.Set("hash123", &models.TokenInfo{UserID: "user1"})
	cache.Invalidate("hash123")

	_, ok := cache.Get("hash123")
	assert.False(t, ok)
}

func TestCache_Auth_InvalidateAll(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	cache.Set("hash1", &models.TokenInfo{UserID: "user1"})
	cache.Set("hash2", &models.TokenInfo{UserID: "user2"})
	cache.InvalidateAll()

	assert.Equal(t, 0, cache.Len())
}

func TestCache_Auth_Stats(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	cache.Set("hash123", &models.TokenInfo{UserID: "user1"})

	// One hit
	cache.Get("hash123")
	// One miss
	cache.Get("nonexistent")

	stats := cache.Stats()
	assert.Equal(t, 1, stats.Size)
	assert.Equal(t, uint64(1), stats.Hits)
	assert.Equal(t, uint64(1), stats.Misses)
	assert.Equal(t, 50.0, stats.HitRate)
}

// ==================== Authenticator tests ====================

func TestNewAuthenticator(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	logger := slog.Default()

	auth := NewAuthenticator(nil, cache, logger)
	assert.NotNil(t, auth)
	assert.Equal(t, cache, auth.cache)
	assert.Equal(t, logger, auth.logger)
}

func TestAuthenticator_ValidateToken_EmptyToken(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	info, err := auth.ValidateToken(context.Background(), "")
	assert.Nil(t, info)
	assert.ErrorIs(t, err, models.ErrTokenNotFound)
}

func TestAuthenticator_ValidateToken_FromCache(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with valid token
	tokenInfo := &models.TokenInfo{
		Token:   "test-token",
		UserID:  "user1",
		Blocked: false,
	}
	hashedToken := HashToken("sk-test-token-123")
	cache.Set(hashedToken, tokenInfo)

	// Validate should find it in cache without needing DB
	info, err := auth.ValidateToken(context.Background(), "sk-test-token-123")
	assert.NoError(t, err)
	assert.Equal(t, "user1", info.UserID)
	assert.Equal(t, "test-token", info.Token)

	// Check cache stats
	stats := cache.Stats()
	assert.Equal(t, 1, stats.Size)
	assert.Greater(t, stats.Hits, uint64(0))
}

func TestAuthenticator_ValidateToken_CacheBlockedToken(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with blocked token
	tokenInfo := &models.TokenInfo{
		Token:   "test-token",
		UserID:  "user1",
		Blocked: true,
	}
	hashedToken := HashToken("sk-blocked-token")
	cache.Set(hashedToken, tokenInfo)

	// Validate should fail even if token is cached
	info, err := auth.ValidateToken(context.Background(), "sk-blocked-token")
	assert.Nil(t, info)
	assert.ErrorIs(t, err, models.ErrTokenBlocked)
}

func TestAuthenticator_ValidateToken_CacheExpiredToken(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with expired token
	past := time.Now().UTC().Add(-time.Hour)
	tokenInfo := &models.TokenInfo{
		Token:   "test-token",
		UserID:  "user1",
		Expires: &past,
	}
	hashedToken := HashToken("sk-expired-token")
	cache.Set(hashedToken, tokenInfo)

	// Validate should fail even if token is cached
	info, err := auth.ValidateToken(context.Background(), "sk-expired-token")
	assert.Nil(t, info)
	assert.ErrorIs(t, err, models.ErrTokenExpired)
}

func TestAuthenticator_ValidateToken_CacheBudgetExceeded(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with budget exceeded token
	budget := 100.0
	tokenInfo := &models.TokenInfo{
		Token:     "test-token",
		UserID:    "user1",
		Spend:     150.0,
		MaxBudget: &budget,
	}
	hashedToken := HashToken("sk-over-budget-token")
	cache.Set(hashedToken, tokenInfo)

	// Validate should fail even if token is cached
	info, err := auth.ValidateToken(context.Background(), "sk-over-budget-token")
	assert.Nil(t, info)
	assert.ErrorIs(t, err, models.ErrBudgetExceeded)
}

func TestAuthenticator_ValidateTokenForModel_Success(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with token that allows gpt-4
	tokenInfo := &models.TokenInfo{
		Token:   "test-token",
		UserID:  "user1",
		Blocked: false,
		Models:  []string{"gpt-4", "gpt-3.5-turbo"},
	}
	hashedToken := HashToken("sk-gpt4-token")
	cache.Set(hashedToken, tokenInfo)

	// Should succeed for allowed model
	info, err := auth.ValidateTokenForModel(context.Background(), "sk-gpt4-token", "gpt-4")
	assert.NoError(t, err)
	assert.Equal(t, "user1", info.UserID)
}

func TestAuthenticator_ValidateTokenForModel_ModelNotAllowed(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with token that only allows gpt-4
	tokenInfo := &models.TokenInfo{
		Token:   "test-token",
		UserID:  "user1",
		Blocked: false,
		Models:  []string{"gpt-4"},
	}
	hashedToken := HashToken("sk-gpt4-only-token")
	cache.Set(hashedToken, tokenInfo)

	// Should fail for disallowed model
	info, err := auth.ValidateTokenForModel(context.Background(), "sk-gpt4-only-token", "claude-3")
	assert.Nil(t, info)
	assert.ErrorIs(t, err, models.ErrModelNotAllowed)
}

func TestAuthenticator_ValidateTokenForModel_EmptyToken(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Try to validate with empty token
	info, err := auth.ValidateTokenForModel(context.Background(), "", "gpt-4")
	assert.Nil(t, info)
	assert.ErrorIs(t, err, models.ErrTokenNotFound)
}

func TestAuthenticator_InvalidateToken(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Add token to cache
	tokenInfo := &models.TokenInfo{
		Token:  "test-token",
		UserID: "user1",
	}
	hashedToken := HashToken("sk-to-invalidate")
	cache.Set(hashedToken, tokenInfo)

	// Verify it's in cache
	_, ok := cache.Get(hashedToken)
	assert.True(t, ok)

	// Invalidate it
	auth.InvalidateToken(hashedToken)

	// Verify it's no longer in cache
	_, ok = cache.Get(hashedToken)
	assert.False(t, ok)
}

func TestAuthenticator_CacheStats(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Add token to cache
	tokenInfo := &models.TokenInfo{
		Token:  "test-token",
		UserID: "user1",
	}
	hashedToken := HashToken("sk-stats-token")
	cache.Set(hashedToken, tokenInfo)

	// Access it
	cache.Get(hashedToken)
	cache.Get(hashedToken) // Hit again

	// Get stats
	cache.Get("nonexistent") // Miss

	stats := auth.CacheStats()
	assert.Equal(t, 1, stats.Size)
	assert.Greater(t, stats.Hits, uint64(0))
	assert.Greater(t, stats.Misses, uint64(0))
}

func TestAuthenticator_ValidateToken_NonSKToken(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Pre-populate cache with non-sk- token (already hashed)
	tokenInfo := &models.TokenInfo{
		Token:   "pre-hashed-token",
		UserID:  "user1",
		Blocked: false,
	}
	hashedToken := "pre-hashed-token" // Not prefixed with sk-
	cache.Set(hashedToken, tokenInfo)

	// Validate should find it using the same hashing logic (no-op for non-sk-)
	info, err := auth.ValidateToken(context.Background(), "pre-hashed-token")
	assert.NoError(t, err)
	assert.Equal(t, "user1", info.UserID)
}

func TestAuthenticator_CacheHitRate(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	// Add multiple tokens
	for i := 1; i <= 5; i++ {
		tokenInfo := &models.TokenInfo{
			Token:  "test-token",
			UserID: string(rune(i + 48)), // "1", "2", etc
		}
		hashedToken := HashToken("sk-token-" + string(rune(i+48)))
		cache.Set(hashedToken, tokenInfo)
	}

	// Access some tokens multiple times
	cache.Get(HashToken("sk-token-1"))
	cache.Get(HashToken("sk-token-1")) // Hit
	cache.Get(HashToken("sk-token-2"))
	cache.Get(HashToken("sk-token-2")) // Hit

	// Miss
	cache.Get("nonexistent")

	stats := auth.CacheStats()
	assert.Greater(t, stats.HitRate, 0.0)
	assert.LessOrEqual(t, stats.HitRate, 100.0)
}
