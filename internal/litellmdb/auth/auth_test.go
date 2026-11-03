package auth

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
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

	t.Run("spend == max_budget - exceeded (embedded uses >=)", func(t *testing.T) {
		maxBudget := 100.0
		info := &models.TokenInfo{Spend: 100, MaxBudget: &maxBudget}
		assert.True(t, info.IsBudgetExceeded())
	})

	t.Run("zero max_budget - exceeded before any spend", func(t *testing.T) {
		maxBudget := 0.0
		info := &models.TokenInfo{Spend: 0, MaxBudget: &maxBudget}
		assert.True(t, info.IsBudgetExceeded())
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

	// LiteLLM stores these sentinel values inside VerificationToken.models
	// instead of real model names - confirmed present in a production dump
	// (178 keys use all-proxy-models, 42 use all-team-models).
	t.Run("all-proxy-models sentinel - any model allowed", func(t *testing.T) {
		info := &models.TokenInfo{Models: []string{"all-proxy-models"}}
		assert.True(t, info.IsModelAllowed("gpt-4"))
		assert.True(t, info.IsModelAllowed("claude-3"))
	})

	t.Run("all-team-models sentinel with no team - unrestricted", func(t *testing.T) {
		info := &models.TokenInfo{Models: []string{"all-team-models"}, TeamID: ""}
		assert.True(t, info.IsModelAllowed("gpt-4"))
	})

	t.Run("all-team-models sentinel - resolves against team allow-list", func(t *testing.T) {
		info := &models.TokenInfo{
			Models:     []string{"all-team-models"},
			TeamID:     "team1",
			TeamModels: []string{"gpt-4"},
		}
		assert.True(t, info.IsModelAllowed("gpt-4"))
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

	t.Run("team budget exceeded (embedded, >=)", func(t *testing.T) {
		teamBudget := 100.0
		teamSpend := 150.0
		info := &models.TokenInfo{
			TeamMaxBudget: &teamBudget,
			TeamSpend:     &teamSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("team budget at limit - exceeded (embedded uses >=)", func(t *testing.T) {
		teamBudget := 100.0
		teamSpend := 100.0
		info := &models.TokenInfo{
			TeamMaxBudget: &teamBudget,
			TeamSpend:     &teamSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("zero token budget - rejected on the first request", func(t *testing.T) {
		maxBudget := 0.0
		info := &models.TokenInfo{Spend: 0, MaxBudget: &maxBudget}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
	})

	t.Run("zero user budget - rejected on the first request", func(t *testing.T) {
		userBudget := 0.0
		userSpend := 0.0
		info := &models.TokenInfo{
			UserID:        "user1",
			TeamID:        "",
			UserMaxBudget: &userBudget,
			UserSpend:     &userSpend,
		}
		err := info.Validate("")
		assert.ErrorIs(t, err, models.ErrBudgetExceeded)
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

	t.Run("user budget exceeded (personal key, embedded, >=)", func(t *testing.T) {
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

	t.Run("orphan user reference keeps optional user policy unrestricted", func(t *testing.T) {
		// Production contains active tokens whose user_id no longer has a
		// LiteLLM_UserTable row. LEFT JOIN fields therefore scan as nil while
		// the token identity itself remains present and valid.
		info := &models.TokenInfo{
			UserID:     "deleted-owner",
			UserModels: nil,
		}
		assert.NoError(t, info.Validate("gpt-4"))
	})

	t.Run("dangling team reference fails closed", func(t *testing.T) {
		// Unlike an orphan user (intentionally unrestricted, see above), a
		// team_id whose LiteLLM_TeamTable row is gone must deny: the team
		// scope cannot be resolved, and an empty scope would otherwise be
		// treated as unrestricted — a fail-open for all-team-models keys.
		info := &models.TokenInfo{
			Models:       []string{models.AllTeamModels},
			TeamID:       "deleted-team",
			TeamDangling: true,
		}
		assert.ErrorIs(t, info.Validate("gpt-4"), models.ErrModelNotAllowed)
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

func TestAuthenticator_CostMarginConfigs(t *testing.T) {
	margin := func(pct float64) []byte {
		return []byte(fmt.Sprintf(`{"cost_margin_config": {"global": %v}}`, pct))
	}
	key, user, team, org := margin(0.1), margin(0.2), margin(0.3), margin(0.4)

	tests := []struct {
		name string
		info models.TokenInfo
		want []float64
	}{
		{"personal key", models.TokenInfo{UserID: "u"}, []float64{0.1, 0.2}},
		{"team key of a user", models.TokenInfo{UserID: "u", TeamID: "t", OrganizationID: "o"}, []float64{0.1, 0.3, 0.4}},
		{"team key without organization", models.TokenInfo{TeamID: "t"}, []float64{0.1, 0.3, 0.4}},
		{"organization key of a user", models.TokenInfo{UserID: "u", OrganizationID: "o"}, []float64{0.1, 0.4}},
	}
	auth := NewAuthenticator(nil, nil, slog.Default()).WithCostMargin(true)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configs := auth.costMarginConfigs(&tt.info, key, user, team, org)
			var got []float64
			for _, cfg := range configs {
				got = append(got, cfg["global"].Percentage)
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAuthenticator_CostMarginConfigs_SkipsMissingEntities(t *testing.T) {
	auth := NewAuthenticator(nil, nil, slog.Default()).WithCostMargin(true)
	info := &models.TokenInfo{TeamID: "t"}
	configs := auth.costMarginConfigs(info, []byte(`{}`), nil, []byte(`{"cost_margin_config": {"global": 0.3}}`), nil)
	assert.Equal(t, []models.CostMarginConfig{{"global": {Percentage: 0.3}}}, configs)
}

func TestAuthenticator_CostMarginConfigs_DisabledByDefault(t *testing.T) {
	auth := NewAuthenticator(nil, nil, slog.Default())
	info := &models.TokenInfo{TeamID: "t"}
	margin := []byte(`{"cost_margin_config": {"global": 0.3}}`)
	assert.Nil(t, auth.costMarginConfigs(info, margin, margin, margin, margin))
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

func TestAuthenticator_ValidateTokenForModel_AllProxyModelsSentinel(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	tokenInfo := &models.TokenInfo{
		Token:   "test-token",
		UserID:  "user1",
		Blocked: false,
		Models:  []string{"all-proxy-models"},
	}
	hashedToken := HashToken("sk-all-proxy-models-token")
	cache.Set(hashedToken, tokenInfo)

	info, err := auth.ValidateTokenForModel(context.Background(), "sk-all-proxy-models-token", "claude-3")
	assert.NoError(t, err)
	assert.Equal(t, "user1", info.UserID)
}

func TestAuthenticator_ValidateTokenForModel_AllTeamModelsSentinel(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	auth := NewAuthenticator(nil, cache, slog.Default())

	tokenInfo := &models.TokenInfo{
		Token:      "test-token",
		UserID:     "user1",
		TeamID:     "team1",
		Blocked:    false,
		Models:     []string{"all-team-models"},
		TeamModels: []string{"gpt-4"},
	}
	hashedToken := HashToken("sk-all-team-models-token")
	cache.Set(hashedToken, tokenInfo)

	// Allowed: in the team's allow-list.
	info, err := auth.ValidateTokenForModel(context.Background(), "sk-all-team-models-token", "gpt-4")
	assert.NoError(t, err)
	assert.Equal(t, "user1", info.UserID)

	// Not allowed: outside the team's allow-list.
	info, err = auth.ValidateTokenForModel(context.Background(), "sk-all-team-models-token", "claude-3")
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

func TestFetchMasterKey_ConfigKeyCachedWhenDBUnavailable(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)
	auth := NewAuthenticator(nil, cache, slog.Default())

	require.NoError(t, auth.FetchMasterKey(context.Background(), "sk-config-master"))

	info, ok := cache.Get(HashToken("sk-config-master"))
	require.True(t, ok)
	assert.Equal(t, "litellm-master-key", info.UserID)
	assert.Equal(t, "litellm-master-key", info.KeyName)
}

func TestFetchMasterKey_EmptyConfigAndNoDB(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)
	auth := NewAuthenticator(nil, cache, slog.Default())

	assert.ErrorIs(t, auth.FetchMasterKey(context.Background(), ""), models.ErrTokenNotFound)
	_, ok := cache.Get(HashToken(""))
	assert.False(t, ok)
}

// tokenQueryColumns names QueryValidateTokenWithHierarchy's SELECT list in
// order: a column's "as" alias, else its name after the table prefix.
func tokenQueryColumns(t *testing.T) []string {
	t.Helper()
	query := queries.QueryValidateTokenWithHierarchy
	from := strings.Index(query, `FROM "LiteLLM_VerificationToken"`)
	require.Positive(t, from, "FROM clause not found")

	var columns []string
	for _, line := range strings.Split(query[:from], "\n") {
		line = strings.TrimSuffix(strings.TrimSpace(line), ",")
		if line == "" || line == "SELECT" || strings.HasPrefix(line, "--") {
			continue
		}
		if _, alias, ok := strings.Cut(line, " as "); ok {
			line = alias
		} else if _, column, ok := strings.Cut(line, "."); ok {
			line = column
		}
		columns = append(columns, line)
	}
	return columns
}

// columnRow is a token query row in which every column holds a value naming
// it: text its column's name, a number its column's position, JSON
// {"column": name}. A Scan destination that reads a column other than the one
// meant for it then holds the wrong name or number.
type columnRow []string

func (r columnRow) Scan(dest ...any) error {
	if len(dest) != len(r) {
		return fmt.Errorf("%d columns into %d destinations", len(r), len(dest))
	}
	for i, column := range r {
		target := reflect.ValueOf(dest[i]).Elem()
		if target.Kind() == reflect.Pointer { // a nullable column
			target.Set(reflect.New(target.Type().Elem()))
			target = target.Elem()
		}
		switch v := target.Addr().Interface().(type) {
		case *string:
			*v = column
		case *float64:
			*v = float64(i)
		case *int64:
			*v = int64(i)
		case *bool:
			*v = true
		case *time.Time:
			*v = time.Unix(int64(i), 0)
		case *[]string:
			*v = []string{column}
		case *[]byte:
			*v = []byte(`{"column":"` + column + `"}`)
		default:
			return fmt.Errorf("column %s: unexpected destination %T", column, dest[i])
		}
	}
	return nil
}

// The token query's SELECT list and tokenFromRow's Scan are positional: a
// column moved in one and not the other reads a neighbour into a field (or
// fails every uncached token lookup on its type), so every field must hold
// the value of the column named for it.
func TestAuthenticator_TokenFromRow_ReadsEachColumnIntoItsField(t *testing.T) {
	columns := tokenQueryColumns(t)
	at := func(column string) int {
		i := slices.Index(columns, column)
		require.GreaterOrEqual(t, i, 0, "column %s not in the query", column)
		return i
	}
	number := func(column string) *float64 {
		v := float64(at(column))
		return &v
	}
	limit := func(column string) *int64 {
		v := int64(at(column))
		return &v
	}
	metadata := func(column string) map[string]interface{} {
		return map[string]interface{}{"column": column}
	}
	expires := time.Unix(int64(at("expires")), 0)
	teamBlocked := true

	info, err := NewAuthenticator(nil, nil, slog.Default()).tokenFromRow(columnRow(columns), "hash")
	require.NoError(t, err)
	assert.Equal(t, &models.TokenInfo{
		Token:                "token",
		KeyName:              "key_name",
		KeyAlias:             "key_alias",
		UserID:               "user_id",
		TeamID:               "team_id",
		OrganizationID:       "organization_id",
		DirectOrganizationID: "organization_id",
		TeamOrganizationID:   "team_organization_id",
		Spend:                float64(at("token_spend")),
		MaxBudget:            number("token_max_budget"),
		TPMLimit:             limit("token_tpm_limit"),
		RPMLimit:             limit("token_rpm_limit"),
		Expires:              &expires,
		Models:               []string{"token_models"},
		UserModels:           []string{"user_models"},
		TeamModels:           []string{"team_models"},
		Blocked:              true,

		UserAlias:     "user_alias",
		UserEmail:     "user_email",
		UserMaxBudget: number("user_max_budget"),
		UserSpend:     number("user_spend"),
		UserTPMLimit:  limit("user_tpm_limit"),
		UserRPMLimit:  limit("user_rpm_limit"),

		TeamAlias:     "team_alias",
		TeamMaxBudget: number("team_max_budget"),
		TeamSpend:     number("team_spend"),
		TeamBlocked:   &teamBlocked,
		TeamTPMLimit:  limit("team_tpm_limit"),
		TeamRPMLimit:  limit("team_rpm_limit"),

		OrganizationAlias: "org_alias",
		OrgSpend:          number("org_spend"),
		OrgMaxBudget:      number("org_max_budget"),
		OrgTPMLimit:       limit("org_tpm_limit"),
		OrgRPMLimit:       limit("org_rpm_limit"),

		TeamMemberSpend:     number("team_member_spend"),
		TeamMemberMaxBudget: number("team_member_max_budget"),
		TeamMemberTPMLimit:  limit("team_member_tpm_limit"),
		TeamMemberRPMLimit:  limit("team_member_rpm_limit"),
		TeamMemberModels:    []string{"team_member_models"},

		OrgMemberSpend:     number("org_member_spend"),
		OrgMemberMaxBudget: number("org_member_max_budget"),
		OrgMemberTPMLimit:  limit("org_member_tpm_limit"),
		OrgMemberRPMLimit:  limit("org_member_rpm_limit"),

		Metadata:             metadata("token_metadata"),
		TeamMetadata:         metadata("team_metadata"),
		OrganizationMetadata: metadata("org_metadata"),

		// {"column": "token_budget_limits"} is no budget_limits list: the key
		// gets one unenforceable window and Validate refuses it.
		BudgetWindows: []models.BudgetWindow{{}},
	}, info)
}

// A cached token carries the spend of the budget window it was loaded in.
// Once that window has ended the cache entry is not checked against it: the
// token is reloaded (here the reload fails, as there is no database).
func TestAuthenticator_ValidateToken_ReloadsATokenFromAnEndedBudgetWindow(t *testing.T) {
	cache, err := NewCache(100, time.Hour)
	require.NoError(t, err)
	auth := NewAuthenticator(nil, cache, slog.Default())

	now := time.Now().UTC()
	day, ok := models.ParseBudgetDuration("1d")
	require.True(t, ok)
	current, _ := day.Bounds(now, models.BudgetCalendar{})
	window := models.BudgetWindow{BudgetLimit: models.BudgetLimit{Duration: "1d", MaxBudget: 10}, Spend: 1}

	window.Start = current
	cache.Set(HashToken("sk-current-window"), &models.TokenInfo{Token: "current", BudgetWindows: []models.BudgetWindow{window}})
	info, err := auth.ValidateToken(context.Background(), "sk-current-window")
	require.NoError(t, err)
	assert.Equal(t, "current", info.Token)

	window.Start = current.AddDate(0, 0, -1)
	cache.Set(HashToken("sk-ended-window"), &models.TokenInfo{Token: "ended", BudgetWindows: []models.BudgetWindow{window}})
	_, err = auth.ValidateToken(context.Background(), "sk-ended-window")
	assert.ErrorIs(t, err, models.ErrConnectionFailed)
}

// budgetWindowConn serves QuerySelectKeyBudgetWindowSpend from rows (or
// rowsErr) and QuerySumKeyWindowSpendLogs from sum, recording each sum's
// window bounds.
type budgetWindowConn struct {
	rows    [][]any
	rowsErr error
	sum     float64
	sums    [][2]time.Time
}

func (c *budgetWindowConn) Query(_ context.Context, sql string, _ ...any) (pgx.Rows, error) {
	if sql != queries.QuerySelectKeyBudgetWindowSpend {
		return nil, fmt.Errorf("unexpected query %q", sql)
	}
	if c.rowsErr != nil {
		return nil, c.rowsErr
	}
	return &budgetWindowRows{rows: c.rows, index: -1}, nil
}

func (c *budgetWindowConn) QueryRow(_ context.Context, sql string, args ...any) pgx.Row {
	if sql != queries.QuerySumKeyWindowSpendLogs {
		return errRow{fmt.Errorf("unexpected query %q", sql)}
	}
	c.sums = append(c.sums, [2]time.Time{args[1].(time.Time), args[2].(time.Time)})
	return floatRow(c.sum)
}

type budgetWindowRows struct {
	pgx.Rows
	rows  [][]any
	index int
}

func (r *budgetWindowRows) Next() bool { r.index++; return r.index < len(r.rows) }
func (r *budgetWindowRows) Close()     {}
func (r *budgetWindowRows) Err() error { return nil }
func (r *budgetWindowRows) Scan(dest ...any) error {
	row := r.rows[r.index]
	*dest[0].(*string) = row[0].(string)
	*dest[1].(*time.Time) = row[1].(time.Time)
	*dest[2].(*float64) = row[2].(float64)
	return nil
}

type floatRow float64

func (r floatRow) Scan(dest ...any) error { *dest[0].(*float64) = float64(r); return nil }

type errRow struct{ err error }

func (r errRow) Scan(...any) error { return r.err }

func TestAuthenticator_LoadBudgetWindowSpend(t *testing.T) {
	moscow, err := time.LoadLocation("Europe/Moscow")
	require.NoError(t, err)
	now := time.Date(2026, 3, 10, 22, 30, 0, 0, time.UTC) // 11 March, 01:30 Moscow
	dayStart := time.Date(2026, 3, 10, 21, 0, 0, 0, time.UTC)
	dayEnd := time.Date(2026, 3, 11, 21, 0, 0, 0, time.UTC)
	monthStart := time.Date(2026, 2, 28, 21, 0, 0, 0, time.UTC)
	monthEnd := time.Date(2026, 3, 31, 21, 0, 0, 0, time.UTC)
	windows := func() []models.BudgetWindow {
		return []models.BudgetWindow{
			{BudgetLimit: models.BudgetLimit{Duration: "1d", MaxBudget: 10}},
			{BudgetLimit: models.BudgetLimit{Duration: "1mo", MaxBudget: 100}},
			{BudgetLimit: models.BudgetLimit{Duration: "2d", MaxBudget: 1}},
		}
	}
	auth := NewAuthenticator(nil, nil, slog.Default()).WithBudgetCalendar(models.BudgetCalendar{Location: moscow})

	t.Run("row on the current window", func(t *testing.T) {
		conn := &budgetWindowConn{
			rows: [][]any{{"1d", dayStart, 4.0}, {"1mo", monthStart, 40.0}},
		}
		info := &models.TokenInfo{Token: "tok", BudgetWindows: windows()}
		require.NoError(t, auth.loadBudgetWindowSpend(context.Background(), conn, info, now))
		assert.Equal(t, []models.BudgetWindow{
			{BudgetLimit: models.BudgetLimit{Duration: "1d", MaxBudget: 10}, Start: dayStart, Spend: 4},
			{BudgetLimit: models.BudgetLimit{Duration: "1mo", MaxBudget: 100}, Start: monthStart, Spend: 40},
			{BudgetLimit: models.BudgetLimit{Duration: "2d", MaxBudget: 1}},
		}, info.BudgetWindows)
		assert.Empty(t, conn.sums)
	})

	t.Run("row on an earlier window or no row reads the spend logs", func(t *testing.T) {
		conn := &budgetWindowConn{
			rows: [][]any{{"1d", dayStart.AddDate(0, 0, -1), 9.0}},
			sum:  2.5,
		}
		info := &models.TokenInfo{Token: "tok", BudgetWindows: windows()}
		require.NoError(t, auth.loadBudgetWindowSpend(context.Background(), conn, info, now))
		assert.Equal(t, 2.5, info.BudgetWindows[0].Spend)
		assert.Equal(t, 2.5, info.BudgetWindows[1].Spend)
		assert.Equal(t, [][2]time.Time{{dayStart, dayEnd}, {monthStart, monthEnd}}, conn.sums)
	})

	t.Run("schema without the window table reads the spend logs", func(t *testing.T) {
		conn := &budgetWindowConn{rowsErr: &pgconn.PgError{Code: "42P01"}, sum: 3}
		info := &models.TokenInfo{Token: "tok", BudgetWindows: windows()}
		require.NoError(t, auth.loadBudgetWindowSpend(context.Background(), conn, info, now))
		assert.Equal(t, 3.0, info.BudgetWindows[0].Spend)
		assert.Len(t, conn.sums, 2)
	})

	t.Run("other database errors fail the lookup", func(t *testing.T) {
		conn := &budgetWindowConn{rowsErr: &pgconn.PgError{Code: "57P01"}}
		info := &models.TokenInfo{Token: "tok", BudgetWindows: windows()}
		assert.Error(t, auth.loadBudgetWindowSpend(context.Background(), conn, info, now))
	})

	t.Run("a key without windows reads nothing", func(t *testing.T) {
		conn := &budgetWindowConn{rowsErr: fmt.Errorf("must not be called")}
		require.NoError(t, auth.loadBudgetWindowSpend(context.Background(), conn, &models.TokenInfo{Token: "tok"}, now))
	})
}

func TestAuthenticator_BudgetWindows(t *testing.T) {
	auth := NewAuthenticator(nil, nil, slog.Default())
	assert.Nil(t, auth.budgetWindows(nil, "hash"))
	assert.Nil(t, auth.budgetWindows([]byte(`[]`), "hash"))
	assert.Equal(t, []models.BudgetWindow{{BudgetLimit: models.BudgetLimit{Duration: "7d", MaxBudget: 5}}},
		auth.budgetWindows([]byte(`[{"budget_duration":"7d","max_budget":5}]`), "hash"))
	assert.Equal(t, []models.BudgetWindow{{}}, auth.budgetWindows([]byte(`[{"budget_duration":"7d"}]`), "hash"),
		"an unreadable budget_limits refuses the key")
}
