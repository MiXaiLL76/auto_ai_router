package auth

import (
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

func TestCache_SetGet(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	info := &models.TokenInfo{Token: "test123", UserID: "user1"}
	cache.Set("hash123", info)

	got, ok := cache.Get("hash123")
	assert.True(t, ok)
	assert.Equal(t, "user1", got.UserID)
}

func TestCache_NotFound(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	_, ok := cache.Get("nonexistent")
	assert.False(t, ok)
}

func TestCache_TTLExpired(t *testing.T) {
	cache, err := NewCache(100, 10*time.Millisecond)
	require.NoError(t, err)

	cache.Set("hash123", &models.TokenInfo{UserID: "user1"})
	time.Sleep(20 * time.Millisecond)

	_, ok := cache.Get("hash123")
	assert.False(t, ok)
}

func TestCache_Invalidate(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	cache.Set("hash123", &models.TokenInfo{UserID: "user1"})
	cache.Invalidate("hash123")

	_, ok := cache.Get("hash123")
	assert.False(t, ok)
}

func TestCache_InvalidateAll(t *testing.T) {
	cache, err := NewCache(100, time.Minute)
	require.NoError(t, err)

	cache.Set("hash1", &models.TokenInfo{UserID: "user1"})
	cache.Set("hash2", &models.TokenInfo{UserID: "user2"})
	cache.InvalidateAll()

	assert.Equal(t, 0, cache.Len())
}

func TestCache_Stats(t *testing.T) {
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
