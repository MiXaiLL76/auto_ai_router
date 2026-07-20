package models

import (
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// ==================== DefaultConfig Tests ====================

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	assert.NotNil(t, cfg)
	assert.Equal(t, int32(10), cfg.MaxConns)
	assert.Equal(t, int32(2), cfg.MinConns)
	assert.Equal(t, 10*time.Second, cfg.HealthCheckInterval)
	assert.Equal(t, 5*time.Second, cfg.ConnectTimeout)
	assert.Equal(t, 5*time.Second, cfg.AuthCacheTTL)
	assert.Equal(t, 10000, cfg.AuthCacheSize)
	assert.Equal(t, 10000, cfg.LogQueueSize)
	assert.Equal(t, 100, cfg.LogBatchSize)
	assert.Equal(t, 5*time.Second, cfg.LogFlushInterval)
}

// ==================== Config.ApplyDefaults Tests ====================

func TestConfig_ApplyDefaults_AllZero(t *testing.T) {
	cfg := &Config{}
	cfg.ApplyDefaults()

	defaults := DefaultConfig()
	assert.Equal(t, defaults.MaxConns, cfg.MaxConns)
	assert.Equal(t, defaults.MinConns, cfg.MinConns)
	assert.Equal(t, defaults.HealthCheckInterval, cfg.HealthCheckInterval)
	assert.Equal(t, defaults.ConnectTimeout, cfg.ConnectTimeout)
	assert.Equal(t, defaults.AuthCacheTTL, cfg.AuthCacheTTL)
	assert.Equal(t, defaults.AuthCacheSize, cfg.AuthCacheSize)
	assert.Equal(t, defaults.LogQueueSize, cfg.LogQueueSize)
	assert.Equal(t, defaults.LogBatchSize, cfg.LogBatchSize)
	assert.Equal(t, defaults.LogFlushInterval, cfg.LogFlushInterval)
	assert.NotNil(t, cfg.Logger)
}

func TestConfig_ApplyDefaults_KeepNonZero(t *testing.T) {
	cfg := &Config{
		MaxConns:            20,
		AuthCacheSize:       5000,
		LogQueueSize:        50000,
		HealthCheckInterval: 30 * time.Second,
	}
	cfg.ApplyDefaults()

	assert.Equal(t, int32(20), cfg.MaxConns)
	assert.Equal(t, 5000, cfg.AuthCacheSize)
	assert.Equal(t, 50000, cfg.LogQueueSize)
	assert.Equal(t, 30*time.Second, cfg.HealthCheckInterval)
}

func TestConfig_ApplyDefaults_MinConnsGreaterThanMax(t *testing.T) {
	cfg := &Config{
		MaxConns: 5,
		MinConns: 10,
	}
	cfg.ApplyDefaults()

	// MinConns should be capped to MaxConns
	assert.Equal(t, cfg.MaxConns, cfg.MinConns)
	assert.Equal(t, int32(5), cfg.MinConns)
}

func TestConfig_ApplyDefaults_CustomLogger(t *testing.T) {
	customLogger := slog.New(slog.NewTextHandler(nil, nil))
	cfg := &Config{
		Logger: customLogger,
	}
	cfg.ApplyDefaults()

	assert.Equal(t, customLogger, cfg.Logger)
}

func TestConfig_ApplyDefaults_NilLogger(t *testing.T) {
	cfg := &Config{}
	cfg.ApplyDefaults()

	assert.NotNil(t, cfg.Logger)
}

// ==================== Config.Validate Tests ====================

func TestConfig_Validate_Success(t *testing.T) {
	cfg := &Config{
		DatabaseURL: "postgresql://localhost/test",
	}

	err := cfg.Validate()
	assert.NoError(t, err)
}

func TestConfig_Validate_EmptyDatabaseURL(t *testing.T) {
	cfg := &Config{
		DatabaseURL: "",
	}

	err := cfg.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database_url")
}

// ==================== TokenInfo Tests ====================

func TestTokenInfo_IsExpired_NilExpires(t *testing.T) {
	token := &TokenInfo{
		Expires: nil,
	}

	assert.False(t, token.IsExpired())
}

func TestTokenInfo_IsExpired_FutureExpires(t *testing.T) {
	future := time.Now().Add(time.Hour)
	token := &TokenInfo{
		Expires: &future,
	}

	assert.False(t, token.IsExpired())
}

func TestTokenInfo_IsExpired_PastExpires(t *testing.T) {
	past := time.Now().Add(-time.Hour)
	token := &TokenInfo{
		Expires: &past,
	}

	assert.True(t, token.IsExpired())
}

func TestTokenInfo_IsBudgetExceeded_NilMaxBudget(t *testing.T) {
	token := &TokenInfo{
		Spend:     1000,
		MaxBudget: nil,
	}

	assert.False(t, token.IsBudgetExceeded())
}

func TestTokenInfo_IsBudgetExceeded_SpendLessThanMax(t *testing.T) {
	maxBudget := 100.0
	token := &TokenInfo{
		Spend:     50,
		MaxBudget: &maxBudget,
	}

	assert.False(t, token.IsBudgetExceeded())
}

func TestTokenInfo_IsBudgetExceeded_SpendEqualToMax(t *testing.T) {
	maxBudget := 100.0
	token := &TokenInfo{
		Spend:     100,
		MaxBudget: &maxBudget,
	}

	assert.False(t, token.IsBudgetExceeded())
}

func TestTokenInfo_IsBudgetExceeded_SpendGreaterThanMax(t *testing.T) {
	maxBudget := 100.0
	token := &TokenInfo{
		Spend:     150,
		MaxBudget: &maxBudget,
	}

	assert.True(t, token.IsBudgetExceeded())
}

func TestTokenInfo_IsModelAllowed_EmptyModelsList(t *testing.T) {
	token := &TokenInfo{
		Models: nil,
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.True(t, token.IsModelAllowed("claude-3"))
}

func TestTokenInfo_IsModelAllowed_ModelInList(t *testing.T) {
	token := &TokenInfo{
		Models: []string{"gpt-4", "gpt-3.5-turbo"},
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
}

func TestTokenInfo_IsModelAllowed_ModelNotInList(t *testing.T) {
	token := &TokenInfo{
		Models: []string{"gpt-4", "gpt-3.5-turbo"},
	}

	assert.False(t, token.IsModelAllowed("claude-3"))
}

// ==================== IsModelAllowed sentinel tests ====================
// LiteLLM stores "all-proxy-models" / "all-team-models" inside
// VerificationToken.models instead of real model names (confirmed against a
// production LiteLLM_VerificationToken dump: 178 keys use all-proxy-models,
// 42 use all-team-models).

func TestTokenInfo_IsModelAllowed_AllProxyModels_AnyModelAllowed(t *testing.T) {
	token := &TokenInfo{
		Models: []string{"all-proxy-models"},
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.True(t, token.IsModelAllowed("claude-3"))
	assert.True(t, token.IsModelAllowed("anything"))
}

func TestTokenInfo_IsModelAllowed_AllProxyModels_MixedWithRealNames(t *testing.T) {
	// Sentinel can appear alongside real model names in the array - still unrestricted.
	token := &TokenInfo{
		Models: []string{"gpt-4", "all-proxy-models"},
	}

	assert.True(t, token.IsModelAllowed("claude-3"))
}

func TestTokenInfo_IsModelAllowed_AllTeamModels_NoTeam_Unrestricted(t *testing.T) {
	// No team to inherit from -> falls back to unrestricted, same as empty Models.
	token := &TokenInfo{
		Models: []string{"all-team-models"},
		TeamID: "",
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.True(t, token.IsModelAllowed("claude-3"))
}

func TestTokenInfo_IsModelAllowed_AllTeamModels_UsesTeamAllowList(t *testing.T) {
	token := &TokenInfo{
		Models:     []string{"all-team-models"},
		TeamID:     "team1",
		TeamModels: []string{"gpt-4"},
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.False(t, token.IsModelAllowed("claude-3"))
}

func TestTokenInfo_IsModelAllowed_AllTeamModels_TeamHasEmptyList_Unrestricted(t *testing.T) {
	// Team itself has no restriction (empty models = all allowed).
	token := &TokenInfo{
		Models:     []string{"all-team-models"},
		TeamID:     "team1",
		TeamModels: nil,
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.True(t, token.IsModelAllowed("claude-3"))
}

func TestTokenInfo_IsModelAllowed_AllTeamModels_TeamAlsoAllowsAllProxyModels(t *testing.T) {
	token := &TokenInfo{
		Models:     []string{"all-team-models"},
		TeamID:     "team1",
		TeamModels: []string{"all-proxy-models"},
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.True(t, token.IsModelAllowed("claude-3"))
}

func TestTokenInfo_IsModelAllowed_AllTeamModels_KeyTeamModelsIgnoredWithoutSentinel(t *testing.T) {
	// TeamModels must only apply when the "all-team-models" sentinel is present -
	// a key with its own explicit model list is not implicitly widened by the team's list.
	token := &TokenInfo{
		Models:     []string{"gpt-4"},
		TeamID:     "team1",
		TeamModels: []string{"claude-3"},
	}

	assert.True(t, token.IsModelAllowed("gpt-4"))
	assert.False(t, token.IsModelAllowed("claude-3"))
}

// ==================== IsAnyModelAllowed alias-group tests ====================

func TestTokenInfo_IsAnyModelAllowed_MatchesOnAnyCandidate(t *testing.T) {
	// Key is restricted to one alias of a model; client called it under a
	// different alias for the same underlying credential+model.
	token := &TokenInfo{
		Models: []string{"claude-haiku-4.5"},
	}

	assert.True(t, token.IsAnyModelAllowed([]string{"anthropic/claude-haiku-4.5", "claude-haiku-4.5", "claude-haiku-4-5-20251001"}))
}

func TestTokenInfo_IsAnyModelAllowed_NoneMatch(t *testing.T) {
	token := &TokenInfo{
		Models: []string{"gpt-4"},
	}

	assert.False(t, token.IsAnyModelAllowed([]string{"claude-3", "claude-3-opus"}))
}

func TestTokenInfo_IsAnyModelAllowed_EmptyModelsList(t *testing.T) {
	token := &TokenInfo{Models: nil}

	assert.True(t, token.IsAnyModelAllowed([]string{"anything"}))
}

// ==================== Budget Check Helper Tests ====================

func TestTokenInfo_checkUserBudget_PersonalKey(t *testing.T) {
	userBudget := 100.0
	userSpend := 150.0
	token := &TokenInfo{
		UserID:        "user1",
		TeamID:        "",
		UserMaxBudget: &userBudget,
		UserSpend:     &userSpend,
	}

	assert.True(t, token.checkUserBudget())
}

func TestTokenInfo_checkUserBudget_NotPersonalKey(t *testing.T) {
	userBudget := 100.0
	userSpend := 150.0
	token := &TokenInfo{
		UserID:        "user1",
		TeamID:        "team1",
		UserMaxBudget: &userBudget,
		UserSpend:     &userSpend,
	}

	assert.False(t, token.checkUserBudget())
}

func TestTokenInfo_checkUserBudget_NilBudget(t *testing.T) {
	token := &TokenInfo{
		UserID:        "user1",
		TeamID:        "",
		UserMaxBudget: nil,
		UserSpend:     nil,
	}

	assert.False(t, token.checkUserBudget())
}

func TestTokenInfo_checkTeamBudget_ExceededEmbedded(t *testing.T) {
	teamBudget := 100.0
	teamSpend := 150.0
	token := &TokenInfo{
		TeamMaxBudget: &teamBudget,
		TeamSpend:     &teamSpend,
	}

	assert.True(t, token.checkTeamBudget())
}

func TestTokenInfo_checkTeamBudget_NotExceeded(t *testing.T) {
	teamBudget := 100.0
	teamSpend := 50.0
	token := &TokenInfo{
		TeamMaxBudget: &teamBudget,
		TeamSpend:     &teamSpend,
	}

	assert.False(t, token.checkTeamBudget())
}

func TestTokenInfo_checkTeamMemberBudget_ExceededExternal(t *testing.T) {
	memberBudget := 100.0
	memberSpend := 100.0 // >= triggers
	token := &TokenInfo{
		UserID:              "user1",
		TeamID:              "team1",
		TeamMemberMaxBudget: &memberBudget,
		TeamMemberSpend:     &memberSpend,
	}

	assert.True(t, token.checkTeamMemberBudget())
}

func TestTokenInfo_checkTeamMemberBudget_JustBelowLimit(t *testing.T) {
	memberBudget := 100.0
	memberSpend := 99.99
	token := &TokenInfo{
		UserID:              "user1",
		TeamID:              "team1",
		TeamMemberMaxBudget: &memberBudget,
		TeamMemberSpend:     &memberSpend,
	}

	assert.False(t, token.checkTeamMemberBudget())
}

func TestTokenInfo_checkOrganizationBudget_Exceeded(t *testing.T) {
	orgBudget := 100.0
	orgSpend := 100.0 // >= triggers
	token := &TokenInfo{
		OrganizationID: "org1",
		OrgMaxBudget:   &orgBudget,
		OrgSpend:       &orgSpend,
	}

	assert.True(t, token.checkOrganizationBudget())
}

func TestTokenInfo_checkOrganizationBudget_ZeroMaxBudget(t *testing.T) {
	orgBudget := 0.0
	orgSpend := 50.0
	token := &TokenInfo{
		OrganizationID: "org1",
		OrgMaxBudget:   &orgBudget,
		OrgSpend:       &orgSpend,
	}

	assert.False(t, token.checkOrganizationBudget())
}

func TestTokenInfo_checkOrganizationMemberBudget_Exceeded(t *testing.T) {
	memberBudget := 100.0
	memberSpend := 100.0 // >= triggers
	token := &TokenInfo{
		UserID:             "user1",
		OrganizationID:     "org1",
		OrgMemberMaxBudget: &memberBudget,
		OrgMemberSpend:     &memberSpend,
	}

	assert.True(t, token.checkOrganizationMemberBudget())
}

// ==================== Validate Tests ====================

func TestTokenInfo_Validate_ValidToken(t *testing.T) {
	token := &TokenInfo{
		Token:   "test",
		Blocked: false,
	}

	err := token.Validate("")
	assert.NoError(t, err)
}

func TestTokenInfo_Validate_BlockedToken(t *testing.T) {
	token := &TokenInfo{
		Blocked: true,
	}

	err := token.Validate("")
	assert.ErrorIs(t, err, ErrTokenBlocked)
}

func TestTokenInfo_Validate_ExpiredToken(t *testing.T) {
	past := time.Now().Add(-time.Hour)
	token := &TokenInfo{
		Expires: &past,
	}

	err := token.Validate("")
	assert.ErrorIs(t, err, ErrTokenExpired)
}

func TestTokenInfo_Validate_TokenBudgetExceeded(t *testing.T) {
	maxBudget := 100.0
	token := &TokenInfo{
		Spend:     150,
		MaxBudget: &maxBudget,
	}

	err := token.Validate("")
	assert.ErrorIs(t, err, ErrBudgetExceeded)
}

func TestTokenInfo_Validate_TeamBudgetExceeded(t *testing.T) {
	teamBudget := 100.0
	teamSpend := 150.0
	token := &TokenInfo{
		TeamMaxBudget: &teamBudget,
		TeamSpend:     &teamSpend,
	}

	err := token.Validate("")
	assert.ErrorIs(t, err, ErrBudgetExceeded)
}

func TestTokenInfo_Validate_ModelNotAllowed(t *testing.T) {
	token := &TokenInfo{
		Models: []string{"gpt-4"},
	}

	err := token.Validate("claude-3")
	assert.ErrorIs(t, err, ErrModelNotAllowed)
}

func TestTokenInfo_Validate_ModelAllowedWithEmptyCheck(t *testing.T) {
	token := &TokenInfo{
		Models: []string{"gpt-4"},
	}

	err := token.Validate("")
	assert.NoError(t, err)
}

func TestTokenInfo_Validate_ValidationOrder(t *testing.T) {
	// When multiple issues exist, blocked check should come first
	blockedToken := true
	pastExpires := time.Now().Add(-time.Hour)
	maxBudget := 100.0

	token := &TokenInfo{
		Blocked:   blockedToken,
		Expires:   &pastExpires,
		MaxBudget: &maxBudget,
		Spend:     150,
	}

	err := token.Validate("")
	assert.ErrorIs(t, err, ErrTokenBlocked)
}

// ==================== Integration Tests ====================

func TestConfig_FullWorkflow(t *testing.T) {
	// Create config with custom values
	cfg := &Config{
		DatabaseURL:      "postgresql://localhost/test",
		MaxConns:         20,
		LogQueueSize:     50000,
		LogFlushInterval: 10 * time.Second,
	}

	// Apply defaults for zero values
	cfg.ApplyDefaults()

	// Verify defaults are applied
	assert.Equal(t, int32(2), cfg.MinConns)
	assert.Equal(t, 100, cfg.LogBatchSize) // Default

	// Verify custom values are kept
	assert.Equal(t, int32(20), cfg.MaxConns)
	assert.Equal(t, 50000, cfg.LogQueueSize)

	// Validate configuration
	err := cfg.Validate()
	assert.NoError(t, err)
}

func TestTokenInfo_ComplexValidation(t *testing.T) {
	// Create a complex token with multiple budget levels
	now := time.Now()
	future := now.Add(time.Hour)

	teamBudget := 1000.0
	teamSpend := 500.0
	memberBudget := 200.0
	memberSpend := 150.0

	token := &TokenInfo{
		Token:               "test-token",
		Blocked:             false,
		Expires:             &future,
		TeamMaxBudget:       &teamBudget,
		TeamSpend:           &teamSpend,
		TeamMemberMaxBudget: &memberBudget,
		TeamMemberSpend:     &memberSpend,
		UserID:              "user1",
		TeamID:              "team1",
		Models:              []string{"gpt-4", "gpt-3.5-turbo"},
	}

	// Should pass validation
	err := token.Validate("gpt-4")
	assert.NoError(t, err)

	// Should fail for disallowed model
	err = token.Validate("claude-3")
	assert.ErrorIs(t, err, ErrModelNotAllowed)

	// Should fail if team member spend reaches budget
	newMemberSpend := 200.0
	token.TeamMemberSpend = &newMemberSpend
	err = token.Validate("gpt-4")
	assert.ErrorIs(t, err, ErrBudgetExceeded)
}

func TestTokenInfo_Validate_AllTeamModelsSentinel_ResolvedThroughFullValidate(t *testing.T) {
	token := &TokenInfo{
		Token:      "test-token",
		UserID:     "user1",
		TeamID:     "team1",
		Models:     []string{"all-team-models"},
		TeamModels: []string{"gpt-4"},
	}

	assert.NoError(t, token.Validate("gpt-4"))
	assert.ErrorIs(t, token.Validate("claude-3"), ErrModelNotAllowed)
}
