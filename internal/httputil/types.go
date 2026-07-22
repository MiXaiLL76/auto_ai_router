package httputil

import (
	"time"

	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// ProxyHealthResponse represents the JSON response from /health endpoint
type ProxyHealthResponse struct {
	Status               string                           `json:"status"`
	CredentialsAvailable int                              `json:"credentials_available"`
	CredentialsBanned    int                              `json:"credentials_banned"`
	TotalCredentials     int                              `json:"total_credentials"`
	Credentials          map[string]CredentialHealthStats `json:"credentials"`
	Models               map[string]ModelHealthStats      `json:"models"`
}

// CredentialHealthStats represents health stats for a single credential
type CredentialHealthStats struct {
	Type              string            `json:"type"`
	BaseURL           string            `json:"base_url,omitempty"`
	IsFallback        bool              `json:"is_fallback"`
	IsBanned          bool              `json:"is_banned"`
	Weight            int               `json:"weight"`
	FallbackPriority  int               `json:"fallback_priority,omitempty"`
	Scopes            []string          `json:"scopes,omitempty"`
	DeniedScopes      []string          `json:"denied_scopes,omitempty"`
	ScopeExpression   *scope.Expression `json:"scope_expression,omitempty"`
	CurrentRPM        int               `json:"current_rpm"`
	CurrentTPM        int               `json:"current_tpm"`
	LimitRPM          int               `json:"limit_rpm"`
	LimitTPM          int               `json:"limit_tpm"`
	BannedErrorCounts map[int]int       `json:"banned_error_counts,omitempty"` // aggregated error counts from banned models
}

// ModelHealthStats represents health stats for a single model
type ModelHealthStats struct {
	Credential      string            `json:"credential"`
	Model           string            `json:"model"`
	IsBanned        bool              `json:"is_banned"`
	Weight          int               `json:"weight"`
	CurrentRPM      int               `json:"current_rpm"`
	CurrentTPM      int               `json:"current_tpm"`
	LimitRPM        int               `json:"limit_rpm"`
	LimitTPM        int               `json:"limit_tpm"`
	Scopes          []string          `json:"scopes,omitempty"`
	DeniedScopes    []string          `json:"denied_scopes,omitempty"`
	ScopeExpression *scope.Expression `json:"scope_expression,omitempty"`
	ErrorCodeCounts map[int]int       `json:"error_code_counts,omitempty"` // error code -> count when banned
	ProviderError   string            `json:"provider_error,omitempty"`
	BanUntil        *time.Time        `json:"ban_until,omitempty"`
}

// EffectiveHealthWeight resolves the health weight fallback chain:
// model-level override, then credential default, then 1.
func EffectiveHealthWeight(modelStats ModelHealthStats, credStats CredentialHealthStats) int {
	if modelStats.Weight > 0 {
		return modelStats.Weight
	}
	if credStats.Weight > 0 {
		return credStats.Weight
	}
	return 1
}
