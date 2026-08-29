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
	Type       string `json:"type"`
	BaseURL    string `json:"base_url,omitempty"`
	IsFallback bool   `json:"is_fallback"`
	// IsProxyLike is the backend's own cred.IsProxyLike() verdict (proxy or air today),
	// exposed so dashboard code does not have to hardcode the list of proxy-like types
	// and drift when a new one is added.
	IsProxyLike       bool              `json:"is_proxy_like"`
	IsBanned          bool              `json:"is_banned"`
	Weight            int               `json:"weight"`
	FallbackPriority  int               `json:"fallback_priority,omitempty"`
	Priority          int               `json:"priority,omitempty"`
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
	Credential string `json:"credential"`
	// RealCredential is the actual leaf credential serving this model behind a
	// proxy/AIR credential (e.g. Credential="router2", RealCredential="mock2"),
	// learned from that upstream's own /health response. Empty when Credential
	// already is the real one (not a proxy/AIR relay) — display code should
	// fall back to Credential in that case. Display-only, never used for routing.
	RealCredential  string            `json:"real_credential,omitempty"`
	Model           string            `json:"model"`
	IsBanned        bool              `json:"is_banned"`
	Weight          int               `json:"weight"`
	Priority        int               `json:"priority,omitempty"`
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

// EffectiveHealthPriority resolves the priority-group fallback chain for a model on this
// credential's connection, mirroring EffectiveHealthWeight: model-level priority, then
// the credential-level priority, then 0 (the default group). Only the explicit priority
// field is consulted — never fallback_priority — so a value learned here can safely drive
// a downstream proxy credential's primary-pool grouping (see balancer.primaryPriority);
// folding fallback_priority in would turn a retry-only knob into hard primary tiers.
func EffectiveHealthPriority(modelStats ModelHealthStats, credStats CredentialHealthStats) int {
	if modelStats.Priority > 0 {
		return modelStats.Priority
	}
	if credStats.Priority > 0 {
		return credStats.Priority
	}
	return 0
}
