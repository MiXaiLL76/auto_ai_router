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

// EffectiveHealthPriority resolves the priority group for a model on this
// credential's connection.
//
// Design choice (see todo_round_robin.md T3/6.2): this function itself does
// NOT layer a per-model override on top of the credential-level value —
// credStats.Priority (this connection's own owning-credential priority, as
// reported by the remote /health response) is authoritative here. Note this
// is a different concern from ModelHealthStats.Priority's own population on
// the *serving* side: internal/proxy/health.go addModelHealthStats resolves
// that field per (credential, model) using the dynamic per-model priority
// learned from an upstream proxy/AIR credential's own /health poll when
// known, falling back to the owning credential's static
// CredentialConfig.EffectivePriority() otherwise — see that function's
// comment for the resolution order. modelStats is still accepted here so the
// signature stays symmetric with EffectiveHealthWeight and so a future
// per-model override can be added to this aggregation step too without an
// API-breaking change.
//
// Contract: 0 is a valid, meaningful value (the default priority group), not
// a sentinel for "not configured" — CredentialConfig.EffectivePriority()
// already collapses "no priority/fallback_priority set" to 0 before it ever
// reaches CredStats.Priority, so there is no separate "unset" case to detect
// here (unlike Weight, where 0 genuinely means "no override").
func EffectiveHealthPriority(modelStats ModelHealthStats, credStats CredentialHealthStats) int {
	return credStats.Priority
}
