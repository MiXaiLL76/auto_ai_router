package httputil

import (
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
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

// ModelPriorityTier is one learned primary-priority tier of a proxy/AIR credential for a
// model: the aggregate (summed) capacity + weight of all the upstream leaf credentials
// that serve the model at that priority. A proxy credential expands into one balancer
// candidate per tier (see internal/balancer candidate expansion), each in its own
// primary priority group with its own local rate-limit bucket, so this router can
// cascade off its own tier-1 saturation without waiting for the next /health poll.
type ModelPriorityTier struct {
	Priority   int  `json:"priority"`
	Weight     int  `json:"weight"`
	LimitRPM   int  `json:"limit_rpm"`
	LimitTPM   int  `json:"limit_tpm"`
	CurrentRPM int  `json:"current_rpm"`
	CurrentTPM int  `json:"current_tpm"`
	Banned     bool `json:"banned"` // every upstream leaf credential in this tier is currently banned
}

// ModelHealthStats represents health stats for a single model
type ModelHealthStats struct {
	Credential string `json:"credential"`
	// PriorityTiers is the per-tier breakdown for a proxy/AIR credential serving this
	// model from an upstream that itself spans several priority groups. Empty for a
	// direct provider credential (or an upstream with a single tier) — callers then fall
	// back to the scalar Priority/Weight/Limit* fields below as one implicit tier. A
	// downstream router reads this to keep the tier structure across chain hops.
	PriorityTiers []ModelPriorityTier `json:"priority_tiers,omitempty"`
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
//
// Compatibility shim: an older upstream router emits is_fallback: true together with
// priority: 0 (it never normalized the fallback flag into a priority number). Treat that
// as the last-resort group so a chained downstream tiers those models correctly instead
// of merging them into its primary pool.
func EffectiveHealthPriority(modelStats ModelHealthStats, credStats CredentialHealthStats) int {
	priority := 0
	if modelStats.Priority > 0 {
		priority = modelStats.Priority
	} else if credStats.Priority > 0 {
		priority = credStats.Priority
	}
	if credStats.IsFallback && priority < config.FallbackPriorityGroup {
		priority = config.FallbackPriorityGroup
	}
	return priority
}

// ModelHealthEntryLive reports whether the upstream leaf credential behind this
// /health model entry can take traffic for the model right now: not banned, and
// neither its RPM nor TPM budget currently exhausted. Mirrors the webui dashboard's
// isRowLive().
//
// A proxy credential learns a single scalar priority per model by MIN-folding its
// upstream's per-tier entries (internal/proxy/remote.go updateModelLimits). Folding in
// only *live* entries keeps that scalar tracking the tier the upstream is actually
// serving from: when the cheap tier is saturated the upstream has already cascaded to a
// pricier one, and MIN must rise with it so the local balancer can prefer a mid-priced
// alternative instead of over-sending to a proxy that only looks cheap.
func ModelHealthEntryLive(s ModelHealthStats) bool {
	if s.IsBanned {
		return false
	}
	if s.LimitRPM > 0 && s.CurrentRPM >= s.LimitRPM {
		return false
	}
	if s.LimitTPM > 0 && s.CurrentTPM >= s.LimitTPM {
		return false
	}
	return true
}

// ModelPriorityTierLive mirrors ModelHealthEntryLive for a single learned tier: not
// fully banned, and neither its aggregate RPM nor TPM budget exhausted.
func ModelPriorityTierLive(t ModelPriorityTier) bool {
	if t.Banned {
		return false
	}
	if t.LimitRPM > 0 && t.CurrentRPM >= t.LimitRPM {
		return false
	}
	if t.LimitTPM > 0 && t.CurrentTPM >= t.LimitTPM {
		return false
	}
	return true
}
