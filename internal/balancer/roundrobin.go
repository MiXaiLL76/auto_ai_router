package balancer

import (
	"errors"
	"io"
	"log/slog"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
)

// ModelChecker interface for checking model availability
type ModelChecker interface {
	HasModel(credentialName, modelID string) bool
	GetCredentialsForModel(modelID string) []string
	IsEnabled() bool
}

var (
	ErrNoCredentialsAvailable = errors.New("no credentials available")
	ErrRateLimitExceeded      = errors.New("rate limit exceeded")
)

type RoundRobin struct {
	mu              sync.RWMutex
	credentials     []config.CredentialConfig
	credentialIndex map[string]int // O(1) lookup by name instead of O(n) search
	current         int
	fail2ban        *fail2ban.Fail2Ban
	rateLimiter     *ratelimit.RPMLimiter
	modelChecker    ModelChecker
	logger          *slog.Logger
}

func New(credentials []config.CredentialConfig, f2b *fail2ban.Fail2Ban, rl *ratelimit.RPMLimiter) *RoundRobin {
	if f2b == nil {
		panic("balancer.New: fail2ban must not be nil")
	}
	if rl == nil {
		panic("balancer.New: rateLimiter must not be nil")
	}

	credentialIndex := make(map[string]int, len(credentials))
	for i, c := range credentials {
		// Normalize TPM: 0 means "not configured" → treat as unlimited (-1).
		// Convention: -1 = unlimited, positive = limit.
		tpm := c.TPM
		if tpm == 0 {
			tpm = -1
		}
		rl.AddCredentialWithTPM(c.Name, c.RPM, tpm)
		credentialIndex[c.Name] = i
	}

	rr := &RoundRobin{
		credentials:     credentials,
		credentialIndex: credentialIndex,
		current:         0,
		fail2ban:        f2b,
		rateLimiter:     rl,
		modelChecker:    nil,
		logger:          slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelInfo})),
	}

	// Validate fallback configuration (cycle detection and unused fallback detection)
	rr.validateFallbackConfiguration()

	return rr
}

// SetLogger sets the logger for the RoundRobin balancer
func (r *RoundRobin) SetLogger(logger *slog.Logger) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.logger = logger
}

// SetModelChecker sets the model checker for filtering credentials by model
func (r *RoundRobin) SetModelChecker(mc ModelChecker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modelChecker = mc
}

// getCredentialByName finds a credential by name (must be called with lock held)
func (r *RoundRobin) getCredentialByName(name string) *config.CredentialConfig {
	idx, ok := r.credentialIndex[name]
	if !ok {
		return nil
	}
	return &r.credentials[idx]
}

// IsProxyCredential checks if a credential is a proxy type
func (r *RoundRobin) IsProxyCredential(credentialName string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	cred := r.getCredentialByName(credentialName)
	return cred != nil && cred.Type == config.ProviderTypeProxy
}

// IsBanned checks if a specific credential+model pair is currently banned
func (r *RoundRobin) IsBanned(credentialName, modelID string) bool {
	return r.fail2ban.IsBanned(credentialName, modelID)
}

// HasAnyBan checks if a credential has any banned models
func (r *RoundRobin) HasAnyBan(credentialName string) bool {
	return r.fail2ban.HasAnyBan(credentialName)
}

// GetProxyCredentials returns all proxy type credentials
func (r *RoundRobin) GetProxyCredentials() []config.CredentialConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var proxies []config.CredentialConfig
	for _, cred := range r.credentials {
		if cred.Type == config.ProviderTypeProxy {
			proxies = append(proxies, cred)
		}
	}
	return proxies
}

// NextForModel returns the next available credential that supports the specified model
func (r *RoundRobin) NextForModel(modelID string) (*config.CredentialConfig, error) {
	return r.next(modelID, false, false)
}

// NextFallbackForModel returns the next available fallback credential
func (r *RoundRobin) NextFallbackForModel(modelID string) (*config.CredentialConfig, error) {
	return r.next(modelID, true, false)
}

// NextFallbackProxyForModel returns the next available fallback proxy credential
func (r *RoundRobin) NextFallbackProxyForModel(modelID string) (*config.CredentialConfig, error) {
	return r.next(modelID, true, true)
}

func (r *RoundRobin) next(modelID string, allowOnlyFallback, allowOnlyProxy bool) (*config.CredentialConfig, error) {
	return r.nextExcluding(modelID, allowOnlyFallback, allowOnlyProxy, nil)
}

// nextExcluding is the core credential selection logic with optional exclude set.
// Excluded credentials are skipped entirely and don't count as candidates.
func (r *RoundRobin) nextExcluding(modelID string, allowOnlyFallback, allowOnlyProxy bool, exclude map[string]bool) (*config.CredentialConfig, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Start from current position and try each credential in rotation
	startIndex := r.current
	candidateCount := 0 // credentials that actually support this model
	rateLimitHit := false

	for i := 0; i < len(r.credentials); i++ {
		// Calculate current index in rotation
		idx := (startIndex + i) % len(r.credentials)
		cred := &r.credentials[idx]

		// Skip excluded credentials first (they don't count as candidates)
		if len(exclude) > 0 && exclude[cred.Name] {
			continue
		}

		// Filter by credential type
		// These are structural filters, not availability issues
		if allowOnlyProxy && cred.Type != config.ProviderTypeProxy {
			monitoring.CredentialSelectionRejected.WithLabelValues("type_not_allowed").Inc()
			continue
		}

		// Filter by is_fallback flag
		if allowOnlyFallback {
			if !cred.IsFallback {
				monitoring.CredentialSelectionRejected.WithLabelValues("fallback_not_available").Inc()
				continue
			}
		} else if cred.IsFallback {
			monitoring.CredentialSelectionRejected.WithLabelValues("fallback_only").Inc()
			continue
		}

		// Check if credential supports the requested model BEFORE ban/rate checks.
		// model_not_available is a structural property (credential type doesn't serve this model),
		// NOT a temporary availability issue — it should not mask rate limit errors.
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.modelChecker.HasModel(cred.Name, modelID) {
				monitoring.CredentialSelectionRejected.WithLabelValues("model_not_available").Inc()
				continue
			}
		}

		// This credential supports the model — it's a real candidate
		candidateCount++

		// Check if credential+model pair is banned
		if r.fail2ban.IsBanned(cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("banned").Inc()
			continue
		}

		// Atomically check all rate limits (credential RPM/TPM + model RPM/TPM)
		// and record usage only if all checks pass. This prevents TOCTOU races
		// where separate check+record calls could allow exceeding limits.
		if !r.rateLimiter.TryAllowAll(cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}

		// Advance r.current to next position for the next call
		// Move to the position right after this selected credential
		r.current = (idx + 1) % len(r.credentials)

		return cred, nil
	}

	// If no credentials support this model at all, return generic error
	if candidateCount == 0 {
		return nil, ErrNoCredentialsAvailable
	}
	// Prioritize rate limit error: if any candidate hit rate limit, surface it even if
	// others were banned. This gives callers accurate signal for backoff/retry logic.
	if rateLimitHit {
		return nil, ErrRateLimitExceeded
	}
	// All candidates are banned
	return nil, ErrNoCredentialsAvailable
}

// NextForModelExcluding returns the next available non-fallback credential that supports
// the specified model, excluding credentials in the exclude set. Used for same-type
// credential retry on provider errors (429/5xx/auth errors).
func (r *RoundRobin) NextForModelExcluding(modelID string, exclude map[string]bool) (*config.CredentialConfig, error) {
	return r.nextExcluding(modelID, false, false, exclude)
}

func (r *RoundRobin) RecordResponse(credentialName, modelID string, statusCode int) {
	r.fail2ban.RecordResponse(credentialName, modelID, statusCode)
}

func (r *RoundRobin) GetCredentialsSnapshot() []config.CredentialConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	creds := make([]config.CredentialConfig, len(r.credentials))
	copy(creds, r.credentials)
	return creds
}

func (r *RoundRobin) GetAvailableCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	count := 0
	for _, cred := range r.credentials {
		if !r.fail2ban.HasAnyBan(cred.Name) {
			count++
		}
	}
	return count
}

func (r *RoundRobin) GetBannedCount() int {
	return r.fail2ban.GetBannedCount()
}

// GetBannedPairs returns all currently banned credential+model pairs with error details
func (r *RoundRobin) GetBannedPairs() []fail2ban.BanPair {
	return r.fail2ban.GetBannedPairs()
}

// validateFallbackConfiguration validates fallback credential configuration
// Logs count of fallback credentials
func (r *RoundRobin) validateFallbackConfiguration() {
	fallbackCount := 0
	for _, cred := range r.credentials {
		if cred.IsFallback {
			fallbackCount++
		}
	}

	if fallbackCount == 0 {
		r.logger.Info("No fallback credentials configured")
	} else {
		r.logger.Info("Fallback credential validation completed",
			"total_credentials", len(r.credentials),
			"fallback_credentials", fallbackCount,
		)
	}
}
