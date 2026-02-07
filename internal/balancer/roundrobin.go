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
	credentialIndex := make(map[string]int, len(credentials))
	for i, c := range credentials {
		// Normalize TPM: treat 0 as unlimited (-1) for consistency
		// Convention: -1 = unlimited, 0 = invalid, positive = limit
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
	r.mu.Lock()
	defer r.mu.Unlock()

	cred := r.getCredentialByName(credentialName)
	return cred != nil && cred.Type == config.ProviderTypeProxy
}

// IsBanned checks if a credential is currently banned
func (r *RoundRobin) IsBanned(credentialName string) bool {
	return r.fail2ban.IsBanned(credentialName)
}

// GetProxyCredentials returns all proxy type credentials
func (r *RoundRobin) GetProxyCredentials() []config.CredentialConfig {
	r.mu.Lock()
	defer r.mu.Unlock()

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
	return r.next(modelID, false)
}

// NextFallbackForModel returns the next available fallback proxy credential
func (r *RoundRobin) NextFallbackForModel(modelID string) (*config.CredentialConfig, error) {
	return r.next(modelID, true)
}

func (r *RoundRobin) next(modelID string, allowOnlyFallback bool) (*config.CredentialConfig, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	attempts := 0
	rateLimitHit := false
	otherReasonsHit := false

	for {
		if attempts >= len(r.credentials) {
			// If all credentials were blocked due to rate limit, return rate limit error
			if rateLimitHit && !otherReasonsHit {
				return nil, ErrRateLimitExceeded
			}
			// Otherwise return generic error
			return nil, ErrNoCredentialsAvailable
		}

		cred := &r.credentials[r.current]
		r.current = (r.current + 1) % len(r.credentials)
		attempts++

		// Filter by is_fallback flag
		if allowOnlyFallback && (cred.Type != config.ProviderTypeProxy || !cred.IsFallback) {
			monitoring.CredentialSelectionRejected.WithLabelValues("fallback_not_available").Inc()
			otherReasonsHit = true
			continue
		}

		// Check if credential is banned
		if r.fail2ban.IsBanned(cred.Name) {
			monitoring.CredentialSelectionRejected.WithLabelValues("banned").Inc()
			otherReasonsHit = true
			continue
		}

		// Check if credential supports the requested model BEFORE rate limiting
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.modelChecker.HasModel(cred.Name, modelID) {
				monitoring.CredentialSelectionRejected.WithLabelValues("model_not_available").Inc()
				otherReasonsHit = true
				continue
			}
		}

		// Check credential RPM limit (without recording)
		if !r.rateLimiter.CanAllow(cred.Name) {
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}

		// Check credential TPM limit
		if !r.rateLimiter.AllowTokens(cred.Name) {
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}

		// Check model RPM limit if model is specified (without recording)
		if modelID != "" {
			if !r.rateLimiter.CanAllowModel(cred.Name, modelID) {
				monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
				rateLimitHit = true
				continue
			}
		}

		// Check model TPM limit if model is specified
		if modelID != "" {
			if !r.rateLimiter.AllowModelTokens(cred.Name, modelID) {
				monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
				rateLimitHit = true
				continue
			}
		}

		// All checks passed - now record the requests
		r.rateLimiter.Allow(cred.Name) // Record credential RPM
		if modelID != "" {
			r.rateLimiter.AllowModel(cred.Name, modelID) // Record model RPM
		}

		return cred, nil
	}
}

func (r *RoundRobin) RecordResponse(credentialName string, statusCode int) {
	r.fail2ban.RecordResponse(credentialName, statusCode)
}

func (r *RoundRobin) GetCredentials() []config.CredentialConfig {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.credentials
}

// GetCredentialsSnapshot returns a copy of the credentials slice to avoid data races
// when the credentials list is being read from multiple goroutines
func (r *RoundRobin) GetCredentialsSnapshot() []config.CredentialConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	creds := make([]config.CredentialConfig, len(r.credentials))
	copy(creds, r.credentials)
	return creds
}

func (r *RoundRobin) GetAvailableCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	count := 0
	for _, cred := range r.credentials {
		if !r.fail2ban.IsBanned(cred.Name) {
			count++
		}
	}
	return count
}

func (r *RoundRobin) GetBannedCount() int {
	return r.fail2ban.GetBannedCount()
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
