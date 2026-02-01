package balancer

import (
	"errors"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
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
	mu           sync.Mutex
	credentials  []config.CredentialConfig
	current      int
	fail2ban     *fail2ban.Fail2Ban
	rateLimiter  *ratelimit.RPMLimiter
	modelChecker ModelChecker
}

func New(credentials []config.CredentialConfig, f2b *fail2ban.Fail2Ban, rl *ratelimit.RPMLimiter) *RoundRobin {
	for _, c := range credentials {
		// Use -1 as default for unlimited TPM if not specified
		tpm := c.TPM
		if tpm == 0 {
			tpm = -1
		}
		rl.AddCredentialWithTPM(c.Name, c.RPM, tpm)
	}

	return &RoundRobin{
		credentials:  credentials,
		current:      0,
		fail2ban:     f2b,
		rateLimiter:  rl,
		modelChecker: nil,
	}
}

// SetModelChecker sets the model checker for filtering credentials by model
func (r *RoundRobin) SetModelChecker(mc ModelChecker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modelChecker = mc
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
		if allowOnlyFallback {
			// Looking for fallback proxy only
			if cred.Type != config.ProviderTypeProxy || !cred.IsFallback {
				otherReasonsHit = true
				continue
			}
		} else {
			// Looking for non-fallback credentials
			// Proxy type should NEVER be selected directly - must always be fallback or not used at all
			if cred.IsFallback || cred.Type == config.ProviderTypeProxy {
				continue
			}
		}

		// Check if credential is banned
		if r.fail2ban.IsBanned(cred.Name) {
			otherReasonsHit = true
			continue
		}

		// Check if credential supports the requested model BEFORE rate limiting
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.modelChecker.HasModel(cred.Name, modelID) {
				otherReasonsHit = true
				continue
			}
		}

		// Check credential RPM limit (without recording)
		if !r.rateLimiter.CanAllow(cred.Name) {
			rateLimitHit = true
			continue
		}

		// Check credential TPM limit
		if !r.rateLimiter.AllowTokens(cred.Name) {
			rateLimitHit = true
			continue
		}

		// Check model RPM limit if model is specified (without recording)
		if modelID != "" {
			if !r.rateLimiter.CanAllowModel(cred.Name, modelID) {
				// Model RPM exceeded for this credential+model combination
				rateLimitHit = true
				continue
			}
		}

		// Check model TPM limit if model is specified
		if modelID != "" {
			if !r.rateLimiter.AllowModelTokens(cred.Name, modelID) {
				// Model TPM exceeded for this credential+model combination
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
	return len(r.fail2ban.GetBannedCredentials())
}
