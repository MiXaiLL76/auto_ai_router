package balancer

import (
	"errors"
	"log/slog"
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
	logger       *slog.Logger
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
		logger:       slog.New(slog.NewTextHandler(nil, nil)), // Default no-op logger
	}
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

	r.logger.Debug("starting credential selection",
		"model_id", modelID,
		"allow_only_fallback", allowOnlyFallback,
		"total_credentials", len(r.credentials),
	)

	attempts := 0
	rateLimitHit := false
	otherReasonsHit := false

	for {
		if attempts >= len(r.credentials) {
			r.logger.Debug("exhausted all credentials",
				"model_id", modelID,
				"total_attempts", attempts,
				"rate_limit_hit", rateLimitHit,
				"other_reasons_hit", otherReasonsHit,
			)

			// If all credentials were blocked due to rate limit, return rate limit error
			if rateLimitHit && !otherReasonsHit {
				r.logger.Debug("returning rate limit exceeded error")
				return nil, ErrRateLimitExceeded
			}
			// Otherwise return generic error
			r.logger.Debug("returning no credentials available error")
			return nil, ErrNoCredentialsAvailable
		}

		cred := &r.credentials[r.current]
		prevIndex := r.current
		r.current = (r.current + 1) % len(r.credentials)
		attempts++

		r.logger.Debug("checking credential",
			"attempt", attempts,
			"credential_index", prevIndex,
			"credential_name", cred.Name,
			"credential_type", cred.Type,
			"is_fallback", cred.IsFallback,
		)

		// Filter by is_fallback flag
		if allowOnlyFallback {
			// Looking for fallback proxy only
			if cred.Type != config.ProviderTypeProxy || !cred.IsFallback {
				r.logger.Debug("credential filtered by fallback requirement",
					"credential_name", cred.Name,
					"reason", "not a fallback proxy",
					"credential_type", cred.Type,
					"is_fallback", cred.IsFallback,
				)
				otherReasonsHit = true
				continue
			}
		}

		// Check if credential is banned
		if r.fail2ban.IsBanned(cred.Name) {
			r.logger.Debug("credential is banned",
				"credential_name", cred.Name,
			)
			otherReasonsHit = true
			continue
		}

		// Check if credential supports the requested model BEFORE rate limiting
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.modelChecker.HasModel(cred.Name, modelID) {
				r.logger.Debug("credential does not support model",
					"credential_name", cred.Name,
					"model_id", modelID,
				)
				otherReasonsHit = true
				continue
			}
			r.logger.Debug("credential supports model",
				"credential_name", cred.Name,
				"model_id", modelID,
			)
		}

		// Check credential RPM limit (without recording)
		if !r.rateLimiter.CanAllow(cred.Name) {
			r.logger.Debug("credential RPM limit exceeded",
				"credential_name", cred.Name,
			)
			rateLimitHit = true
			continue
		}
		r.logger.Debug("credential RPM limit check passed",
			"credential_name", cred.Name,
		)

		// Check credential TPM limit
		if !r.rateLimiter.AllowTokens(cred.Name) {
			r.logger.Debug("credential TPM limit exceeded",
				"credential_name", cred.Name,
			)
			rateLimitHit = true
			continue
		}
		r.logger.Debug("credential TPM limit check passed",
			"credential_name", cred.Name,
		)

		// Check model RPM limit if model is specified (without recording)
		if modelID != "" {
			if !r.rateLimiter.CanAllowModel(cred.Name, modelID) {
				// Model RPM exceeded for this credential+model combination
				r.logger.Debug("model RPM limit exceeded",
					"credential_name", cred.Name,
					"model_id", modelID,
				)
				rateLimitHit = true
				continue
			}
			r.logger.Debug("model RPM limit check passed",
				"credential_name", cred.Name,
				"model_id", modelID,
			)
		}

		// Check model TPM limit if model is specified
		if modelID != "" {
			if !r.rateLimiter.AllowModelTokens(cred.Name, modelID) {
				// Model TPM exceeded for this credential+model combination
				r.logger.Debug("model TPM limit exceeded",
					"credential_name", cred.Name,
					"model_id", modelID,
				)
				rateLimitHit = true
				continue
			}
			r.logger.Debug("model TPM limit check passed",
				"credential_name", cred.Name,
				"model_id", modelID,
			)
		}

		// All checks passed - now record the requests
		r.rateLimiter.Allow(cred.Name) // Record credential RPM
		if modelID != "" {
			r.rateLimiter.AllowModel(cred.Name, modelID) // Record model RPM
		}

		r.logger.Debug("credential selected successfully",
			"credential_name", cred.Name,
			"model_id", modelID,
			"total_attempts", attempts,
		)

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
