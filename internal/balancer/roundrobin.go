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
	typeCounters    map[config.ProviderType]int // per-type counters to prevent cross-type interference
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
		typeCounters:    make(map[config.ProviderType]int),
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
//
// The algorithm runs in two phases:
//  1. Build a candidate list via structural filters (exclude, type/fallback, model availability).
//     These are time-stable properties — they don't change between requests.
//  2. Select the next candidate using an independent per-type counter when all candidates
//     share the same ProviderType. This prevents high-frequency traffic of one provider type
//     (e.g. OpenAI) from interfering with the round-robin cycling of another (e.g. Vertex AI).
func (r *RoundRobin) nextExcluding(modelID string, allowOnlyFallback, allowOnlyProxy bool, exclude map[string]bool) (*config.CredentialConfig, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Phase 1: Build candidate list using only structural (time-stable) filters.
	type candidateEntry struct {
		absIdx int
		cred   *config.CredentialConfig
	}
	var candidates []candidateEntry

	for i := range r.credentials {
		cred := &r.credentials[i]

		if len(exclude) > 0 && exclude[cred.Name] {
			continue
		}

		if allowOnlyProxy && cred.Type != config.ProviderTypeProxy {
			monitoring.CredentialSelectionRejected.WithLabelValues("type_not_allowed").Inc()
			continue
		}

		if allowOnlyFallback {
			if !cred.IsFallback {
				monitoring.CredentialSelectionRejected.WithLabelValues("fallback_not_available").Inc()
				continue
			}
		} else if cred.IsFallback {
			monitoring.CredentialSelectionRejected.WithLabelValues("fallback_only").Inc()
			continue
		}

		// Check model availability before ban/rate checks.
		// model_not_available is a structural property, not a temporary issue.
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.modelChecker.HasModel(cred.Name, modelID) {
				monitoring.CredentialSelectionRejected.WithLabelValues("model_not_available").Inc()
				continue
			}
		}

		candidates = append(candidates, candidateEntry{absIdx: i, cred: cred})
	}

	if len(candidates) == 0 {
		return nil, ErrNoCredentialsAvailable
	}

	// Phase 2: Determine start offset using a per-type counter when all candidates
	// share the same ProviderType; otherwise fall back to the global counter.
	sameType := true
	candidateType := candidates[0].cred.Type
	for _, c := range candidates[1:] {
		if c.cred.Type != candidateType {
			sameType = false
			break
		}
	}

	startOffset := 0
	if sameType {
		typeStart := r.typeCounters[candidateType]
		for i, c := range candidates {
			if c.absIdx >= typeStart {
				startOffset = i
				break
			}
		}
		// If typeStart is past all candidates, wrap to beginning (startOffset stays 0).
	} else {
		globalStart := r.current
		for i, c := range candidates {
			if c.absIdx >= globalStart {
				startOffset = i
				break
			}
		}
		// If globalStart is past all candidates, wrap to beginning.
	}

	// Phase 3: Try candidates in round-robin order, applying ban and rate-limit checks.
	rateLimitHit := false
	for i := 0; i < len(candidates); i++ {
		ci := (startOffset + i) % len(candidates)
		c := candidates[ci]

		if r.fail2ban.IsBanned(c.cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("banned").Inc()
			continue
		}

		// Atomically check all rate limits (credential RPM/TPM + model RPM/TPM)
		// and record usage only if all checks pass. This prevents TOCTOU races
		// where separate check+record calls could allow exceeding limits.
		if !r.rateLimiter.TryAllowAll(c.cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}

		// Advance the appropriate counter past the selected credential.
		nextIdx := (c.absIdx + 1) % len(r.credentials)
		if sameType {
			r.typeCounters[candidateType] = nextIdx
		} else {
			r.current = nextIdx
		}

		return c.cred, nil
	}

	// Prioritize rate limit error: if any candidate hit rate limit, surface it even if
	// others were banned. This gives callers accurate signal for backoff/retry logic.
	if rateLimitHit {
		return nil, ErrRateLimitExceeded
	}
	// All candidates are banned (or none remain after ban + rate-limit filtering).
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
