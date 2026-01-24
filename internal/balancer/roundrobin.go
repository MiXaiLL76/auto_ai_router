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
)

type Credential struct {
	Name    string
	APIKey  string
	BaseURL string
	RPM     int
}

type RoundRobin struct {
	mu           sync.Mutex
	credentials  []Credential
	current      int
	fail2ban     *fail2ban.Fail2Ban
	rateLimiter  *ratelimit.RPMLimiter
	modelChecker ModelChecker
}

func New(credentials []config.CredentialConfig, f2b *fail2ban.Fail2Ban, rl *ratelimit.RPMLimiter) *RoundRobin {
	creds := make([]Credential, len(credentials))
	for i, c := range credentials {
		creds[i] = Credential{
			Name:    c.Name,
			APIKey:  c.APIKey,
			BaseURL: c.BaseURL,
			RPM:     c.RPM,
		}
		rl.AddCredential(c.Name, c.RPM)
	}

	return &RoundRobin{
		credentials:  creds,
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

func (r *RoundRobin) Next() (*Credential, error) {
	return r.next("")
}

// NextForModel returns the next available credential that supports the specified model
func (r *RoundRobin) NextForModel(modelID string) (*Credential, error) {
	return r.next(modelID)
}

func (r *RoundRobin) next(modelID string) (*Credential, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	attempts := 0

	for {
		if attempts >= len(r.credentials) {
			return nil, ErrNoCredentialsAvailable
		}

		cred := &r.credentials[r.current]
		r.current = (r.current + 1) % len(r.credentials)
		attempts++

		// Check if credential is banned
		if r.fail2ban.IsBanned(cred.Name) {
			continue
		}

		// Check RPM limit
		if !r.rateLimiter.Allow(cred.Name) {
			continue
		}

		// Check if credential supports the requested model
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.modelChecker.HasModel(cred.Name, modelID) {
				continue
			}
		}

		return cred, nil
	}
}

func (r *RoundRobin) RecordResponse(credentialName string, statusCode int) {
	r.fail2ban.RecordResponse(credentialName, statusCode)
}

func (r *RoundRobin) GetCredentials() []Credential {
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
