package ratelimit

import (
	"fmt"
	"sync"
	"time"
)

type RPMLimiter struct {
	mu            sync.RWMutex
	limiters      map[string]*limiter // credential limiters
	modelLimiters map[string]*limiter // (credential:model) limiters
}

type tokenUsage struct {
	timestamp time.Time
	count     int
}

type limiter struct {
	rpm      int
	tpm      int
	requests []time.Time
	tokens   []tokenUsage
	mu       sync.Mutex
}

func New() *RPMLimiter {
	return &RPMLimiter{
		limiters:      make(map[string]*limiter),
		modelLimiters: make(map[string]*limiter),
	}
}

// makeModelKey creates a key for (credential, model) pair
func makeModelKey(credentialName, modelName string) string {
	return fmt.Sprintf("%s:%s", credentialName, modelName)
}

func (r *RPMLimiter) AddCredential(name string, rpm int) {
	r.AddCredentialWithTPM(name, rpm, -1)
}

func (r *RPMLimiter) AddCredentialWithTPM(name string, rpm int, tpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.limiters[name] = &limiter{
		rpm:      rpm,
		tpm:      tpm,
		requests: make([]time.Time, 0),
		tokens:   make([]tokenUsage, 0),
	}
}

// AddModel adds a model with RPM limit for a specific credential
func (r *RPMLimiter) AddModel(credentialName, modelName string, rpm int) {
	r.AddModelWithTPM(credentialName, modelName, rpm, -1)
}

// AddModelWithTPM adds a model with both RPM and TPM limits for a specific credential
func (r *RPMLimiter) AddModelWithTPM(credentialName, modelName string, rpm int, tpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := makeModelKey(credentialName, modelName)
	r.modelLimiters[key] = &limiter{
		rpm:      rpm,
		tpm:      tpm,
		requests: make([]time.Time, 0),
		tokens:   make([]tokenUsage, 0),
	}
}

func (r *RPMLimiter) Allow(credentialName string) bool {
	r.mu.RLock()
	limiter, exists := r.limiters[credentialName]
	r.mu.RUnlock()

	if !exists {
		return false
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old requests first
	validRequests := make([]time.Time, 0)
	for _, reqTime := range limiter.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	limiter.requests = validRequests

	// Check limit only if RPM is not unlimited (-1)
	if limiter.rpm != -1 && len(limiter.requests) >= limiter.rpm {
		return false
	}

	// Always record the request for metrics
	limiter.requests = append(limiter.requests, now)
	return true
}

func (r *RPMLimiter) GetCurrentRPM(credentialName string) int {
	r.mu.RLock()
	limiter, exists := r.limiters[credentialName]
	r.mu.RUnlock()

	if !exists {
		return 0
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old requests and count valid ones
	validRequests := make([]time.Time, 0)
	for _, reqTime := range limiter.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	limiter.requests = validRequests

	return len(validRequests)
}

// AllowModel checks if request to a specific model for a credential is allowed based on RPM limit
func (r *RPMLimiter) AllowModel(credentialName, modelName string) bool {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		// Model not tracked for this credential, allow request
		return true
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old requests first
	validRequests := make([]time.Time, 0)
	for _, reqTime := range modelLimiter.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	modelLimiter.requests = validRequests

	// Check limit only if RPM is not unlimited (-1)
	if modelLimiter.rpm != -1 && len(modelLimiter.requests) >= modelLimiter.rpm {
		return false
	}

	// Always record the request for metrics
	modelLimiter.requests = append(modelLimiter.requests, now)
	return true
}

// GetCurrentModelRPM returns current RPM for a model within a credential
func (r *RPMLimiter) GetCurrentModelRPM(credentialName, modelName string) int {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		return 0
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old requests and count valid ones
	validRequests := make([]time.Time, 0)
	for _, reqTime := range modelLimiter.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	modelLimiter.requests = validRequests

	return len(validRequests)
}

// GetAllModels returns list of all tracked (credential:model) keys
func (r *RPMLimiter) GetAllModels() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	keys := make([]string, 0, len(r.modelLimiters))
	for key := range r.modelLimiters {
		keys = append(keys, key)
	}
	return keys
}

// ConsumeTokens records token usage for a credential
func (r *RPMLimiter) ConsumeTokens(credentialName string, tokenCount int) {
	r.mu.RLock()
	limiter, exists := r.limiters[credentialName]
	r.mu.RUnlock()

	if !exists {
		return
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	limiter.tokens = append(limiter.tokens, tokenUsage{
		timestamp: time.Now(),
		count:     tokenCount,
	})
}

// AllowTokens checks if the given number of tokens can be consumed without exceeding TPM limit
// This is a pre-check before making a request, but actual token count will be updated after response
func (r *RPMLimiter) AllowTokens(credentialName string) bool {
	r.mu.RLock()
	limiter, exists := r.limiters[credentialName]
	r.mu.RUnlock()

	if !exists {
		return false
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	// -1 means unlimited TPM
	if limiter.tpm == -1 {
		return true
	}

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old token usages and calculate current TPM
	validTokens := make([]tokenUsage, 0)
	currentTPM := 0
	for _, tu := range limiter.tokens {
		if tu.timestamp.After(oneMinuteAgo) {
			validTokens = append(validTokens, tu)
			currentTPM += tu.count
		}
	}
	limiter.tokens = validTokens

	// Check if we're at or over the limit
	if currentTPM >= limiter.tpm {
		return false
	}

	return true
}

// GetCurrentTPM returns current token usage per minute for a credential
func (r *RPMLimiter) GetCurrentTPM(credentialName string) int {
	r.mu.RLock()
	limiter, exists := r.limiters[credentialName]
	r.mu.RUnlock()

	if !exists {
		return 0
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old token usages and count valid ones
	validTokens := make([]tokenUsage, 0)
	count := 0
	for _, tu := range limiter.tokens {
		if tu.timestamp.After(oneMinuteAgo) {
			validTokens = append(validTokens, tu)
			count += tu.count
		}
	}
	limiter.tokens = validTokens

	return count
}

// ConsumeModelTokens records token usage for a model within a credential
func (r *RPMLimiter) ConsumeModelTokens(credentialName, modelName string, tokenCount int) {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		return
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	modelLimiter.tokens = append(modelLimiter.tokens, tokenUsage{
		timestamp: time.Now(),
		count:     tokenCount,
	})
}

// AllowModelTokens checks if request to a specific model for a credential is allowed based on TPM limit
func (r *RPMLimiter) AllowModelTokens(credentialName, modelName string) bool {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		// Model not tracked for this credential, allow request
		return true
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	// -1 means unlimited TPM
	if modelLimiter.tpm == -1 {
		return true
	}

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old token usages and calculate current TPM
	validTokens := make([]tokenUsage, 0)
	currentTPM := 0
	for _, tu := range modelLimiter.tokens {
		if tu.timestamp.After(oneMinuteAgo) {
			validTokens = append(validTokens, tu)
			currentTPM += tu.count
		}
	}
	modelLimiter.tokens = validTokens

	// Check if we're at or over the limit
	if currentTPM >= modelLimiter.tpm {
		return false
	}

	return true
}

// GetCurrentModelTPM returns current token usage per minute for a model within a credential
func (r *RPMLimiter) GetCurrentModelTPM(credentialName, modelName string) int {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		return 0
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old token usages and count valid ones
	validTokens := make([]tokenUsage, 0)
	count := 0
	for _, tu := range modelLimiter.tokens {
		if tu.timestamp.After(oneMinuteAgo) {
			validTokens = append(validTokens, tu)
			count += tu.count
		}
	}
	modelLimiter.tokens = validTokens

	return count
}

// GetModelLimitRPM returns the RPM limit for a model within a credential
func (r *RPMLimiter) GetModelLimitRPM(credentialName, modelName string) int {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		return -1 // Not tracked
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return modelLimiter.rpm
}

// GetModelLimitTPM returns the TPM limit for a model within a credential
func (r *RPMLimiter) GetModelLimitTPM(credentialName, modelName string) int {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	modelLimiter, exists := r.modelLimiters[key]
	r.mu.RUnlock()

	if !exists {
		return -1 // Not tracked
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return modelLimiter.tpm
}
