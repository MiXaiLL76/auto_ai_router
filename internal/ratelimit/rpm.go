package ratelimit

import (
	"fmt"
	"strings"
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

// MaxRequestsBufferSize limits the maximum number of request timestamps stored
// This prevents unbounded memory growth. 10x the typical max RPM (1M) is safe upper bound
const MaxRequestsBufferSize = 10_000_000

// MaxTokensBufferSize limits the maximum number of token records stored
// Similar protection for token tracking
const MaxTokensBufferSize = 10_000_000

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

// getCredentialLimiter retrieves credential limiter safely
// Returns nil if credential not found
func (r *RPMLimiter) getCredentialLimiter(credentialName string) *limiter {
	r.mu.RLock()
	limiter := r.limiters[credentialName]
	r.mu.RUnlock()
	return limiter
}

// getModelLimiter retrieves model limiter safely
// Returns nil if model not tracked
func (r *RPMLimiter) getModelLimiter(credentialName, modelName string) *limiter {
	r.mu.RLock()
	key := makeModelKey(credentialName, modelName)
	limiter := r.modelLimiters[key]
	r.mu.RUnlock()
	return limiter
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

// setCurrentUsage fills request and token arrays to simulate current usage
// Must be called with limiter.mu locked
func setCurrentUsage(limiter *limiter, currentRPM, currentTPM int) {
	now := time.Now().UTC()
	oneMinuteAgo := now.Add(-time.Minute)

	// Fill requests array with dummy timestamps distributed over the minute
	if currentRPM > 0 {
		limiter.requests = make([]time.Time, currentRPM)
		for i := 0; i < currentRPM; i++ {
			// Distribute requests evenly over the last minute
			// Use int64 to prevent integer overflow for large RPM values
			offset := time.Duration(int64(i)*60000/int64(currentRPM)) * time.Millisecond
			limiter.requests[i] = oneMinuteAgo.Add(offset)
		}
	} else {
		limiter.requests = make([]time.Time, 0)
	}

	// Fill tokens array similarly for TPM
	if currentTPM > 0 {
		limiter.tokens = make([]tokenUsage, currentTPM)
		for i := 0; i < currentTPM; i++ {
			// Use int64 to prevent integer overflow for large TPM values
			offset := time.Duration(int64(i)*60000/int64(currentTPM)) * time.Millisecond
			limiter.tokens[i] = tokenUsage{
				timestamp: oneMinuteAgo.Add(offset),
				count:     1,
			}
		}
	} else {
		limiter.tokens = make([]tokenUsage, 0)
	}
}

// SetCredentialCurrentUsage sets the current RPM/TPM usage for a credential
// Used to synchronize usage from remote proxies
func (r *RPMLimiter) SetCredentialCurrentUsage(credentialName string, currentRPM, currentTPM int) {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	setCurrentUsage(limiter, currentRPM, currentTPM)
}

// SetModelCurrentUsage sets the current RPM/TPM usage for a model
// Used to synchronize usage from remote proxies
func (r *RPMLimiter) SetModelCurrentUsage(credentialName, modelName string, currentRPM, currentTPM int) {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		return
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	setCurrentUsage(modelLimiter, currentRPM, currentTPM)
}

// checkRPMLimit checks if RPM limit allows request and optionally records it
// Must be called with limiter.mu locked
func checkRPMLimit(l *limiter, record bool) bool {
	cleanOldRequests(l)

	// Check limit only if RPM is not unlimited (-1)
	if l.rpm != -1 && len(l.requests) >= l.rpm {
		return false
	}

	// Record the request if requested
	if record {
		// Always record - but clean old requests first if buffer is full
		if len(l.requests) >= MaxRequestsBufferSize {
			cleanOldRequests(l)
		}
		// Only skip if still at capacity after cleaning (extremely rare edge case)
		if len(l.requests) < MaxRequestsBufferSize {
			l.requests = append(l.requests, time.Now().UTC())
		}
	}

	return true
}

func (r *RPMLimiter) Allow(credentialName string) bool {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return false
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return checkRPMLimit(limiter, true)
}

// CanAllow checks if a request would be allowed without recording it
func (r *RPMLimiter) CanAllow(credentialName string) bool {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return false
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return checkRPMLimit(limiter, false)
}

// cleanOldRequests removes requests older than 1 minute and returns count of valid ones
// Must be called with limiter.mu locked
func cleanOldRequests(l *limiter) int {
	now := time.Now().UTC()
	oneMinuteAgo := now.Add(-time.Minute)

	// Pre-allocate with original capacity to avoid excessive allocations
	validRequests := make([]time.Time, 0, len(l.requests))
	for _, reqTime := range l.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	l.requests = validRequests

	return len(validRequests)
}

// cleanOldTokens removes tokens older than 1 minute and returns total count
// Must be called with limiter.mu locked
func cleanOldTokens(l *limiter) int {
	now := time.Now().UTC()
	oneMinuteAgo := now.Add(-time.Minute)

	// Pre-allocate with original capacity to avoid excessive allocations
	validTokens := make([]tokenUsage, 0, len(l.tokens))
	count := 0
	for _, tu := range l.tokens {
		if tu.timestamp.After(oneMinuteAgo) {
			validTokens = append(validTokens, tu)
			count += tu.count
		}
	}
	l.tokens = validTokens

	return count
}

func (r *RPMLimiter) GetCurrentRPM(credentialName string) int {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return 0
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return cleanOldRequests(limiter)
}

// AllowModel checks if request to a specific model for a credential is allowed based on RPM limit
func (r *RPMLimiter) AllowModel(credentialName, modelName string) bool {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		// Model not tracked for this credential, allow request
		return true
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return checkRPMLimit(modelLimiter, true)
}

// CanAllowModel checks if a model request would be allowed without recording it
func (r *RPMLimiter) CanAllowModel(credentialName, modelName string) bool {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		// Model not tracked for this credential, allow request
		return true
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return checkRPMLimit(modelLimiter, false)
}

// GetCurrentModelRPM returns current RPM for a model within a credential
func (r *RPMLimiter) GetCurrentModelRPM(credentialName, modelName string) int {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		return 0
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return cleanOldRequests(modelLimiter)
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

// ModelPair represents a parsed credential:model pair
type ModelPair struct {
	Credential string
	Model      string
}

// GetAllModelPairs returns list of all tracked (credential:model) pairs pre-parsed
func (r *RPMLimiter) GetAllModelPairs() []ModelPair {
	r.mu.RLock()
	defer r.mu.RUnlock()

	pairs := make([]ModelPair, 0, len(r.modelLimiters))
	for key := range r.modelLimiters {
		// key format is always "credential:model" since we control the key creation
		parts := strings.Split(key, ":")
		if len(parts) == 2 {
			pairs = append(pairs, ModelPair{Credential: parts[0], Model: parts[1]})
		}
	}
	return pairs
}

// ConsumeTokens records token usage for a credential
func (r *RPMLimiter) ConsumeTokens(credentialName string, tokenCount int) {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	// Always record - but clean old tokens first if buffer is full
	if len(limiter.tokens) >= MaxTokensBufferSize {
		cleanOldTokens(limiter)
	}
	// Only skip if still at capacity after cleaning (extremely rare edge case)
	if len(limiter.tokens) < MaxTokensBufferSize {
		limiter.tokens = append(limiter.tokens, tokenUsage{
			timestamp: time.Now().UTC(),
			count:     tokenCount,
		})
	}
}

// checkTPMLimit checks if TPM limit allows request
// Must be called with limiter.mu locked
func checkTPMLimit(l *limiter) bool {
	// -1 means unlimited TPM
	if l.tpm == -1 {
		return true
	}

	currentTPM := cleanOldTokens(l)

	// Check if we're at or over the limit
	return currentTPM < l.tpm
}

// AllowTokens checks if the given number of tokens can be consumed without exceeding TPM limit
// This is a pre-check before making a request, but actual token count will be updated after response
func (r *RPMLimiter) AllowTokens(credentialName string) bool {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return false
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return checkTPMLimit(limiter)
}

// GetCurrentTPM returns current token usage per minute for a credential
func (r *RPMLimiter) GetCurrentTPM(credentialName string) int {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return 0
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return cleanOldTokens(limiter)
}

// ConsumeModelTokens records token usage for a model within a credential
func (r *RPMLimiter) ConsumeModelTokens(credentialName, modelName string, tokenCount int) {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		return
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	// Always record - but clean old tokens first if buffer is full
	if len(modelLimiter.tokens) >= MaxTokensBufferSize {
		cleanOldTokens(modelLimiter)
	}
	// Only skip if still at capacity after cleaning (extremely rare edge case)
	if len(modelLimiter.tokens) < MaxTokensBufferSize {
		modelLimiter.tokens = append(modelLimiter.tokens, tokenUsage{
			timestamp: time.Now().UTC(),
			count:     tokenCount,
		})
	}
}

// AllowModelTokens checks if request to a specific model for a credential is allowed based on TPM limit
func (r *RPMLimiter) AllowModelTokens(credentialName, modelName string) bool {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		// Model not tracked for this credential, allow request
		return true
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return checkTPMLimit(modelLimiter)
}

// GetCurrentModelTPM returns current token usage per minute for a model within a credential
func (r *RPMLimiter) GetCurrentModelTPM(credentialName, modelName string) int {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		return 0
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return cleanOldTokens(modelLimiter)
}

// GetLimitRPM returns the RPM limit for a credential
func (r *RPMLimiter) GetLimitRPM(credentialName string) int {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return -1 // Not tracked
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return limiter.rpm
}

// GetLimitTPM returns the TPM limit for a credential
func (r *RPMLimiter) GetLimitTPM(credentialName string) int {
	limiter := r.getCredentialLimiter(credentialName)
	if limiter == nil {
		return -1 // Not tracked
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()

	return limiter.tpm
}

// GetModelLimitRPM returns the RPM limit for a model within a credential
func (r *RPMLimiter) GetModelLimitRPM(credentialName, modelName string) int {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		return -1 // Not tracked
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return modelLimiter.rpm
}

// GetModelLimitTPM returns the TPM limit for a model within a credential
func (r *RPMLimiter) GetModelLimitTPM(credentialName, modelName string) int {
	modelLimiter := r.getModelLimiter(credentialName, modelName)
	if modelLimiter == nil {
		return -1 // Not tracked
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	return modelLimiter.tpm
}
