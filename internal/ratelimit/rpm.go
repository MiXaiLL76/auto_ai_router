package ratelimit

import (
	"sync"
	"time"
)

type RPMLimiter struct {
	mu             sync.RWMutex
	limiters       map[string]*limiter // credential limiters
	modelLimiters  map[string]*limiter // model limiters
}

type limiter struct {
	rpm       int
	requests  []time.Time
	mu        sync.Mutex
}

func New() *RPMLimiter {
	return &RPMLimiter{
		limiters:      make(map[string]*limiter),
		modelLimiters: make(map[string]*limiter),
	}
}

func (r *RPMLimiter) AddCredential(name string, rpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.limiters[name] = &limiter{
		rpm:      rpm,
		requests: make([]time.Time, 0),
	}
}

// AddModel adds a model with RPM limit
func (r *RPMLimiter) AddModel(modelName string, rpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.modelLimiters[modelName] = &limiter{
		rpm:      rpm,
		requests: make([]time.Time, 0),
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

	validRequests := make([]time.Time, 0)
	for _, reqTime := range limiter.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	limiter.requests = validRequests

	if len(limiter.requests) >= limiter.rpm {
		return false
	}

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

	count := 0
	for _, reqTime := range limiter.requests {
		if reqTime.After(oneMinuteAgo) {
			count++
		}
	}

	return count
}

// AllowModel checks if request to a specific model is allowed based on RPM limit
func (r *RPMLimiter) AllowModel(modelName string) bool {
	r.mu.RLock()
	modelLimiter, exists := r.modelLimiters[modelName]
	r.mu.RUnlock()

	if !exists {
		// Model not tracked, allow request
		return true
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	// Clean old requests
	validRequests := make([]time.Time, 0)
	for _, reqTime := range modelLimiter.requests {
		if reqTime.After(oneMinuteAgo) {
			validRequests = append(validRequests, reqTime)
		}
	}
	modelLimiter.requests = validRequests

	// Check limit
	if len(modelLimiter.requests) >= modelLimiter.rpm {
		return false
	}

	// Record request
	modelLimiter.requests = append(modelLimiter.requests, now)
	return true
}

// GetCurrentModelRPM returns current RPM for a model
func (r *RPMLimiter) GetCurrentModelRPM(modelName string) int {
	r.mu.RLock()
	modelLimiter, exists := r.modelLimiters[modelName]
	r.mu.RUnlock()

	if !exists {
		return 0
	}

	modelLimiter.mu.Lock()
	defer modelLimiter.mu.Unlock()

	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)

	count := 0
	for _, reqTime := range modelLimiter.requests {
		if reqTime.After(oneMinuteAgo) {
			count++
		}
	}

	return count
}

// GetAllModels returns list of all tracked model names
func (r *RPMLimiter) GetAllModels() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	models := make([]string, 0, len(r.modelLimiters))
	for modelName := range r.modelLimiters {
		models = append(models, modelName)
	}
	return models
}
