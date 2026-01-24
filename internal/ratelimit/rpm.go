package ratelimit

import (
	"sync"
	"time"
)

type RPMLimiter struct {
	mu       sync.RWMutex
	limiters map[string]*credentialLimiter
}

type credentialLimiter struct {
	rpm       int
	requests  []time.Time
	mu        sync.Mutex
}

func New() *RPMLimiter {
	return &RPMLimiter{
		limiters: make(map[string]*credentialLimiter),
	}
}

func (r *RPMLimiter) AddCredential(name string, rpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.limiters[name] = &credentialLimiter{
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
