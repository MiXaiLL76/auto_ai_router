package proxyhealth

import (
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// Tracker monitors the health status of proxy credentials.
// Tracks successful and failed health checks, maintains state changes for logging.
// Uses mutex for thread-safe access to health state.
type Tracker struct {
	mu                   sync.RWMutex
	healthStatus         map[string]bool      // true = healthy, false = unhealthy
	failureCount         map[string]int       // consecutive failures per proxy
	lastStatusChangeTime map[string]time.Time // track when status changed
	recovered            map[string]bool      // one-shot recovered markers
}

// NewTracker creates a new proxy health tracker with empty state.
func NewTracker() *Tracker {
	return &Tracker{
		healthStatus:         make(map[string]bool),
		failureCount:         make(map[string]int),
		lastStatusChangeTime: make(map[string]time.Time),
		recovered:            make(map[string]bool),
	}
}

// RecordSuccess records a successful health check for a proxy.
// Resets the failure counter and marks the proxy as healthy.
func (t *Tracker) RecordSuccess(proxyName string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	wasHealthy := t.healthStatus[proxyName]
	t.healthStatus[proxyName] = true
	t.failureCount[proxyName] = 0

	// Record status change time if transitioning from unhealthy to healthy
	if !wasHealthy {
		t.lastStatusChangeTime[proxyName] = utils.NowUTC()
		t.recovered[proxyName] = true
	}
}

// RecordFailure records a failed health check for a proxy.
// Increments the failure counter and may mark the proxy as unhealthy.
func (t *Tracker) RecordFailure(proxyName string, _ error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	wasHealthy := t.healthStatus[proxyName]
	t.failureCount[proxyName]++

	// For simplicity: mark unhealthy after first failure
	// Callers can implement circuit breaker logic with threshold checking
	if t.failureCount[proxyName] >= 1 {
		t.healthStatus[proxyName] = false

		// Record status change time if transitioning from healthy to unhealthy
		if wasHealthy {
			t.lastStatusChangeTime[proxyName] = utils.NowUTC()
		}
		delete(t.recovered, proxyName)
	}
}

// GetFailureCount returns the number of consecutive failures for a proxy.
func (t *Tracker) GetFailureCount(proxyName string) int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.failureCount[proxyName]
}

// IsUnhealthy returns true if the proxy is marked as unhealthy.
func (t *Tracker) IsUnhealthy(proxyName string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()

	healthy, exists := t.healthStatus[proxyName]
	if !exists {
		return false // unknown proxy is assumed healthy (fail-open)
	}
	return !healthy
}

// IsHealthy returns true if the proxy is marked as healthy.
func (t *Tracker) IsHealthy(proxyName string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.healthStatus[proxyName]
}

// GetFailedNames returns a slice of proxy names that are currently unhealthy.
func (t *Tracker) GetFailedNames() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	failed := make([]string, 0)
	for name, isHealthy := range t.healthStatus {
		if !isHealthy {
			failed = append(failed, name)
		}
	}
	return failed
}

// GetRecoveredNames returns a slice of proxy names that recovered from unhealthy state.
// A proxy is considered "recovered" if it was previously unhealthy and now is healthy,
// based on the lastStatusChangeTime tracking.
func (t *Tracker) GetRecoveredNames() []string {
	t.mu.Lock()
	defer t.mu.Unlock()

	recovered := make([]string, 0)
	for name := range t.recovered {
		if t.healthStatus[name] {
			recovered = append(recovered, name)
		}
		delete(t.recovered, name)
	}

	return recovered
}

// ResetFailureCount resets the failure counter for a proxy.
// Useful when a proxy recovers after being unhealthy.
func (t *Tracker) ResetFailureCount(proxyName string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.failureCount[proxyName] = 0
}

// GetStatus returns the current health status and failure count for a proxy.
func (t *Tracker) GetStatus(proxyName string) (healthy bool, failureCount int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.healthStatus[proxyName], t.failureCount[proxyName]
}

// GetAllStatuses returns a snapshot of all proxy health statuses.
func (t *Tracker) GetAllStatuses() map[string]bool {
	t.mu.RLock()
	defer t.mu.RUnlock()

	snapshot := make(map[string]bool)
	for name, healthy := range t.healthStatus {
		snapshot[name] = healthy
	}
	return snapshot
}
