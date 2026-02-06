package health

import (
	"sync/atomic"
)

// DBHealthChecker provides cached health status of LiteLLM DB.
// Updated by periodic health monitor goroutine.
// Uses atomic operations for lock-free reads and minimal performance impact.
type DBHealthChecker struct {
	// Protected by atomic operations for lock-free reads
	// 1 = healthy, 0 = unhealthy
	dbHealthy *int32
}

// NewDBHealthChecker creates a new database health checker with initial healthy state.
func NewDBHealthChecker() *DBHealthChecker {
	healthy := int32(1) // Start healthy
	return &DBHealthChecker{
		dbHealthy: &healthy,
	}
}

// IsHealthy returns the cached health status without performing I/O.
// Uses atomic load for thread-safe read without locks.
func (hc *DBHealthChecker) IsHealthy() bool {
	if hc == nil || hc.dbHealthy == nil {
		// No health checker provided, default to healthy
		return true
	}
	return atomic.LoadInt32(hc.dbHealthy) == 1
}

// IsDBHealthy is an alias for IsHealthy to match the proxy.HealthChecker interface.
// This allows DBHealthChecker to be used directly by the proxy package.
func (hc *DBHealthChecker) IsDBHealthy() bool {
	return hc.IsHealthy()
}

// SetHealthy updates the health status atomically.
// This method is called by the health monitor goroutine.
func (hc *DBHealthChecker) SetHealthy(healthy bool) {
	if hc == nil || hc.dbHealthy == nil {
		return
	}

	healthValue := int32(0)
	if healthy {
		healthValue = 1
	}
	atomic.StoreInt32(hc.dbHealthy, healthValue)
}

// GetHealthValue returns the raw atomic value for direct atomic operations.
// Advanced usage: allows direct manipulation for special cases.
func (hc *DBHealthChecker) GetHealthValue() *int32 {
	if hc == nil {
		return nil
	}
	return hc.dbHealthy
}
