package health

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
)

// MonitorConfig contains configuration for the database health monitor.
type MonitorConfig struct {
	// Interval between health checks
	CheckInterval time.Duration
	// Number of consecutive failures before marking unhealthy
	FailureThreshold int32
	// Logger for health check events
	Logger *slog.Logger
}

// MonitorStats contains statistics about the health monitor.
type MonitorStats struct {
	LastCheckTime       time.Time
	ConsecutiveFailures int32
	IsHealthy           bool
	AcquiredConns       int32
	AvailableConns      int32
}

// Monitor manages periodic database health checks and updates a DBHealthChecker.
// It implements a circuit breaker pattern with configurable failure thresholds.
type Monitor struct {
	config              *MonitorConfig
	healthChecker       *DBHealthChecker
	dbManager           litellmdb.Manager
	consecutiveFailures int32
	lastCheckTime       time.Time
	mu                  sync.RWMutex
}

// NewMonitor creates a new database health monitor.
func NewMonitor(
	cfg *MonitorConfig,
	healthChecker *DBHealthChecker,
	dbManager litellmdb.Manager,
) *Monitor {
	if cfg == nil {
		cfg = &MonitorConfig{
			CheckInterval:    30 * time.Second,
			FailureThreshold: 3,
			Logger:           slog.Default(),
		}
	}

	if cfg.CheckInterval == 0 {
		cfg.CheckInterval = 30 * time.Second
	}

	if cfg.FailureThreshold == 0 {
		cfg.FailureThreshold = 3
	}

	return &Monitor{
		config:        cfg,
		healthChecker: healthChecker,
		dbManager:     dbManager,
		lastCheckTime: time.Now().UTC(),
	}
}

// Start begins the health monitoring loop.
// Runs in a background goroutine, checking database health periodically.
// Blocks until the context is cancelled.
func (m *Monitor) Start(ctx context.Context) {
	ticker := time.NewTicker(m.config.CheckInterval)
	defer ticker.Stop()

	m.config.Logger.Info("Database health monitor started",
		"check_interval", m.config.CheckInterval,
		"failure_threshold", m.config.FailureThreshold,
	)

	for {
		select {
		case <-ctx.Done():
			m.config.Logger.Info("Database health monitor stopped")
			return

		case <-ticker.C:
			m.checkHealth()
		}
	}
}

// checkHealth performs a single health check and updates the health checker.
func (m *Monitor) checkHealth() {
	now := time.Now().UTC()
	isHealthy := m.dbManager.IsHealthy()

	// Load current cached state (thread-safe)
	wasHealthy := m.healthChecker.IsHealthy()

	if isHealthy {
		// Health check passed - reset failure counter
		atomic.StoreInt32(&m.consecutiveFailures, 0)

		// If transitioning from unhealthy to healthy, log recovery
		if !wasHealthy {
			m.config.Logger.Warn("Database recovered (state: unhealthy -> healthy)",
				"timestamp", now.Format(time.RFC3339),
				"last_check_time", m.lastCheckTime.Format(time.RFC3339),
				"check_duration", now.Sub(m.lastCheckTime),
			)
		}

		m.healthChecker.SetHealthy(true)
	} else {
		// Health check failed - increment failure counter
		failures := atomic.AddInt32(&m.consecutiveFailures, 1)

		// Log each failure once, then periodically
		if failures == 1 {
			// First failure - log at WARN level
			m.config.Logger.Warn("Database health check failed",
				"timestamp", now.Format(time.RFC3339),
				"failure_count", failures,
				"threshold", m.config.FailureThreshold,
				"impact", fmt.Sprintf(
					"database temporarily unavailable, circuit breaker will engage after %d consecutive failures",
					m.config.FailureThreshold,
				),
			)
		} else if failures%3 == 0 {
			// Log every 3rd failure at DEBUG level to avoid spam
			m.config.Logger.Debug("Database health check still failing",
				"timestamp", now.Format(time.RFC3339),
				"failure_count", failures,
				"threshold", m.config.FailureThreshold,
			)
		}

		// Circuit breaker: mark unhealthy after threshold reached
		if failures >= m.config.FailureThreshold && wasHealthy {
			m.config.Logger.Error("Database circuit breaker engaged (state: healthy -> unhealthy)",
				"timestamp", now.Format(time.RFC3339),
				"consecutive_failures", failures,
				"threshold", m.config.FailureThreshold,
				"impact", "Database marked as unhealthy, fallback to master_key auth",
				"recovery", fmt.Sprintf("circuit breaker will retry every %s", m.config.CheckInterval),
			)
			m.healthChecker.SetHealthy(false)
		}
	}

	// Update last check time
	m.mu.Lock()
	m.lastCheckTime = now
	m.mu.Unlock()

	// Log detailed status at DEBUG level
	if m.config.Logger.Enabled(context.Background(), slog.LevelDebug) {
		m.logDetailedStatus(now)
	}
}

// logDetailedStatus logs detailed health check statistics.
func (m *Monitor) logDetailedStatus(now time.Time) {
	connStats := m.dbManager.ConnectionStats()
	var acquiredConns, availableConns int32
	if connStats != nil {
		acquiredConns = connStats.AcquiredConns()
		availableConns = connStats.IdleConns()
	}

	currentFailures := atomic.LoadInt32(&m.consecutiveFailures)
	isHealthy := m.healthChecker.IsHealthy()

	m.config.Logger.Debug("Database health check result",
		"timestamp", now.Format(time.RFC3339),
		"is_healthy", isHealthy,
		"consecutive_failures", currentFailures,
		"acquired_conns", acquiredConns,
		"available_conns", availableConns,
	)
}

// Stats returns current health monitor statistics.
func (m *Monitor) Stats() MonitorStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	failures := atomic.LoadInt32(&m.consecutiveFailures)
	isHealthy := m.healthChecker.IsHealthy()

	connStats := m.dbManager.ConnectionStats()
	var acquiredConns, availableConns int32
	if connStats != nil {
		acquiredConns = connStats.AcquiredConns()
		availableConns = connStats.IdleConns()
	}

	return MonitorStats{
		LastCheckTime:       m.lastCheckTime,
		ConsecutiveFailures: failures,
		IsHealthy:           isHealthy,
		AcquiredConns:       acquiredConns,
		AvailableConns:      availableConns,
	}
}
