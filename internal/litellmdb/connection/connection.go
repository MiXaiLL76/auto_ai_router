package connection

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/security"
)

// ConnectionPool manages PostgreSQL connections with auto-reconnect
type ConnectionPool struct {
	pool   *pgxpool.Pool
	config *models.Config
	logger *slog.Logger

	// Health status
	healthy atomic.Bool

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	closed atomic.Bool

	// Reconnection
	reconnectMu    sync.Mutex
	lastReconnect  time.Time
	reconnectDelay time.Duration
}

// NewConnectionPool creates a new connection pool
func NewConnectionPool(cfg *models.Config) (*ConnectionPool, error) {
	cfg.ApplyDefaults()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())

	cp := &ConnectionPool{
		config:         cfg,
		logger:         cfg.Logger,
		ctx:            ctx,
		cancel:         cancel,
		reconnectDelay: time.Second, // Initial delay 1s
	}

	// Create pool config
	poolConfig, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("litellmdb: invalid database URL: %w", err)
	}

	// Configure pool
	poolConfig.MaxConns = cfg.MaxConns
	poolConfig.MinConns = cfg.MinConns
	poolConfig.HealthCheckPeriod = cfg.HealthCheckInterval
	poolConfig.ConnConfig.ConnectTimeout = cfg.ConnectTimeout

	// Notice callback
	poolConfig.ConnConfig.OnNotice = func(c *pgconn.PgConn, n *pgconn.Notice) {
		cp.logger.Debug("PostgreSQL notice",
			"severity", n.Severity,
			"message", n.Message,
		)
	}

	// Connect
	connectCtx, connectCancel := context.WithTimeout(ctx, cfg.ConnectTimeout)
	defer connectCancel()

	pool, err := pgxpool.NewWithConfig(connectCtx, poolConfig)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("litellmdb: failed to connect: %w", err)
	}

	// Verify connection
	if err := pool.Ping(connectCtx); err != nil {
		pool.Close()
		cancel()
		return nil, fmt.Errorf("litellmdb: ping failed: %w", err)
	}

	cp.pool = pool
	cp.healthy.Store(true)

	// Start background health check
	cp.wg.Add(1)
	go cp.healthCheckLoop()

	cp.logger.Info("LiteLLM DB connection pool initialized",
		"max_conns", cfg.MaxConns,
		"min_conns", cfg.MinConns,
		"database", security.MaskDatabaseURL(cfg.DatabaseURL),
	)

	return cp, nil
}

// Acquire gets a connection from the pool
func (cp *ConnectionPool) Acquire(ctx context.Context) (*pgxpool.Conn, error) {
	if cp.closed.Load() {
		return nil, models.ErrConnectionFailed
	}
	if !cp.healthy.Load() {
		return nil, models.ErrConnectionFailed
	}
	return cp.pool.Acquire(ctx)
}

// Pool returns the underlying pgxpool.Pool
func (cp *ConnectionPool) Pool() *pgxpool.Pool {
	return cp.pool
}

// IsHealthy returns connection health status
func (cp *ConnectionPool) IsHealthy() bool {
	return cp.healthy.Load()
}

// Stats returns pool statistics
func (cp *ConnectionPool) Stats() *pgxpool.Stat {
	if cp.pool == nil {
		return nil
	}
	return cp.pool.Stat()
}

// Close closes the connection pool
func (cp *ConnectionPool) Close() {
	if !cp.closed.CompareAndSwap(false, true) {
		return // Already closed
	}

	// Stop background goroutines
	cp.cancel()

	// Wait for goroutines to finish with timeout
	doneChan := make(chan struct{})
	go func() {
		cp.wg.Wait()
		close(doneChan)
	}()

	select {
	case <-doneChan:
		// Goroutine exited cleanly
	case <-time.After(10 * time.Second):
		// Timeout exceeded - goroutine didn't stop
		cp.logger.Warn("Health check goroutine did not stop within timeout")
	}

	// Close pool
	if cp.pool != nil {
		cp.pool.Close()
	}

	cp.logger.Info("LiteLLM DB connection pool closed")
}

// healthCheckLoop periodically checks connection health
func (cp *ConnectionPool) healthCheckLoop() {
	defer cp.wg.Done()

	ticker := time.NewTicker(cp.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-cp.ctx.Done():
			return
		case <-ticker.C:
			cp.performHealthCheck()
		}
	}
}

// performHealthCheck executes a health check
func (cp *ConnectionPool) performHealthCheck() {
	ctx, cancel := context.WithTimeout(cp.ctx, 5*time.Second)
	defer cancel()

	var result int
	err := cp.pool.QueryRow(ctx, queries.QueryHealthCheck).Scan(&result)

	if err != nil {
		wasHealthy := cp.healthy.Swap(false)
		if wasHealthy {
			cp.logger.Error("LiteLLM DB health check failed",
				"error", err,
			)
		}
		// Try to reconnect
		cp.tryReconnect()
	} else {
		wasUnhealthy := !cp.healthy.Swap(true)
		if wasUnhealthy {
			cp.logger.Info("LiteLLM DB connection restored")
			cp.reconnectDelay = time.Second // Reset backoff
		}
	}
}

// tryReconnect attempts to restore connection with exponential backoff
func (cp *ConnectionPool) tryReconnect() {
	cp.reconnectMu.Lock()
	defer cp.reconnectMu.Unlock()

	// Don't reconnect too frequently
	if time.Since(cp.lastReconnect) < cp.reconnectDelay {
		return
	}

	cp.logger.Info("Attempting to reconnect to LiteLLM DB",
		"delay", cp.reconnectDelay,
	)

	ctx, cancel := context.WithTimeout(cp.ctx, cp.config.ConnectTimeout)
	defer cancel()

	err := cp.pool.Ping(ctx)
	cp.lastReconnect = time.Now().UTC()

	if err != nil {
		// Increase backoff (max 30s)
		cp.reconnectDelay = minDuration(cp.reconnectDelay*2, 30*time.Second)
		cp.logger.Error("Reconnection failed",
			"error", err,
			"next_delay", cp.reconnectDelay,
		)
	} else {
		cp.healthy.Store(true)
		cp.reconnectDelay = time.Second
		cp.logger.Info("Reconnection successful")
	}
}

// minDuration returns the minimum of two durations
func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}
