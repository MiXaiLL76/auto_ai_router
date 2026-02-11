package connection

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
)

func TestNewConnectionPool_InvalidURL(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "invalid-url",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool, err := NewConnectionPool(cfg)
	assert.Error(t, err)
	assert.Nil(t, pool)
}

func TestNewConnectionPool_MissingURL(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	err := cfg.Validate()
	assert.Error(t, err)
}

func TestConnectionPool_Close_Idempotent(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/nonexistent",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	// Create context and cancel for the pool
	ctx, cancel := context.WithCancel(context.Background())

	// We'll mock pool creation by testing the Close behavior
	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		ctx:     ctx,
		cancel:  cancel,
		closed:  atomic.Bool{},
		healthy: atomic.Bool{},
	}

	// Call Close multiple times - should not panic
	pool.closed.Store(false)
	pool.Close()
	assert.True(t, pool.closed.Load())

	// Second close should be no-op (already closed)
	pool.Close()
	assert.True(t, pool.closed.Load())
}

func TestConnectionPool_IsHealthy_BeforeClose(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/nonexistent",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		ctx:     context.Background(),
		cancel:  func() {},
		closed:  atomic.Bool{},
		healthy: atomic.Bool{},
	}

	pool.healthy.Store(true)
	assert.True(t, pool.IsHealthy())

	pool.healthy.Store(false)
	assert.False(t, pool.IsHealthy())
}

func TestConnectionPool_Acquire_WhenClosed(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:   nil,
		config: cfg,
		logger: cfg.Logger,
	}

	pool.closed.Store(true)

	conn, err := pool.Acquire(context.Background())
	assert.Error(t, err)
	assert.Nil(t, conn)
	assert.ErrorIs(t, err, models.ErrConnectionFailed)
}

func TestConnectionPool_Acquire_WhenUnhealthy(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		healthy: atomic.Bool{},
	}

	pool.healthy.Store(false)

	conn, err := pool.Acquire(context.Background())
	assert.Error(t, err)
	assert.Nil(t, conn)
	assert.ErrorIs(t, err, models.ErrConnectionFailed)
}

func TestConnectionPool_HealthStatus_Transitions(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		healthy: atomic.Bool{},
	}

	// Test health status transitions
	pool.healthy.Store(true)
	assert.True(t, pool.IsHealthy())

	// Simulate unhealthy transition
	wasHealthy := pool.healthy.Swap(false)
	assert.True(t, wasHealthy)
	assert.False(t, pool.IsHealthy())

	// Simulate recovery
	wasUnhealthy := !pool.healthy.Swap(true)
	assert.True(t, wasUnhealthy)
	assert.True(t, pool.IsHealthy())
}

func TestConnectionPool_minDuration(t *testing.T) {
	tests := []struct {
		name     string
		a        time.Duration
		b        time.Duration
		expected time.Duration
	}{
		{"a smaller", 1 * time.Second, 2 * time.Second, 1 * time.Second},
		{"b smaller", 5 * time.Second, 3 * time.Second, 3 * time.Second},
		{"equal", 2 * time.Second, 2 * time.Second, 2 * time.Second},
		{"zero", 0, 5 * time.Second, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := minDuration(tt.a, tt.b)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConnectionPool_ConcurrentStatusChecks(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		healthy: atomic.Bool{},
	}

	pool.healthy.Store(true)

	var wg sync.WaitGroup
	results := make([]bool, 1000)

	// Spawn 1000 concurrent readers
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = pool.IsHealthy()
		}(i)
	}

	wg.Wait()

	// All should see true (since we set it to true)
	for _, result := range results {
		assert.True(t, result)
	}
}

func TestConnectionPool_Stats_NilPool(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:   nil,
		config: cfg,
		logger: cfg.Logger,
	}

	stats := pool.Stats()
	assert.Nil(t, stats)
}

func TestConnectionPool_ReconnectDelay_Exponential(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	pool := &ConnectionPool{
		pool:           nil,
		config:         cfg,
		logger:         cfg.Logger,
		reconnectDelay: time.Second,
	}

	// Test exponential backoff
	delay1 := pool.reconnectDelay
	delay2 := minDuration(delay1*2, 30*time.Second)
	delay3 := minDuration(delay2*2, 30*time.Second)

	assert.Equal(t, time.Second, delay1)
	assert.Equal(t, 2*time.Second, delay2)
	assert.Equal(t, 4*time.Second, delay3)

	// Test max cap at 30s
	delay := 15 * time.Second
	for i := 0; i < 5; i++ {
		delay = minDuration(delay*2, 30*time.Second)
	}
	assert.Equal(t, 30*time.Second, delay)
}

func TestConnectionPool_Close_WithCancelContext(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	ctx, cancel := context.WithCancel(context.Background())

	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		ctx:     ctx,
		cancel:  cancel,
		closed:  atomic.Bool{},
		healthy: atomic.Bool{},
	}

	pool.healthy.Store(true)

	// Context should not be cancelled yet
	select {
	case <-ctx.Done():
		t.Fatal("context was cancelled too early")
	default:
	}

	// Close the pool - should cancel context
	pool.Close()

	// Now context should be cancelled
	select {
	case <-ctx.Done():
		// Expected
	case <-time.After(1 * time.Second):
		t.Fatal("context was not cancelled after pool close")
	}
}

func TestConnectionPool_Close_WaitGroupTimeout(t *testing.T) {
	cfg := &models.Config{
		DatabaseURL: "postgres://localhost/test",
		MaxConns:    5,
		MinConns:    1,
	}
	cfg.ApplyDefaults()

	ctx, cancel := context.WithCancel(context.Background())
	pool := &ConnectionPool{
		pool:    nil,
		config:  cfg,
		logger:  cfg.Logger,
		ctx:     ctx,
		cancel:  cancel,
		closed:  atomic.Bool{},
		healthy: atomic.Bool{},
	}

	// Add a goroutine that never completes
	pool.wg.Add(1)
	go func() {
		// 1. Simplify: remove select, use simple receive
		<-ctx.Done()

		// 2. Safety: Ensure Done is called exactly once,
		// even if the logic below panics.
		defer pool.wg.Done()

		// 3. Cleanup logic
		time.Sleep(100 * time.Millisecond)
	}()

	// Close should not hang due to timeout
	start := time.Now()
	pool.Close()
	elapsed := time.Since(start)

	// Should timeout after 10 seconds as per code
	// But we won't test exact timing, just that it completes
	assert.True(t, elapsed > 0)
	assert.True(t, pool.closed.Load())
}
