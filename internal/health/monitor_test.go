package health

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
)

// MockDBManager implements litellmdb.Manager for testing.
type MockDBManager struct {
	healthy bool
}

func (m *MockDBManager) ValidateToken(ctx context.Context, rawToken string) (*models.TokenInfo, error) {
	return nil, nil
}
func (m *MockDBManager) ValidateTokenForModel(ctx context.Context, rawToken, model string) (*models.TokenInfo, error) {
	return nil, nil
}
func (m *MockDBManager) LogSpend(entry *models.SpendLogEntry) error { return nil }
func (m *MockDBManager) IsEnabled() bool                            { return true }
func (m *MockDBManager) IsHealthy() bool                            { return m.healthy }
func (m *MockDBManager) AuthCacheStats() models.AuthCacheStats      { return models.AuthCacheStats{} }
func (m *MockDBManager) SpendLoggerStats() models.SpendLoggerStats  { return models.SpendLoggerStats{} }
func (m *MockDBManager) ConnectionStats() *pgxpool.Stat             { return nil }
func (m *MockDBManager) GetPool() *pgxpool.Pool                     { return nil }
func (m *MockDBManager) Shutdown(ctx context.Context) error         { return nil }

// Compile-time check
var _ litellmdb.Manager = (*MockDBManager)(nil)

func TestNewMonitor_Defaults(t *testing.T) {
	hc := NewDBHealthChecker()
	db := &MockDBManager{healthy: true}

	// nil config → defaults
	m := NewMonitor(nil, hc, db)
	require.NotNil(t, m)
	assert.Equal(t, 30*time.Second, m.config.CheckInterval)
	assert.Equal(t, int32(3), m.config.FailureThreshold)
}

func TestCheckHealth_HealthyTransition(t *testing.T) {
	hc := NewDBHealthChecker()
	db := &MockDBManager{healthy: false}
	logger := testhelpers.NewTestLogger()

	m := NewMonitor(&MonitorConfig{
		CheckInterval:    time.Second,
		FailureThreshold: 3,
		Logger:           logger,
	}, hc, db)

	// Initially healthy
	assert.True(t, hc.IsHealthy())

	// After 1 failure, still healthy (threshold=3)
	m.checkHealth()
	assert.True(t, hc.IsHealthy(), "should stay healthy after 1 failure (threshold=3)")

	// After 3 failures, unhealthy
	m.checkHealth() // 2nd
	m.checkHealth() // 3rd
	assert.False(t, hc.IsHealthy(), "should be unhealthy after 3 failures")

	// Recovery
	db.healthy = true
	m.checkHealth()
	assert.True(t, hc.IsHealthy(), "should recover when DB is healthy again")
}

func TestCheckHealth_CircuitBreaker(t *testing.T) {
	hc := NewDBHealthChecker()
	db := &MockDBManager{healthy: false}

	m := NewMonitor(&MonitorConfig{
		CheckInterval:    time.Second,
		FailureThreshold: 2,
		Logger:           slog.Default(),
	}, hc, db)

	// 2 failures → circuit breaker engaged
	m.checkHealth()
	m.checkHealth()
	assert.False(t, hc.IsHealthy(), "circuit breaker should engage after threshold")

	// Stats should reflect failures
	stats := m.Stats()
	assert.False(t, stats.IsHealthy)
	assert.Equal(t, int32(2), stats.ConsecutiveFailures)
}
