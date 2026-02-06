package litellmdb

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/auth"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/spendlog"
)

// Manager is the main interface for the litellmdb module
type Manager interface {
	// Auth - synchronous authentication
	ValidateToken(ctx context.Context, rawToken string) (*models.TokenInfo, error)
	ValidateTokenForModel(ctx context.Context, rawToken, model string) (*models.TokenInfo, error)

	// Logging - asynchronous logging
	LogSpend(entry *models.SpendLogEntry) error

	// Status
	IsEnabled() bool
	IsHealthy() bool

	// Stats
	AuthCacheStats() models.AuthCacheStats
	SpendLoggerStats() models.SpendLoggerStats
	ConnectionStats() *pgxpool.Stat

	// Lifecycle
	Shutdown(ctx context.Context) error
}

// ==================== NoopManager ====================

// NoopManager is a no-op implementation when module is disabled
type NoopManager struct{}

// NewNoopManager creates a new no-op manager
func NewNoopManager() *NoopManager {
	return &NoopManager{}
}

func (n *NoopManager) ValidateToken(ctx context.Context, rawToken string) (*models.TokenInfo, error) {
	return nil, models.ErrModuleDisabled
}

func (n *NoopManager) ValidateTokenForModel(ctx context.Context, rawToken, model string) (*models.TokenInfo, error) {
	return nil, models.ErrModuleDisabled
}

func (n *NoopManager) LogSpend(entry *models.SpendLogEntry) error {
	// no-op
	return nil
}

func (n *NoopManager) IsEnabled() bool {
	return false
}

func (n *NoopManager) IsHealthy() bool {
	return false
}

func (n *NoopManager) AuthCacheStats() models.AuthCacheStats {
	return models.AuthCacheStats{}
}

func (n *NoopManager) SpendLoggerStats() models.SpendLoggerStats {
	return models.SpendLoggerStats{}
}

func (n *NoopManager) ConnectionStats() *pgxpool.Stat {
	return nil
}

func (n *NoopManager) Shutdown(ctx context.Context) error {
	return nil
}

// ==================== DefaultManager ====================

// DefaultManager is the real implementation of Manager
type DefaultManager struct {
	pool        *connection.ConnectionPool
	auth        *auth.Authenticator
	spendLogger *spendlog.Logger
	config      *models.Config
	logger      *slog.Logger
}

// New creates a new Manager instance
// Returns error if database connection fails
func New(cfg *models.Config) (Manager, error) {
	cfg.ApplyDefaults()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	// Create connection pool
	pool, err := connection.NewConnectionPool(cfg)
	if err != nil {
		return nil, err
	}

	// Create auth cache
	cache, err := auth.NewCache(cfg.AuthCacheSize, cfg.AuthCacheTTL)
	if err != nil {
		pool.Close()
		return nil, err
	}

	// Create authenticator
	authenticator := auth.NewAuthenticator(pool, cache, cfg.Logger)

	// Create spend logger
	logger := spendlog.NewLogger(pool, cfg)
	logger.Start()

	m := &DefaultManager{
		pool:        pool,
		auth:        authenticator,
		spendLogger: logger,
		config:      cfg,
		logger:      cfg.Logger,
	}

	cfg.Logger.Info("LiteLLM DB Manager initialized",
		"database", maskDatabaseURL(cfg.DatabaseURL),
		"max_conns", cfg.MaxConns,
		"auth_cache_size", cfg.AuthCacheSize,
		"log_queue_size", cfg.LogQueueSize,
	)

	return m, nil
}

// ValidateToken validates a token
func (m *DefaultManager) ValidateToken(ctx context.Context, rawToken string) (*models.TokenInfo, error) {
	return m.auth.ValidateToken(ctx, rawToken)
}

// ValidateTokenForModel validates a token with model access check
func (m *DefaultManager) ValidateTokenForModel(ctx context.Context, rawToken, model string) (*models.TokenInfo, error) {
	return m.auth.ValidateTokenForModel(ctx, rawToken, model)
}

// LogSpend adds an entry to the logging queue
func (m *DefaultManager) LogSpend(entry *models.SpendLogEntry) error {
	return m.spendLogger.Log(entry)
}

// IsEnabled returns true (module is enabled)
func (m *DefaultManager) IsEnabled() bool {
	return true
}

// IsHealthy returns database connection health status
func (m *DefaultManager) IsHealthy() bool {
	return m.pool.IsHealthy()
}

// AuthCacheStats returns auth cache statistics
func (m *DefaultManager) AuthCacheStats() models.AuthCacheStats {
	return m.auth.CacheStats()
}

// SpendLoggerStats returns spend logger statistics
func (m *DefaultManager) SpendLoggerStats() models.SpendLoggerStats {
	return m.spendLogger.Stats()
}

// ConnectionStats returns connection pool statistics
func (m *DefaultManager) ConnectionStats() *pgxpool.Stat {
	return m.pool.Stats()
}

// Shutdown stops all components
func (m *DefaultManager) Shutdown(ctx context.Context) error {
	m.logger.Info("Shutting down LiteLLM DB Manager...")

	// First stop spend logger (to write all pending logs)
	if err := m.spendLogger.Shutdown(ctx); err != nil {
		m.logger.Error("SpendLogger shutdown error", "error", err)
	}

	// Close connection pool
	m.pool.Close()

	m.logger.Info("LiteLLM DB Manager shutdown complete")
	return nil
}

// ==================== Compile-time interface check ====================

var _ Manager = (*DefaultManager)(nil)
var _ Manager = (*NoopManager)(nil)
