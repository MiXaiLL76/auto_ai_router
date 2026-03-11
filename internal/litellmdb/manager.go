package litellmdb

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/auth"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	modeltable "github.com/mixaill76/auto_ai_router/internal/litellmdb/model_table"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/spendlog"
	imodels "github.com/mixaill76/auto_ai_router/internal/models"
)

// Manager is the main interface for the litellmdb module
type Manager interface {
	FetchMasterKey(ctx context.Context, default_key string) error

	// Auth - synchronous authentication
	ValidateToken(ctx context.Context, rawToken string) (*models.TokenInfo, error)
	ValidateTokenForModel(ctx context.Context, rawToken, model string) (*models.TokenInfo, error)

	// Logging - asynchronous logging
	LogSpend(entry *models.SpendLogEntry) error

	// Model table - fetch credentials/models/prices from LiteLLM DB for AIR
	FetchModelsForAIR(ctx context.Context, signingKey string) ([]config.CredentialConfig, []config.ModelRPMConfig, map[string]*imodels.ModelPrice, error)

	// Status
	IsEnabled() bool
	IsHealthy() bool

	// Stats
	AuthCacheStats() models.AuthCacheStats
	SpendLoggerStats() models.SpendLoggerStats
	ConnectionStats() *pgxpool.Stat

	// Pool access (for login queries)
	GetPool() *pgxpool.Pool

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

// FetchMasterKey validates a token
func (m *NoopManager) FetchMasterKey(ctx context.Context, default_key string) error {
	return nil
}

func (n *NoopManager) FetchModelsForAIR(_ context.Context, _ string) ([]config.CredentialConfig, []config.ModelRPMConfig, map[string]*imodels.ModelPrice, error) {
	return nil, nil, nil, nil
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

func (n *NoopManager) GetPool() *pgxpool.Pool {
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
	modelTable  *modeltable.ProxyModelTable
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

	// Ensure pool is cleaned up if any subsequent initialization fails
	defer func() {
		if err != nil && pool != nil {
			pool.Close()
		}
	}()

	// Create auth cache
	cache, err := auth.NewCache(cfg.AuthCacheSize, cfg.AuthCacheTTL)
	if err != nil {
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
		modelTable:  modeltable.NewProxyModelTable(pool, cfg.Logger),
		config:      cfg,
		logger:      cfg.Logger,
	}

	// Clear error so defer doesn't close pool
	err = nil

	cfg.Logger.Info("LiteLLM DB Manager initialized",
		"database", maskDatabaseURL(cfg.DatabaseURL),
		"max_conns", cfg.MaxConns,
		"auth_cache_size", cfg.AuthCacheSize,
		"log_queue_size", cfg.LogQueueSize,
	)

	return m, err
}

// FetchMasterKey validates a token
func (m *DefaultManager) FetchMasterKey(ctx context.Context, default_key string) error {
	return m.auth.FetchMasterKey(ctx, default_key)
}

// FetchModelsForAIR loads credentials, model RPM configs and prices from LiteLLM DB
func (m *DefaultManager) FetchModelsForAIR(ctx context.Context, signingKey string) ([]config.CredentialConfig, []config.ModelRPMConfig, map[string]*imodels.ModelPrice, error) {
	return m.modelTable.FetchModelsForAIR(ctx, signingKey)
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

// GetPool returns the underlying pgxpool.Pool for direct queries.
func (m *DefaultManager) GetPool() *pgxpool.Pool {
	return m.pool.Pool()
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
