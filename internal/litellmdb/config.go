package litellmdb

// This file re-exports configuration types from the models package
// for backwards compatibility

import (
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	sec "github.com/mixaill76/auto_ai_router/internal/security"
)

// Errors exported from models
var (
	ErrModuleDisabled   = models.ErrModuleDisabled
	ErrTokenNotFound    = models.ErrTokenNotFound
	ErrTokenBlocked     = models.ErrTokenBlocked
	ErrTokenExpired     = models.ErrTokenExpired
	ErrBudgetExceeded   = models.ErrBudgetExceeded
	ErrModelNotAllowed  = models.ErrModelNotAllowed
	ErrConnectionFailed = models.ErrConnectionFailed
)

// Config type alias for backwards compatibility
type Config = models.Config

// TokenInfo type alias for backwards compatibility
type TokenInfo = models.TokenInfo

// SpendLogEntry type alias for backwards compatibility
type SpendLogEntry = models.SpendLogEntry

// AuthCacheStats type alias for backwards compatibility
type AuthCacheStats = models.AuthCacheStats

// SpendLoggerStats type alias for backwards compatibility
type SpendLoggerStats = models.SpendLoggerStats

// DefaultConfig returns configuration with default values
func DefaultConfig() *Config {
	return models.DefaultConfig()
}

// maskDatabaseURL is deprecated - use security.MaskDatabaseURL instead
// Kept for backwards compatibility
func maskDatabaseURL(url string) string {
	return sec.MaskDatabaseURL(url)
}
