package models

import (
	"errors"
	"log/slog"
	"time"
)

// ==================== Errors ====================

var (
	// ErrModuleDisabled is returned when the module is disabled
	ErrModuleDisabled = errors.New("litellmdb: module disabled")

	// ErrTokenNotFound is returned when token doesn't exist in database
	ErrTokenNotFound = errors.New("litellmdb: token not found")

	// ErrTokenBlocked is returned when token is blocked
	ErrTokenBlocked = errors.New("litellmdb: token blocked")

	// ErrTokenExpired is returned when token has expired
	ErrTokenExpired = errors.New("litellmdb: token expired")

	// ErrBudgetExceeded is returned when spend >= max_budget
	ErrBudgetExceeded = errors.New("litellmdb: budget exceeded")

	// ErrModelNotAllowed is returned when model is not in allowed list
	ErrModelNotAllowed = errors.New("litellmdb: model not allowed")

	// ErrConnectionFailed is returned when database is unavailable
	ErrConnectionFailed = errors.New("litellmdb: connection failed")

	// ErrQueueFull is returned when spend log queue is full and timeout reached
	ErrQueueFull = errors.New("litellmdb: spend log queue full - timeout reached")
)

// ==================== Config ====================

// Config holds configuration for the litellmdb module
type Config struct {
	// Connection
	DatabaseURL string // postgresql://user:pass@host:5432/db
	MaxConns    int32  // Max connections in pool (default: 10)
	MinConns    int32  // Min connections in pool (default: 2)

	// Health check
	HealthCheckInterval time.Duration // Health check interval (default: 10s)
	ConnectTimeout      time.Duration // Connection timeout (default: 5s)

	// Auth cache
	AuthCacheTTL  time.Duration // Token cache TTL (default: 60s)
	AuthCacheSize int           // LRU cache size (default: 10000)

	// Spend logging
	LogQueueSize     int           // Queue buffer size (default: 10000)
	LogBatchSize     int           // Batch size for INSERT (default: 100)
	LogFlushInterval time.Duration // Flush interval (default: 5s)
	LogRetryAttempts int           // Retry attempts on error (default: 3)
	LogRetryDelay    time.Duration // Delay between retries (default: 1s)

	// Logger
	Logger *slog.Logger
}

// DefaultConfig returns configuration with default values
func DefaultConfig() *Config {
	return &Config{
		MaxConns:            10,
		MinConns:            2,
		HealthCheckInterval: 10 * time.Second,
		ConnectTimeout:      5 * time.Second,
		AuthCacheTTL:        60 * time.Second,
		AuthCacheSize:       10000,
		LogQueueSize:        10000,
		LogBatchSize:        100,
		LogFlushInterval:    5 * time.Second,
		LogRetryAttempts:    3,
		LogRetryDelay:       1 * time.Second,
	}
}

// ApplyDefaults applies default values to zero fields
func (c *Config) ApplyDefaults() {
	defaults := DefaultConfig()

	if c.MaxConns == 0 {
		c.MaxConns = defaults.MaxConns
	}
	if c.MinConns == 0 {
		c.MinConns = defaults.MinConns
	}
	if c.MinConns > c.MaxConns {
		c.MinConns = c.MaxConns
	}
	if c.HealthCheckInterval == 0 {
		c.HealthCheckInterval = defaults.HealthCheckInterval
	}
	if c.ConnectTimeout == 0 {
		c.ConnectTimeout = defaults.ConnectTimeout
	}
	if c.AuthCacheTTL == 0 {
		c.AuthCacheTTL = defaults.AuthCacheTTL
	}
	if c.AuthCacheSize == 0 {
		c.AuthCacheSize = defaults.AuthCacheSize
	}
	if c.LogQueueSize == 0 {
		c.LogQueueSize = defaults.LogQueueSize
	}
	if c.LogBatchSize == 0 {
		c.LogBatchSize = defaults.LogBatchSize
	}
	if c.LogFlushInterval == 0 {
		c.LogFlushInterval = defaults.LogFlushInterval
	}
	if c.LogRetryAttempts == 0 {
		c.LogRetryAttempts = defaults.LogRetryAttempts
	}
	if c.LogRetryDelay == 0 {
		c.LogRetryDelay = defaults.LogRetryDelay
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// Validate checks configuration validity
func (c *Config) Validate() error {
	if c.DatabaseURL == "" {
		return errors.New("litellmdb: database_url is required")
	}
	return nil
}

// ==================== TokenInfo ====================

// TokenInfo holds information about a validated token from LiteLLM_VerificationToken
// Includes full budget hierarchy: Token → User → Team → Org → Memberships
type TokenInfo struct {
	// ==================== Token Level (embedded budget) ====================
	// Identification
	Token    string // sha256 hash of token (PRIMARY KEY)
	KeyName  string // Key name (optional)
	KeyAlias string // Key alias (optional) - user-friendly name

	// Owner references
	UserID         string // User ID (optional)
	TeamID         string // Team ID (optional)
	OrganizationID string // Organization ID (optional, resolved from token or team)

	// Token budget (embedded)
	Spend     float64  // Current spend
	MaxBudget *float64 // Max budget (nil = unlimited)
	TPMLimit  *int64   // Tokens per minute limit
	RPMLimit  *int64   // Requests per minute limit

	// Expiration
	Expires *time.Time // Expiration date (nil = no expiration)

	// Access control
	Models  []string // Allowed models (empty = all)
	Blocked bool     // Is token blocked

	// ==================== User Level (embedded budget) ====================
	UserAlias     string   // User alias (optional) - user-friendly name
	UserEmail     string   // User email (optional)
	UserMaxBudget *float64 // User's personal max budget (nil = unlimited)
	UserSpend     *float64 // User's current spend

	// ==================== Team Level (embedded budget) ====================
	TeamAlias     string   // Team alias (optional) - user-friendly name
	TeamMaxBudget *float64 // Team's max budget (nil = unlimited)
	TeamSpend     *float64 // Team's current spend
	TeamBlocked   *bool    // Team is blocked
	TeamTPMLimit  *int64   // Team's TPM limit
	TeamRPMLimit  *int64   // Team's RPM limit

	// ==================== Organization Level (external budget) ====================
	OrgSpend     *float64 // Organization's current spend
	OrgMaxBudget *float64 // Organization's max budget from BudgetTable (nil = unlimited)
	OrgTPMLimit  *int64   // Organization's TPM limit from BudgetTable
	OrgRPMLimit  *int64   // Organization's RPM limit from BudgetTable

	// ==================== TeamMembership Level (external budget) ====================
	TeamMemberSpend     *float64 // Team member's spend within team
	TeamMemberMaxBudget *float64 // Team member's max budget from BudgetTable (nil = unlimited)
	TeamMemberTPMLimit  *int64   // Team member's TPM limit from BudgetTable
	TeamMemberRPMLimit  *int64   // Team member's RPM limit from BudgetTable

	// ==================== OrganizationMembership Level (external budget) ====================
	OrgMemberSpend     *float64 // Org member's spend within organization
	OrgMemberMaxBudget *float64 // Org member's max budget from BudgetTable (nil = unlimited)
	OrgMemberTPMLimit  *int64   // Org member's TPM limit from BudgetTable
	OrgMemberRPMLimit  *int64   // Org member's RPM limit from BudgetTable

	// Metadata
	Metadata map[string]interface{}
}

// IsExpired checks if token has expired
func (t *TokenInfo) IsExpired() bool {
	if t.Expires == nil {
		return false
	}
	return time.Now().UTC().After(*t.Expires)
}

// IsBudgetExceeded checks if token budget is exceeded (embedded, use >)
func (t *TokenInfo) IsBudgetExceeded() bool {
	if t.MaxBudget == nil {
		return false
	}
	return t.Spend > *t.MaxBudget
}

// IsModelAllowed checks if model is in allowed list
func (t *TokenInfo) IsModelAllowed(model string) bool {
	// Empty list = all models allowed
	if len(t.Models) == 0 {
		return true
	}
	for _, m := range t.Models {
		if m == model {
			return true
		}
	}
	return false
}

// checkUserBudget checks user budget (personal key only - embedded, use >)
func (t *TokenInfo) checkUserBudget() bool {
	// Only check user budget for personal keys (no team)
	if t.TeamID != "" {
		return false
	}
	if t.UserMaxBudget == nil || t.UserSpend == nil {
		return false
	}
	return *t.UserSpend > *t.UserMaxBudget
}

// checkTeamBudget checks team budget (embedded, use >)
func (t *TokenInfo) checkTeamBudget() bool {
	if t.TeamMaxBudget == nil || t.TeamSpend == nil {
		return false
	}
	return *t.TeamSpend > *t.TeamMaxBudget
}

// checkTeamMemberBudget checks team member budget (external, use >=)
func (t *TokenInfo) checkTeamMemberBudget() bool {
	if t.TeamMemberMaxBudget == nil || t.TeamMemberSpend == nil {
		return false
	}
	return *t.TeamMemberSpend >= *t.TeamMemberMaxBudget
}

// checkOrganizationBudget checks organization budget (external, use >=)
func (t *TokenInfo) checkOrganizationBudget() bool {
	if t.OrgMaxBudget == nil || *t.OrgMaxBudget <= 0 || t.OrgSpend == nil {
		return false
	}
	return *t.OrgSpend >= *t.OrgMaxBudget
}

// checkOrganizationMemberBudget checks org member budget (external, use >=)
func (t *TokenInfo) checkOrganizationMemberBudget() bool {
	if t.OrgMemberMaxBudget == nil || t.OrgMemberSpend == nil {
		return false
	}
	return *t.OrgMemberSpend >= *t.OrgMemberMaxBudget
}

// Validate checks token validity for a request with full budget hierarchy
// Order of checks (stops on first failure):
// 1. Token blocked/expired
// 2. Token budget
// 3. Team budget
// 4. Team member budget
// 5. Organization budget
// 6. User budget (personal key only)
// 7. Organization member budget
// 8. Model allowed
func (t *TokenInfo) Validate(model string) error {
	// Check basic validity
	if t.Blocked {
		return ErrTokenBlocked
	}
	if t.IsExpired() {
		return ErrTokenExpired
	}

	// Check budget hierarchy (embedded first, then external)
	if t.IsBudgetExceeded() {
		return ErrBudgetExceeded
	}
	if t.checkTeamBudget() {
		return ErrBudgetExceeded
	}
	if t.checkTeamMemberBudget() {
		return ErrBudgetExceeded
	}
	if t.checkOrganizationBudget() {
		return ErrBudgetExceeded
	}
	if t.checkUserBudget() {
		return ErrBudgetExceeded
	}
	if t.checkOrganizationMemberBudget() {
		return ErrBudgetExceeded
	}

	// Check model access
	if model != "" && !t.IsModelAllowed(model) {
		return ErrModelNotAllowed
	}

	return nil
}

// ==================== SpendLogEntry ====================

// SpendLogEntry represents a row for LiteLLM_SpendLogs table
type SpendLogEntry struct {
	// Request identification
	RequestID string    // UUID (PRIMARY KEY)
	StartTime time.Time // Request start time
	EndTime   time.Time // Request end time

	// API info
	CallType string // Path: "/v1/chat/completions", "/v1/embeddings", etc.
	APIBase  string // Base URL (our gateway)

	// Model
	Model      string // Model name
	ModelID    string // Model ID in proxy (credential.name:model_name format)
	ModelGroup string // Model group (public model_name / model_group)

	// LLM Provider
	CustomLLMProvider string // Provider type: openai, vertex-ai, anthropic, proxy

	// Session tracking
	SessionID string // Session ID from request metadata (chat_id, litellm_session_id, session_id, or request_id)

	// Tokens
	PromptTokens     int // Input tokens
	CompletionTokens int // Output tokens
	TotalTokens      int // Total tokens

	Metadata string // Metadata dict

	// Cost
	Spend float64 // Request cost in USD

	// User identification
	APIKey         string // sha256 hash of token
	UserID         string // User ID
	TeamID         string // Team ID
	OrganizationID string // Organization ID
	EndUser        string // End user ID (from metadata)
	AgentID        string // Agent ID (if called via agent)

	// MCP & Tags
	MCPNamespacedToolName string // MCP tool name with namespace
	RequestTags           string // JSON array of request tags

	// Status
	Status string // "success" | "failure"

	// IP address
	RequesterIP string
}

// ==================== Stats ====================

// AuthCacheStats holds auth cache statistics
type AuthCacheStats struct {
	Size    int     // Current cache size
	Hits    uint64  // Cache hits
	Misses  uint64  // Cache misses
	HitRate float64 // Hit rate percentage
}

// SpendLoggerStats holds spend logger statistics
type SpendLoggerStats struct {
	QueueLen            int       // Current queue length
	QueueCap            int       // Queue capacity
	Queued              uint64    // Total queued
	Written             uint64    // Successfully written
	Dropped             uint64    // Dropped (queue full - timeout reached)
	Errors              uint64    // Write errors
	BatchesOK           uint64    // Successful batches
	QueueFullCount      uint64    // Queue full events (timeouts)
	AggregationCount    uint64    // Completed aggregations
	AggregationErrors   uint64    // Aggregation errors
	LastAggregationTime time.Time // Last successful aggregation
}
