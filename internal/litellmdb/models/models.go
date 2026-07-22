package models

import (
	"errors"
	"log/slog"
	"regexp"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// ==================== Errors ====================

var (
	// ErrModuleDisabled is returned when the module is disabled
	ErrModuleDisabled = errors.New("litellmdb: module disabled")

	// ErrTokenNotFound is returned when token doesn't exist in database
	ErrTokenNotFound = errors.New("litellmdb: token not found")

	// ErrTokenBlocked is returned when token is blocked
	ErrTokenBlocked = errors.New("litellmdb: token blocked")

	// ErrTeamBlocked is returned when the token's parent team is blocked
	ErrTeamBlocked = errors.New("litellmdb: team blocked")

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
	AuthCacheTTL  time.Duration // Token cache TTL (default: 5s)
	AuthCacheSize int           // LRU cache size (default: 10000)

	// Spend logging
	LogQueueSize        int           // Queue buffer size (default: 10000)
	LogBatchSize        int           // Batch size for INSERT (default: 100)
	LogFlushInterval    time.Duration // Flush interval (default: 5s)
	LogWorkers          int           // Number of parallel queue-draining workers (default: 4)
	DisableSpendLogging bool          // Control-plane managers set this to avoid creating a writer

	// DisableSpendLogsWrite disables writing SpendLogEntry/Daily* aggregates to
	// Postgres while leaving auth (ValidateToken) untouched (default: false).
	DisableSpendLogsWrite bool

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
		AuthCacheTTL:        5 * time.Second,
		AuthCacheSize:       10000,
		LogQueueSize:        10000,
		LogBatchSize:        100,
		LogFlushInterval:    5 * time.Second,
		LogWorkers:          4,
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
	if c.LogWorkers == 0 {
		c.LogWorkers = defaults.LogWorkers
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
	UserID         string   // User ID (optional)
	TeamID         string   // Team ID (optional)
	OrganizationID string   // Organization ID (optional, resolved from token or team)
	Tags           []string // Request tags from token metadata

	// Token budget (embedded)
	Spend     float64  // Current spend
	MaxBudget *float64 // Max budget (nil = unlimited)
	TPMLimit  *int64   // Tokens per minute limit
	RPMLimit  *int64   // Requests per minute limit

	// Expiration
	Expires *time.Time // Expiration date (nil = no expiration)

	// Access control
	Models        []string // Key-level allowed models (empty = all)
	AllowedRoutes []string // LiteLLM virtual-key routes (empty = unrestricted)
	TeamModels    []string // Team-level allowed models (empty = all)
	Blocked       bool     // Is token blocked

	// ==================== User Level (embedded budget) ====================
	UserAlias     string   // User alias (optional) - user-friendly name
	UserEmail     string   // User email (optional)
	UserMaxBudget *float64 // User's personal max budget (nil = unlimited)
	UserSpend     *float64 // User's current spend
	UserTPMLimit  *int64   // User's TPM limit
	UserRPMLimit  *int64   // User's RPM limit

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

// LiteLLM sentinel values stored in key model allowlists.
const (
	AllTeamModels  = "all-team-models"
	AllProxyModels = "all-proxy-models"
)

// ModelAccessScope identifies one independently enforced model allowlist.
// Applicable non-empty scopes form an intersection, matching LiteLLM's
// key -> team/member or personal-user -> project request checks.
type ModelAccessScope struct {
	Name    string
	Models  []string
	DenyAll bool
}

// ModelScopeMatcher evaluates one allowlist. Empty scopes must be treated as
// unrestricted. A custom matcher lets the routing layer apply LiteLLM's
// directional request-alias expansion without coupling DB models to routing.
type ModelScopeMatcher func(model string, allowedModels []string) bool

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneMetadataValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		cloned := make(map[string]interface{}, len(typed))
		for key, child := range typed {
			cloned[key] = cloneMetadataValue(child)
		}
		return cloned
	case []interface{}:
		cloned := make([]interface{}, len(typed))
		for index, child := range typed {
			cloned[index] = cloneMetadataValue(child)
		}
		return cloned
	case []string:
		return append([]string(nil), typed...)
	case map[string]string:
		cloned := make(map[string]string, len(typed))
		for key, child := range typed {
			cloned[key] = child
		}
		return cloned
	default:
		return value
	}
}

// Clone returns a defensive copy suitable for crossing cache or request
// boundaries. Auth data is immutable once cached; callers may mutate their
// copy without changing permissions observed by another request.
func (t *TokenInfo) Clone() *TokenInfo {
	if t == nil {
		return nil
	}
	clone := *t
	clone.Models = append([]string(nil), t.Models...)
	clone.AllowedRoutes = append([]string(nil), t.AllowedRoutes...)
	clone.TeamModels = append([]string(nil), t.TeamModels...)
	clone.Tags = append([]string(nil), t.Tags...)
	if t.Metadata != nil {
		clone.Metadata = cloneMetadataValue(t.Metadata).(map[string]interface{})
	}

	clone.MaxBudget = clonePointer(t.MaxBudget)
	clone.TPMLimit = clonePointer(t.TPMLimit)
	clone.RPMLimit = clonePointer(t.RPMLimit)
	clone.Expires = clonePointer(t.Expires)
	clone.UserMaxBudget = clonePointer(t.UserMaxBudget)
	clone.UserSpend = clonePointer(t.UserSpend)
	clone.UserTPMLimit = clonePointer(t.UserTPMLimit)
	clone.UserRPMLimit = clonePointer(t.UserRPMLimit)
	clone.TeamMaxBudget = clonePointer(t.TeamMaxBudget)
	clone.TeamSpend = clonePointer(t.TeamSpend)
	clone.TeamBlocked = clonePointer(t.TeamBlocked)
	clone.TeamTPMLimit = clonePointer(t.TeamTPMLimit)
	clone.TeamRPMLimit = clonePointer(t.TeamRPMLimit)
	clone.OrgSpend = clonePointer(t.OrgSpend)
	clone.OrgMaxBudget = clonePointer(t.OrgMaxBudget)
	clone.OrgTPMLimit = clonePointer(t.OrgTPMLimit)
	clone.OrgRPMLimit = clonePointer(t.OrgRPMLimit)
	clone.TeamMemberSpend = clonePointer(t.TeamMemberSpend)
	clone.TeamMemberMaxBudget = clonePointer(t.TeamMemberMaxBudget)
	clone.TeamMemberTPMLimit = clonePointer(t.TeamMemberTPMLimit)
	clone.TeamMemberRPMLimit = clonePointer(t.TeamMemberRPMLimit)
	clone.OrgMemberSpend = clonePointer(t.OrgMemberSpend)
	clone.OrgMemberMaxBudget = clonePointer(t.OrgMemberMaxBudget)
	clone.OrgMemberTPMLimit = clonePointer(t.OrgMemberTPMLimit)
	clone.OrgMemberRPMLimit = clonePointer(t.OrgMemberRPMLimit)
	return &clone
}

// IsExpired checks if token has expired
func (t *TokenInfo) IsExpired() bool {
	if t.Expires == nil {
		return false
	}
	return utils.NowUTC().After(*t.Expires)
}

// IsBudgetExceeded checks if token budget is exceeded (embedded, use >)
func (t *TokenInfo) IsBudgetExceeded() bool {
	if t.MaxBudget == nil {
		return false
	}
	return t.Spend > *t.MaxBudget
}

// ModelAccessScopes returns the ordered set of allowlists applicable to this
// token: the key scope plus the team scope for team keys.
func (t *TokenInfo) ModelAccessScopes() []ModelAccessScope {
	keyModels := t.Models
	for _, model := range t.Models {
		if model == AllTeamModels {
			// LiteLLM fails closed when all-team-models is used without a team.
			// With a team it replaces, rather than extends, the key allowlist.
			if t.TeamID != "" {
				keyModels = t.TeamModels
			}
			break
		}
	}

	scopes := []ModelAccessScope{{Name: "key", Models: keyModels}}
	if t.TeamID != "" {
		scopes = append(scopes, ModelAccessScope{Name: "team", Models: t.TeamModels})
	}
	return scopes
}

// exactModelScopeMatch reports whether model passes one allowlist. Besides the
// exact name and the "*" / all-proxy-models wildcards, an entry may be a regex
// (matched anchored, i.e. it must cover the whole model name): one pattern
// such as "anthropic/.*" or ".*claude-haiku.*" then includes every non-public
// naming of the model without enumerating each alias in the allowlist.
func exactModelScopeMatch(model string, allowedModels []string) bool {
	if len(allowedModels) == 0 {
		return true
	}
	for _, allowed := range allowedModels {
		if allowed == model || allowed == "*" || allowed == AllProxyModels {
			return true
		}
		if modelScopePatternMatches(allowed, model) {
			return true
		}
	}
	return false
}

// modelScopePatternMatches treats an allowlist entry containing regex
// metacharacters as an anchored pattern. Literal entries keep exact-match
// semantics, so existing allowlists behave exactly as before.
func modelScopePatternMatches(pattern, model string) bool {
	if !strings.ContainsAny(pattern, `\.^$*+?()[]{}|`) {
		return false
	}
	matched, err := regexp.MatchString("^(?:"+pattern+")$", model)
	return err == nil && matched
}

// IsModelAllowedBy checks every applicable model scope with matcher.
func (t *TokenInfo) IsModelAllowedBy(model string, matcher ModelScopeMatcher) bool {
	if matcher == nil {
		matcher = exactModelScopeMatch
	}
	for _, scope := range t.ModelAccessScopes() {
		if scope.DenyAll || !matcher(model, scope.Models) {
			return false
		}
	}
	return true
}

// IsModelAllowed checks all applicable allowlists using exact IDs.
func (t *TokenInfo) IsModelAllowed(model string) bool {
	return t.IsModelAllowedBy(model, exactModelScopeMatch)
}

// IsAnyModelAllowed reports whether at least one candidate name passes every
// applicable model scope. The same provider model is often exposed under
// several route aliases (e.g. "claude-haiku-4.5" and
// "anthropic/claude-haiku-4.5" for the same credential+model); callers resolve
// the alias-equivalence group (models.Manager.GetAliasesForModel) and pass it
// here so the check isn't defeated by which spelling the client used.
func (t *TokenInfo) IsAnyModelAllowed(candidates []string) bool {
	for _, candidate := range candidates {
		if t.IsModelAllowed(candidate) {
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
// 2. Team blocked
// 3. Token budget
// 4. Team budget
// 5. Team member budget
// 6. Organization budget
// 7. User budget (personal key only)
// 8. Organization member budget
// 9. Model allowed
func (t *TokenInfo) Validate(model string) error {
	// Check basic validity
	if t.Blocked {
		return ErrTokenBlocked
	}
	if t.IsExpired() {
		return ErrTokenExpired
	}
	if t.TeamBlocked != nil && *t.TeamBlocked {
		return ErrTeamBlocked
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
	RequestID           string     // UUID (PRIMARY KEY)
	AirEventID          string     // Runtime collision fallback; never persisted as a standalone column
	StartTime           time.Time  // Request start time
	EndTime             time.Time  // Request end time
	RequestDurationMS   int        // Whole request duration in milliseconds
	CompletionStartTime *time.Time // First completion token timestamp when available

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
	CacheHit string // LiteLLM-compatible cache marker
	CacheKey string // LiteLLM-compatible cache key marker

	// Cost
	Spend float64 // Request cost in USD

	// User identification
	APIKey         string // sha256 hash of token
	UserID         string // User ID
	TeamID         string // Team ID
	OrganizationID string // Organization ID
	EndUser        string // End user ID (from metadata)

	// Status
	Status string // "success" | "failure"

	// IP address
	RequesterIP string

	// Runtime-only observability flag; persisted inside Metadata rather than as
	// a LiteLLM_SpendLogs column.
	ComparisonEligible bool
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
	QueueLen                   int           // Current input channel length
	QueueCap                   int           // Input channel capacity
	PendingEntries             int           // Accepted entries not yet resolved by the writer
	PendingAggregation         int           // Inserted batches awaiting/in daily aggregation
	DLQSize                    int           // Current dead letter queue size in batches
	Queued                     uint64        // Total queued
	Written                    uint64        // Newly inserted raw rows
	Dropped                    uint64        // Dropped (queue full - timeout reached)
	Errors                     uint64        // Write errors
	BatchesOK                  uint64        // Successful batches
	QueueFullCount             uint64        // Queue full events (timeouts)
	DLQCount                   uint64        // Batches sent to DLQ
	DLQRecovered               uint64        // Batches recovered from DLQ
	DLQOverflow                uint64        // Batches lost because DLQ was full
	Duplicates                 uint64        // Rows ignored by ON CONFLICT
	AggregationCount           uint64        // Completed aggregations
	AggregationErrors          uint64        // Aggregation errors
	PendingAggregationOverflow uint64        // Inserted batches lost before daily aggregation
	ComparisonEligible         uint64        // Newly inserted rows eligible for full comparison
	ComparisonIneligible       uint64        // Newly inserted rows excluded from full comparison
	LastAggregationTime        time.Time     // Last successful aggregation
	AggregationLag             time.Duration // Age of the oldest outstanding daily aggregation
	ComparisonWindowValid      bool          // Conservative process-lifetime transport completeness
}
