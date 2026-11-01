// Package auth authenticates API tokens against the LiteLLM database.
package auth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// Authenticator provides token authentication via LiteLLM database
// Synchronous (blocking) - token validation must complete before request processing
type Authenticator struct {
	pool   *connection.ConnectionPool
	cache  *Cache
	logger *slog.Logger

	// costMarginEnabled gates loading cost_margin_config (see costMarginConfigs)
	costMarginEnabled bool

	// budgetCalendar is where the key budget windows start
	budgetCalendar models.BudgetCalendar
}

// NewAuthenticator creates a new authenticator
func NewAuthenticator(pool *connection.ConnectionPool, cache *Cache, logger *slog.Logger) *Authenticator {
	return &Authenticator{
		pool:   pool,
		cache:  cache,
		logger: logger,
	}
}

// WithCostMargin enables applying metadata.cost_margin_config. When disabled
// TokenInfo.CostMarginConfigs stays empty, so no margin is ever added.
func (a *Authenticator) WithCostMargin(enabled bool) *Authenticator {
	a.costMarginEnabled = enabled
	return a
}

// WithBudgetCalendar sets the calendar the key budget windows start and end
// on (litellm_db.budget_timezone and budget_reset_time).
func (a *Authenticator) WithBudgetCalendar(calendar models.BudgetCalendar) *Authenticator {
	a.budgetCalendar = calendar
	return a
}

// FetchMasterKey seeds the auth cache with the proxy master key. The config
// value is the source of truth: the copy litellm stores in
// LiteLLM_Config.general_settings never overrides it and is read only to
// detect drift — a differing DB value means litellm and AIR are configured
// with different keys, so the trusted litellm->AIR hop would break. The DB
// copy is used only when the config value is empty, and a DB failure is
// non-fatal so the config key keeps working without the DB.
func (a *Authenticator) FetchMasterKey(ctx context.Context, defaultKey string) error {
	dbKey := a.fetchMasterKeyFromDB(ctx)

	masterKey := defaultKey
	source := "config"
	switch {
	case masterKey == "":
		masterKey = dbKey
		source = "DB"
	case dbKey != "" && dbKey != masterKey:
		a.logger.Warn("Master key in LiteLLM DB differs from config; config value takes precedence")
	}
	if masterKey == "" {
		return models.ErrTokenNotFound
	}

	info := models.TokenInfo{
		Token:       HashToken(masterKey),
		KeyName:     "litellm-master-key",
		UserID:      "litellm-master-key",
		IsMasterKey: true,
	}
	a.cache.Set(info.Token, &info)

	a.logger.Debug("Master key loaded", "source", source)

	return nil
}

// fetchMasterKeyFromDB returns the master key stored in LiteLLM_Config, or ""
// when the database is unavailable or stores none.
func (a *Authenticator) fetchMasterKeyFromDB(ctx context.Context) string {
	if a.pool == nil || !a.pool.IsHealthy() {
		return ""
	}

	conn, err := a.pool.Acquire(ctx)
	if err != nil {
		a.logger.Warn("Failed to acquire connection for master key lookup",
			"error", err,
		)
		return ""
	}
	defer conn.Release()

	var masterKey *string
	if err := conn.QueryRow(ctx, queries.QueryMasterKey).Scan(&masterKey); err != nil {
		a.logger.Warn("Failed to fetch master key from DB",
			"error", err,
		)
		return ""
	}
	if masterKey == nil {
		return ""
	}
	return *masterKey
}

// ValidateToken validates a token and returns its information
//
// Algorithm:
// 1. Hash token (sha256) if it starts with "sk-"
// 2. Check cache
// 3. If not in cache - query database
// 4. Validate (blocked, expires, budget)
// 5. Cache result
//
// Returns error if token is invalid or database is unavailable
func (a *Authenticator) ValidateToken(ctx context.Context, rawToken string) (*models.TokenInfo, error) {
	if rawToken == "" {
		return nil, models.ErrTokenNotFound
	}

	// 1. Hash token
	hashedToken := HashToken(rawToken)

	// 2. Check cache. A token loaded in an earlier budget window carries that
	// window's spend, so it is reloaded instead.
	if info, ok := a.cache.Get(hashedToken); ok && !info.BudgetWindowsStale(utils.NowUTC(), a.budgetCalendar) {
		a.logger.Debug("Token found in cache",
			"token_prefix", security.MaskToken(hashedToken),
		)
		// Validate even from cache (expires, budget could have changed externally)
		if err := info.Validate(""); err != nil {
			return nil, err
		}
		return info, nil
	}

	// 3. Query database
	info, err := a.fetchTokenFromDB(ctx, hashedToken)
	if err != nil {
		return nil, err
	}

	// 4. Validate
	if err := info.Validate(""); err != nil {
		// Don't cache invalid tokens
		return nil, err
	}

	// 5. Cache
	a.cache.Set(hashedToken, info)

	a.logger.Debug("Token validated from DB",
		"token_prefix", security.MaskToken(hashedToken),
		"user_id", info.UserID,
		"team_id", info.TeamID,
	)

	return info, nil
}

// ValidateTokenForModel validates a token with model access check
func (a *Authenticator) ValidateTokenForModel(ctx context.Context, rawToken, model string) (*models.TokenInfo, error) {
	info, err := a.ValidateToken(ctx, rawToken)
	if err != nil {
		return nil, err
	}

	// Check model access
	if !info.IsModelAllowed(model) {
		return nil, models.ErrModelNotAllowed
	}

	return info, nil
}

// fetchTokenFromDB loads token from database with full budget hierarchy
// Single query loads: Token → User → Team → Organization → Memberships
// All with their budget data (embedded or external via BudgetTable)
func (a *Authenticator) fetchTokenFromDB(ctx context.Context, hashedToken string) (*models.TokenInfo, error) {
	if a.pool == nil || !a.pool.IsHealthy() {
		return nil, models.ErrConnectionFailed
	}

	conn, err := a.pool.Acquire(ctx)
	if err != nil {
		a.logger.Error("Failed to acquire connection for auth",
			"error", err,
		)
		return nil, models.ErrConnectionFailed
	}
	defer conn.Release()
	info, err := a.tokenFromRow(conn.QueryRow(ctx, queries.QueryValidateTokenWithHierarchy, hashedToken), hashedToken)
	if err != nil {
		return nil, err
	}
	if err := a.loadBudgetWindowSpend(ctx, conn, info, utils.NowUTC()); err != nil {
		a.logger.Error("Failed to load key budget window spend",
			"error", err,
			"token_prefix", security.MaskToken(hashedToken),
		)
		return nil, models.ErrConnectionFailed
	}
	return info, nil
}

// tokenFromRow reads one QueryValidateTokenWithHierarchy row into a TokenInfo.
func (a *Authenticator) tokenFromRow(row pgx.Row, hashedToken string) (*models.TokenInfo, error) {
	var info models.TokenInfo

	// ============ Token fields ============
	var keyName, keyAlias, userID, teamID, orgID *string
	var tokenMaxBudget *float64
	var tokenTPMLimit, tokenRPMLimit *int64
	var expires *time.Time
	var blocked *bool
	var tokenModels []string
	var tokenMetadata []byte
	var tokenBudgetLimits []byte

	// ============ User fields ============
	var userIDCheck, userAlias, userEmail *string
	var userMaxBudget, userSpend *float64
	var userTPMLimit, userRPMLimit *int64
	var userModels []string
	var userMetadata []byte

	// ============ Team fields ============
	var teamIDCheck, teamAlias *string
	var teamOrganizationID *string
	var teamMaxBudget, teamSpend *float64
	var teamBlocked *bool
	var teamTPMLimit, teamRPMLimit *int64
	var teamModels []string
	var teamMetadata []byte

	// ============ Organization fields (with external budget) ============
	var orgIDCheck, orgAlias *string
	var orgSpend *float64
	var orgMaxBudget *float64
	var orgTPMLimit, orgRPMLimit *int64
	var orgMetadata []byte

	// ============ TeamMembership fields (with external budget) ============
	var teamMemberSpend *float64
	var teamMemberMaxBudget *float64
	var teamMemberTPMLimit, teamMemberRPMLimit *int64
	var teamMemberModels []string

	// ============ OrganizationMembership fields (with external budget) ============
	var orgMemberSpend *float64
	var orgMemberMaxBudget *float64
	var orgMemberTPMLimit, orgMemberRPMLimit *int64

	err := row.Scan(
		// Token
		&info.Token,
		&keyName,
		&keyAlias,
		&userID,
		&teamID,
		&orgID,
		&info.Spend,
		&tokenMaxBudget,
		&tokenTPMLimit,
		&tokenRPMLimit,
		&expires,
		&blocked,
		&tokenModels,
		&tokenMetadata,
		&tokenBudgetLimits,

		// User
		&userIDCheck,
		&userAlias,
		&userEmail,
		&userMaxBudget,
		&userSpend,
		&userTPMLimit,
		&userRPMLimit,
		&userModels,
		&userMetadata,

		// Team
		&teamIDCheck,
		&teamAlias,
		&teamOrganizationID, // team_organization_id (nullable, positional)
		&teamMaxBudget,
		&teamSpend,
		&teamBlocked,
		&teamTPMLimit,
		&teamRPMLimit,
		&teamModels,
		&teamMetadata,

		// Organization
		&orgIDCheck,
		&orgAlias,
		&orgSpend,
		&orgMaxBudget,
		&orgTPMLimit,
		&orgRPMLimit,
		&orgMetadata,

		// TeamMembership
		&teamMemberSpend,
		&teamMemberMaxBudget,
		&teamMemberTPMLimit,
		&teamMemberRPMLimit,
		&teamMemberModels,

		// OrganizationMembership
		&orgMemberSpend,
		&orgMemberMaxBudget,
		&orgMemberTPMLimit,
		&orgMemberRPMLimit,
	)

	if err != nil {
		if err == pgx.ErrNoRows {
			a.logger.Debug("Token not found in DB",
				"token_prefix", security.MaskToken(hashedToken),
			)
			return nil, models.ErrTokenNotFound
		}
		a.logger.Error("Failed to query token",
			"error", err,
			"token_prefix", security.MaskToken(hashedToken),
		)
		return nil, models.ErrConnectionFailed
	}

	// Convert nullable Token fields
	if keyName != nil {
		info.KeyName = *keyName
	}
	if keyAlias != nil {
		info.KeyAlias = *keyAlias
	}
	if userID != nil {
		info.UserID = *userID
	}
	if teamID != nil {
		info.TeamID = *teamID
	}
	if orgID != nil {
		info.OrganizationID = *orgID
		info.DirectOrganizationID = *orgID
	}
	if teamOrganizationID != nil {
		info.TeamOrganizationID = *teamOrganizationID
	}
	if blocked != nil {
		info.Blocked = *blocked
	}
	info.Models = tokenModels
	info.Metadata = decodeMetadata(tokenMetadata)
	info.TeamMetadata = decodeMetadata(teamMetadata)
	info.OrganizationMetadata = decodeMetadata(orgMetadata)
	if orgAlias != nil {
		info.OrganizationAlias = *orgAlias
	}

	info.MaxBudget = tokenMaxBudget
	info.BudgetWindows = a.budgetWindows(tokenBudgetLimits, hashedToken)
	info.Expires = expires
	info.TPMLimit = tokenTPMLimit
	info.RPMLimit = tokenRPMLimit

	// Set User fields (if user exists)
	if userAlias != nil {
		info.UserAlias = *userAlias
	}
	if userEmail != nil {
		info.UserEmail = *userEmail
	}
	info.UserMaxBudget = userMaxBudget
	info.UserSpend = userSpend
	info.UserTPMLimit = userTPMLimit
	info.UserRPMLimit = userRPMLimit
	info.UserModels = userModels

	// Set Team fields (if team exists)
	if teamAlias != nil {
		info.TeamAlias = *teamAlias
	}
	info.TeamMaxBudget = teamMaxBudget
	info.TeamSpend = teamSpend
	info.TeamBlocked = teamBlocked
	info.TeamTPMLimit = teamTPMLimit
	info.TeamRPMLimit = teamRPMLimit
	info.TeamModels = teamModels

	// A missing referenced team denies instead of becoming unrestricted.
	info.TeamDangling = teamID != nil && teamIDCheck == nil
	info.DirectOrganizationDangling = orgID != nil && orgIDCheck == nil
	info.TeamOrganizationDangling = orgID == nil && teamOrganizationID != nil && orgIDCheck == nil

	// Set Organization fields (external budget from BudgetTable)
	info.OrgSpend = orgSpend
	info.OrgMaxBudget = orgMaxBudget
	info.OrgTPMLimit = orgTPMLimit
	info.OrgRPMLimit = orgRPMLimit

	// Set TeamMembership fields (external budget from BudgetTable)
	info.TeamMemberSpend = teamMemberSpend
	info.TeamMemberMaxBudget = teamMemberMaxBudget
	info.TeamMemberTPMLimit = teamMemberTPMLimit
	info.TeamMemberRPMLimit = teamMemberRPMLimit
	info.TeamMemberModels = teamMemberModels

	// Set OrganizationMembership fields (external budget from BudgetTable)
	info.OrgMemberSpend = orgMemberSpend
	info.OrgMemberMaxBudget = orgMemberMaxBudget
	info.OrgMemberTPMLimit = orgMemberTPMLimit
	info.OrgMemberRPMLimit = orgMemberRPMLimit

	// Set cost margin layers (key -> team -> org, or key -> user for a personal key)
	info.CostMarginConfigs = a.costMarginConfigs(&info, tokenMetadata, userMetadata, teamMetadata, orgMetadata)

	a.logger.Debug("Token loaded with full hierarchy",
		"token_prefix", security.MaskToken(hashedToken),
		"user_id", info.UserID,
		"team_id", info.TeamID,
		"org_id", info.OrganizationID,
	)

	return &info, nil
}

// budgetWindows reads the key's budget_limits. A budget_limits AIR cannot
// read yields one unenforceable window, so Validate refuses the key instead of
// leaving it unlimited.
func (a *Authenticator) budgetWindows(raw []byte, hashedToken string) []models.BudgetWindow {
	limits, err := models.ParseBudgetLimits(raw)
	if err != nil {
		a.logger.Warn("budget_limits has unexpected structure, refusing the key",
			"token_prefix", security.MaskToken(hashedToken),
			"error", err,
		)
		return []models.BudgetWindow{{}}
	}
	if len(limits) == 0 {
		return nil
	}
	windows := make([]models.BudgetWindow, len(limits))
	for i, limit := range limits {
		windows[i] = models.BudgetWindow{BudgetLimit: limit}
	}
	return windows
}

// budgetWindowQuerier is the part of a pooled connection loadBudgetWindowSpend
// reads through.
type budgetWindowQuerier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// loadBudgetWindowSpend sets each window's current start and the key's spend
// in it: the window row's spend when the row is on that window, else the sum
// of the key's accounted spend logs since the start (a window nothing has
// written yet, or a row still on an earlier window). A schema without
// LiteLLM_BudgetWindowSpend reads every window from the spend logs.
func (a *Authenticator) loadBudgetWindowSpend(ctx context.Context, conn budgetWindowQuerier, info *models.TokenInfo, now time.Time) error {
	if len(info.BudgetWindows) == 0 {
		return nil
	}
	type windowRow struct {
		start time.Time
		spend float64
	}
	stored := map[string]windowRow{}
	rows, err := conn.Query(ctx, queries.QuerySelectKeyBudgetWindowSpend, info.Token)
	if err != nil && !isUndefinedTable(err) {
		return err
	}
	if err == nil {
		for rows.Next() {
			var duration string
			var row windowRow
			if err := rows.Scan(&duration, &row.start, &row.spend); err != nil {
				rows.Close()
				return err
			}
			stored[duration] = row
		}
		rows.Close()
		if err := rows.Err(); err != nil && !isUndefinedTable(err) {
			return err
		}
	}

	for i := range info.BudgetWindows {
		window := &info.BudgetWindows[i]
		duration, ok := models.ParseBudgetDuration(window.Duration)
		if !ok {
			continue
		}
		start, end := duration.Bounds(now, a.budgetCalendar)
		window.Start = start
		if row, ok := stored[window.Duration]; ok && row.start.Equal(start) {
			window.Spend = row.spend
			continue
		}
		if err := conn.QueryRow(ctx, queries.QuerySumKeyWindowSpendLogs, info.Token, start, end).Scan(&window.Spend); err != nil {
			return err
		}
	}
	return nil
}

// isUndefinedTable reports whether err is PostgreSQL's undefined_table (42P01).
func isUndefinedTable(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42P01"
}

// costMarginConfigs collects the cost_margin_config layers in priority order.
// The key comes first. A team or organization key then bills the team and its
// organization, a personal key (no team, no organization) bills the user, the
// same split budgetLevels uses. Parsing is best-effort: a config AIR does not
// understand must not fail authentication.
func (a *Authenticator) costMarginConfigs(info *models.TokenInfo, keyMetadata, userMetadata, teamMetadata, orgMetadata []byte) []models.CostMarginConfig {
	if !a.costMarginEnabled {
		return nil
	}
	type layer struct {
		level string
		raw   []byte
	}
	layers := []layer{{"key", metadataCostMarginConfig(keyMetadata)}}
	switch {
	case info.TeamID != "":
		layers = append(layers,
			layer{"team", metadataCostMarginConfig(teamMetadata)},
			layer{"organization", metadataCostMarginConfig(orgMetadata)},
		)
	case info.OrganizationID != "":
		layers = append(layers, layer{"organization", metadataCostMarginConfig(orgMetadata)})
	default:
		layers = append(layers, layer{"user", metadataCostMarginConfig(userMetadata)})
	}

	var configs []models.CostMarginConfig
	for _, layer := range layers {
		cfg, err := models.ParseCostMarginConfig(layer.raw)
		if err != nil {
			a.logger.Warn("cost_margin_config has unexpected structure, ignoring the unusable parts",
				"level", layer.level,
				"token_prefix", security.MaskToken(info.Token),
				"error", err,
			)
		}
		if len(cfg) > 0 {
			configs = append(configs, cfg)
		}
	}
	return configs
}

// metadataCostMarginConfig extracts cost_margin_config from a LiteLLM metadata column.
func metadataCostMarginConfig(raw []byte) []byte {
	var metadata struct {
		CostMarginConfig json.RawMessage `json:"cost_margin_config"`
	}
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil
	}
	return metadata.CostMarginConfig
}

func decodeMetadata(raw []byte) map[string]interface{} {
	if len(raw) == 0 {
		return nil
	}
	var metadata map[string]interface{}
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil
	}
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

// InvalidateToken removes a token from cache
func (a *Authenticator) InvalidateToken(hashedToken string) {
	a.cache.Invalidate(hashedToken)
}

// CacheStats returns cache statistics
func (a *Authenticator) CacheStats() models.AuthCacheStats {
	return a.cache.Stats()
}

// ==================== Helper functions ====================

// HashToken hashes a token using LiteLLM algorithm
// Tokens starting with "sk-" are hashed with SHA256
// Others are returned as-is (already hashed)
func HashToken(token string) string {
	if strings.HasPrefix(token, "sk-") {
		hash := sha256.Sum256([]byte(token))
		return hex.EncodeToString(hash[:])
	}
	return token
}
