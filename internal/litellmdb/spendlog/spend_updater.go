package spendlog

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// Schema compatibility flags — set to false on first SQLSTATE 42703 (column does not exist).
// Persist across retries: first attempt detects missing column, subsequent retries use fallback query.
var (
	schemaTokenHasLastActive      atomic.Bool
	schemaTeamMemberHasTotalSpend atomic.Bool
)

func init() {
	schemaTokenHasLastActive.Store(true)
	schemaTeamMemberHasTotalSpend.Store(true)
}

// isColumnNotExist returns true if err is PostgreSQL SQLSTATE 42703 (undefined_column).
func isColumnNotExist(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42703"
}

// SpendUpdates holds aggregated spend updates with explicit typing per entity.
// Avoids confusion with string keys and improves code readability.
type SpendUpdates struct {
	Tokens              map[entityModelKey]float64        // apiKey/model -> amount
	Users               map[entityModelKey]float64        // userID/model -> amount
	Teams               map[entityModelKey]float64        // teamID/model -> amount
	Orgs                map[entityModelKey]float64        // orgID/model -> amount
	TeamMembers         map[teamMemberKey]float64         // team/user -> amount
	OrganizationMembers map[organizationMemberKey]float64 // organization/user -> amount
	EndUsers            map[string]float64                // endUserID -> amount
}

type entityModelKey struct {
	EntityID string
	Model    string
}

type teamMemberKey struct {
	TeamID string
	UserID string
}

type organizationMemberKey struct {
	OrganizationID string
	UserID         string
}

type spendUpdateExecer interface {
	Exec(context.Context, string, ...interface{}) (pgconn.CommandTag, error)
}

// sortedSpendKeys returns a stable row-lock acquisition order. PostgreSQL
// transactions from different AIR replicas can touch the same aggregate rows;
// deterministic ordering makes them queue instead of deadlocking because of
// Go's randomized map iteration.
func sortedSpendKeys[K comparable](values map[K]float64, compare func(K, K) int) []K {
	keys := make([]K, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, compare)
	return keys
}

func compareEntityModelKey(left, right entityModelKey) int {
	if order := strings.Compare(left.EntityID, right.EntityID); order != 0 {
		return order
	}
	return strings.Compare(left.Model, right.Model)
}

func compareTeamMemberKey(left, right teamMemberKey) int {
	if order := strings.Compare(left.TeamID, right.TeamID); order != 0 {
		return order
	}
	return strings.Compare(left.UserID, right.UserID)
}

func compareOrganizationMemberKey(left, right organizationMemberKey) int {
	if order := strings.Compare(left.OrganizationID, right.OrganizationID); order != 0 {
		return order
	}
	return strings.Compare(left.UserID, right.UserID)
}

// aggregateSpendUpdates groups spend updates by entity.
// Instead of N UPDATE queries for N SpendLogEntry, aggregates them into ~5-10 operations.
//
// Example:
//
//	Batch of 100 entries:
//	  - APIKey: "abc", Spend: 10
//	  - APIKey: "abc", Spend: 5    ← same entity!
//	  - UserID: "user1", Spend: 8
//	  - UserID: "user1", Spend: 7  ← same entity!
//
//	Result:
//	  SpendUpdates {
//	    Tokens: {"abc": 15},       ← 1 UPDATE instead of 2
//	    Users: {"user1": 15},      ← 1 UPDATE instead of 2
//	  }
func aggregateSpendUpdates(batch []*models.SpendLogEntry) *SpendUpdates {
	updates := &SpendUpdates{
		Tokens:              make(map[entityModelKey]float64),
		Users:               make(map[entityModelKey]float64),
		Teams:               make(map[entityModelKey]float64),
		Orgs:                make(map[entityModelKey]float64),
		TeamMembers:         make(map[teamMemberKey]float64),
		OrganizationMembers: make(map[organizationMemberKey]float64),
		EndUsers:            make(map[string]float64),
	}

	for _, entry := range batch {
		if entry == nil {
			continue
		}

		if entry.APIKey != "" {
			updates.Tokens[entityModelKey{EntityID: entry.APIKey, Model: entry.Model}] += entry.Spend
		}

		// Team spend is charged through team membership, not the personal
		// balance. Routed on BillingTeamID, not TeamID: TeamID can hold a
		// synthetic provider-credential name (credential_name_as_team_id)
		// used only for log/DailyTeamSpend attribution, and that must not
		// suppress the personal debit or be charged to a nonexistent team.
		if entry.UserID != "" && entry.BillingTeamID == "" {
			updates.Users[entityModelKey{EntityID: entry.UserID, Model: entry.Model}] += entry.Spend
		}

		// Team (if present)
		if entry.BillingTeamID != "" {
			updates.Teams[entityModelKey{EntityID: entry.BillingTeamID, Model: entry.Model}] += entry.Spend
		}

		// Organization (if present)
		if entry.OrganizationID != "" {
			updates.Orgs[entityModelKey{EntityID: entry.OrganizationID, Model: entry.Model}] += entry.Spend
		}

		// TeamMembership (if User + Team)
		if entry.UserID != "" && entry.BillingTeamID != "" {
			updates.TeamMembers[teamMemberKey{TeamID: entry.BillingTeamID, UserID: entry.UserID}] += entry.Spend
		}

		if entry.UserID != "" && entry.OrganizationID != "" {
			updates.OrganizationMembers[organizationMemberKey{
				OrganizationID: entry.OrganizationID,
				UserID:         entry.UserID,
			}] += entry.Spend
		}

		if entry.EndUser != "" {
			updates.EndUsers[entry.EndUser] += entry.Spend
		}
	}

	return updates
}

// executeSpendUpdates executes all UPDATE operations within the given transaction.
// If any operation fails, the entire transaction rolls back (atomicity).
func executeSpendUpdates(ctx context.Context, tx pgx.Tx, updates *SpendUpdates) error {
	if updates == nil {
		return nil
	}

	// Execute each update type (skip empty maps)
	if len(updates.Tokens) > 0 {
		if err := updateTokens(ctx, tx, updates.Tokens); err != nil {
			return fmt.Errorf("update tokens: %w", err)
		}
	}
	if len(updates.Users) > 0 {
		if err := updateUsers(ctx, tx, updates.Users); err != nil {
			return fmt.Errorf("update users: %w", err)
		}
	}
	if len(updates.Teams) > 0 {
		if err := updateTeams(ctx, tx, updates.Teams); err != nil {
			return fmt.Errorf("update teams: %w", err)
		}
	}
	if len(updates.TeamMembers) > 0 {
		if err := updateTeamMembers(ctx, tx, updates.TeamMembers); err != nil {
			return fmt.Errorf("update team members: %w", err)
		}
	}
	if len(updates.Orgs) > 0 {
		if err := updateOrgs(ctx, tx, updates.Orgs); err != nil {
			return fmt.Errorf("update orgs: %w", err)
		}
	}
	if len(updates.OrganizationMembers) > 0 {
		if err := updateOrganizationMembers(ctx, tx, updates.OrganizationMembers); err != nil {
			return fmt.Errorf("update organization members: %w", err)
		}
	}
	if len(updates.EndUsers) > 0 {
		if err := updateEndUsers(ctx, tx, updates.EndUsers); err != nil {
			return fmt.Errorf("update end users: %w", err)
		}
	}

	return nil
}

// sortedKeys preserves the current-main helper contract for scalar counter
// maps while sharing the typed deterministic ordering used by migration rows.
func sortedKeys(values map[string]float64) []string {
	return sortedSpendKeys(values, strings.Compare)
}

// updateTokens updates Token.spend and model_spend in LiteLLM_VerificationToken.
// Executes a single UPDATE per apiKey/model pair.
// Falls back to query without last_active if the column doesn't exist (older DB schema).
func updateTokens(ctx context.Context, tx spendUpdateExecer, tokens map[entityModelKey]float64) error {
	for _, key := range sortedSpendKeys(tokens, compareEntityModelKey) {
		amount := tokens[key]
		var err error
		if schemaTokenHasLastActive.Load() {
			if key.Model == "" {
				_, err = tx.Exec(ctx, queries.QueryUpdateTokenSpendWithLastActive, amount, key.EntityID)
			} else {
				_, err = tx.Exec(ctx, queries.QueryUpdateTokenModelSpendWithLastActive, amount, key.Model, key.EntityID)
			}
			if err != nil && isColumnNotExist(err) {
				schemaTokenHasLastActive.Store(false)
				// Transaction aborted — caller will retry; next attempt uses fallback query.
				return err
			}
		} else {
			if key.Model == "" {
				_, err = tx.Exec(ctx, queries.QueryUpdateTokenSpend, amount, key.EntityID)
			} else {
				_, err = tx.Exec(ctx, queries.QueryUpdateTokenModelSpend, amount, key.Model, key.EntityID)
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// updateUsers updates LiteLLM_UserTable.spend and model_spend.
func updateUsers(ctx context.Context, tx spendUpdateExecer, users map[entityModelKey]float64) error {
	for _, key := range sortedSpendKeys(users, compareEntityModelKey) {
		amount := users[key]
		query, args := modelSpendUpdate(
			`"LiteLLM_UserTable"`, "user_id", key, amount,
		)
		_, err := tx.Exec(ctx, query, args...)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateTeams updates LiteLLM_TeamTable.spend and model_spend.
func updateTeams(ctx context.Context, tx spendUpdateExecer, teams map[entityModelKey]float64) error {
	for _, key := range sortedSpendKeys(teams, compareEntityModelKey) {
		amount := teams[key]
		query, args := modelSpendUpdate(
			`"LiteLLM_TeamTable"`, "team_id", key, amount,
		)
		_, err := tx.Exec(ctx, query, args...)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateOrgs updates LiteLLM_OrganizationTable.spend and model_spend.
func updateOrgs(ctx context.Context, tx spendUpdateExecer, orgs map[entityModelKey]float64) error {
	for _, key := range sortedSpendKeys(orgs, compareEntityModelKey) {
		amount := orgs[key]
		query, args := modelSpendUpdate(
			`"LiteLLM_OrganizationTable"`, "organization_id", key, amount,
		)
		_, err := tx.Exec(ctx, query, args...)
		if err != nil {
			return err
		}
	}
	return nil
}

// modelSpendUpdate builds an entity counter update where scalar spend and the
// per-model JSON number are changed by the same statement. Table and ID column
// are compile-time constants supplied only by the wrappers above.
func modelSpendUpdate(table, idColumn string, key entityModelKey, amount float64) (string, []interface{}) {
	if key.Model == "" {
		return fmt.Sprintf(queries.QueryUpdateEntitySpendTemplate, table, idColumn),
			[]interface{}{amount, key.EntityID}
	}

	return fmt.Sprintf(queries.QueryUpdateEntityModelSpendTemplate, table, idColumn),
		[]interface{}{amount, key.Model, key.EntityID}
}

// updateTeamMembers updates LiteLLM_TeamMembership.spend (and total_spend if available).
// Uses a typed composite key so identifiers containing ':' cannot collide.
// Falls back to query without total_spend if the column doesn't exist (older DB schema).
func updateTeamMembers(ctx context.Context, tx spendUpdateExecer, teamMembers map[teamMemberKey]float64) error {
	for _, key := range sortedSpendKeys(teamMembers, compareTeamMemberKey) {
		amount := teamMembers[key]
		if key.TeamID == "" || key.UserID == "" {
			return fmt.Errorf("invalid team member key: empty teamID or userID")
		}

		var err error
		if schemaTeamMemberHasTotalSpend.Load() {
			_, err = tx.Exec(ctx, queries.QueryUpdateTeamMemberSpendWithTotal, amount, key.TeamID, key.UserID)
			if err != nil && isColumnNotExist(err) {
				schemaTeamMemberHasTotalSpend.Store(false)
				// Transaction aborted — caller will retry; next attempt uses fallback query.
				return err
			}
		} else {
			_, err = tx.Exec(ctx, queries.QueryUpdateTeamMemberSpend, amount, key.TeamID, key.UserID)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func updateOrganizationMembers(
	ctx context.Context,
	tx spendUpdateExecer,
	organizationMembers map[organizationMemberKey]float64,
) error {
	for _, key := range sortedSpendKeys(organizationMembers, compareOrganizationMemberKey) {
		amount := organizationMembers[key]
		if key.OrganizationID == "" || key.UserID == "" {
			return fmt.Errorf("invalid organization member key: empty organizationID or userID")
		}
		_, err := tx.Exec(ctx, queries.QueryUpdateOrganizationMemberSpend, amount, key.OrganizationID, key.UserID)
		if err != nil {
			return err
		}
	}
	return nil
}

func updateEndUsers(ctx context.Context, tx spendUpdateExecer, endUsers map[string]float64) error {
	for _, endUserID := range sortedSpendKeys(endUsers, strings.Compare) {
		amount := endUsers[endUserID]
		_, err := tx.Exec(ctx, queries.QueryUpsertEndUserSpend, endUserID, amount)
		if err != nil {
			return err
		}
	}
	return nil
}
