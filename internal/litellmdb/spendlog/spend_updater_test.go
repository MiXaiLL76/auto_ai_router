package spendlog

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAggregateSpendUpdates_AllEntities tests aggregation with all entity types
func TestAggregateSpendUpdates_AllEntities(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey:         "token-1",
			UserID:         "user-1",
			TeamID:         "team-1",
			BillingTeamID:  "team-1",
			OrganizationID: "org-1",
			Model:          "model-1",
			EndUser:        "end-user-1",
			Spend:          10.0,
		},
		{
			APIKey:         "token-1",
			UserID:         "user-1",
			TeamID:         "team-1",
			BillingTeamID:  "team-1",
			OrganizationID: "org-1",
			Model:          "model-1",
			EndUser:        "end-user-1",
			Spend:          5.0,
		},
		{
			APIKey: "token-2",
			UserID: "user-2",
			Spend:  3.0,
		},
	}

	result := aggregateSpendUpdates(batch)

	// Token aggregation
	assert.Equal(t, 15.0, result.Tokens[entityModelKey{EntityID: "token-1", Model: "model-1"}])
	assert.Equal(t, 3.0, result.Tokens[entityModelKey{EntityID: "token-2"}])

	// User aggregation
	assert.Equal(t, 15.0, result.Users[entityModelKey{EntityID: "user-1", Model: "model-1"}])
	assert.Equal(t, 3.0, result.Users[entityModelKey{EntityID: "user-2"}])

	// Team aggregation
	assert.Equal(t, 15.0, result.Teams[entityModelKey{EntityID: "team-1", Model: "model-1"}])

	// Org aggregation
	assert.Equal(t, 15.0, result.Orgs[entityModelKey{EntityID: "org-1", Model: "model-1"}])

	// Team membership
	assert.Equal(t, 15.0, result.TeamMembers[teamMemberKey{TeamID: "team-1", UserID: "user-1"}])
	assert.Equal(t, 15.0, result.OrganizationMembers[organizationMemberKey{OrganizationID: "org-1", UserID: "user-1"}])

	assert.Equal(t, 15.0, result.EndUsers["end-user-1"])
}

// TestAggregateSpendUpdates_EmptyBatch tests empty batch
func TestAggregateSpendUpdates_EmptyBatch(t *testing.T) {
	batch := []*models.SpendLogEntry{}
	result := aggregateSpendUpdates(batch)

	assert.Empty(t, result.Tokens)
	assert.Empty(t, result.Users)
	assert.Empty(t, result.Teams)
	assert.Empty(t, result.Orgs)
	assert.Empty(t, result.TeamMembers)
	assert.Empty(t, result.OrganizationMembers)
	assert.Empty(t, result.EndUsers)
}

// TestAggregateSpendUpdates_NilBatch tests nil batch
func TestAggregateSpendUpdates_NilBatch(t *testing.T) {
	result := aggregateSpendUpdates(nil)
	// Function returns initialized empty map, not nil
	assert.Empty(t, result.Tokens)
}

// TestAggregateSpendUpdates_PartialEntities tests with some entities missing
func TestAggregateSpendUpdates_PartialEntities(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey: "token-1",
			Spend:  10.0,
			// No UserID, TeamID, OrganizationID
		},
		{
			APIKey: "token-1",
			UserID: "user-1",
			Spend:  5.0,
			// No TeamID, OrganizationID
		},
		{
			APIKey:        "token-2",
			TeamID:        "team-1",
			BillingTeamID: "team-1",
			Spend:         3.0,
			// No UserID, OrganizationID
		},
	}

	result := aggregateSpendUpdates(batch)

	// Token aggregation works
	assert.Equal(t, 15.0, result.Tokens[entityModelKey{EntityID: "token-1"}])
	assert.Equal(t, 3.0, result.Tokens[entityModelKey{EntityID: "token-2"}])

	// User aggregation works
	assert.Equal(t, 5.0, result.Users[entityModelKey{EntityID: "user-1"}])

	// Team aggregation works
	assert.Equal(t, 3.0, result.Teams[entityModelKey{EntityID: "team-1"}])

	// Org should be empty
	assert.Empty(t, result.Orgs)
}

// TestAggregateSpendUpdates_TeamMember tests team membership with user
func TestAggregateSpendUpdates_TeamMember(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey:        "token-1",
			UserID:        "user-1",
			TeamID:        "team-1",
			BillingTeamID: "team-1",
			Spend:         10.0,
		},
		{
			APIKey:        "token-1",
			UserID:        "user-2",
			TeamID:        "team-1",
			BillingTeamID: "team-1",
			Spend:         5.0,
		},
	}

	result := aggregateSpendUpdates(batch)

	assert.Equal(t, 10.0, result.Users[entityModelKey{EntityID: "user-1"}])
	assert.Equal(t, 5.0, result.Users[entityModelKey{EntityID: "user-2"}])
	assert.Equal(t, 15.0, result.Teams[entityModelKey{EntityID: "team-1"}])

	// Team membership should aggregate by team:user
	assert.Equal(t, 10.0, result.TeamMembers[teamMemberKey{TeamID: "team-1", UserID: "user-1"}])
	assert.Equal(t, 5.0, result.TeamMembers[teamMemberKey{TeamID: "team-1", UserID: "user-2"}])
}

func TestAggregateSpendUpdates_OrganizationMembership(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey:         "token-1",
			UserID:         "user-1",
			OrganizationID: "org-1",
			Spend:          10.0,
		},
		{
			APIKey:         "token-1",
			UserID:         "user-2",
			OrganizationID: "org-1",
			Spend:          5.0,
		},
	}

	result := aggregateSpendUpdates(batch)
	assert.Equal(t, 15.0, result.Orgs[entityModelKey{EntityID: "org-1"}])
	assert.Equal(t, 10.0, result.OrganizationMembers[organizationMemberKey{OrganizationID: "org-1", UserID: "user-1"}])
	assert.Equal(t, 5.0, result.OrganizationMembers[organizationMemberKey{OrganizationID: "org-1", UserID: "user-2"}])
}

// TestExecuteSpendUpdates_NilUpdates tests nil updates
func TestExecuteSpendUpdates_NilUpdates(t *testing.T) {
	// Can't actually test without DB connection, but verify it doesn't panic
	// This is tested via integration tests with real DB
}

// TestSpendUpdates_Fields verifies SpendUpdates structure
func TestSpendUpdates_Fields(t *testing.T) {
	updates := &SpendUpdates{
		Tokens:              map[entityModelKey]float64{{EntityID: "key1", Model: "model1"}: 1.0},
		Users:               map[entityModelKey]float64{{EntityID: "user1", Model: "model1"}: 2.0},
		Teams:               map[entityModelKey]float64{{EntityID: "team1", Model: "model1"}: 3.0},
		Orgs:                map[entityModelKey]float64{{EntityID: "org1", Model: "model1"}: 4.0},
		TeamMembers:         map[teamMemberKey]float64{{TeamID: "team1", UserID: "user1"}: 5.0},
		OrganizationMembers: map[organizationMemberKey]float64{{OrganizationID: "org1", UserID: "user1"}: 5.5},
		EndUsers:            map[string]float64{"end-user1": 6.0},
	}

	assert.Len(t, updates.Tokens, 1)
	assert.Len(t, updates.Users, 1)
	assert.Len(t, updates.Teams, 1)
	assert.Len(t, updates.Orgs, 1)
	assert.Len(t, updates.TeamMembers, 1)
	assert.Len(t, updates.OrganizationMembers, 1)
	assert.Len(t, updates.EndUsers, 1)

	assert.Equal(t, 1.0, updates.Tokens[entityModelKey{EntityID: "key1", Model: "model1"}])
	assert.Equal(t, 2.0, updates.Users[entityModelKey{EntityID: "user1", Model: "model1"}])
	assert.Equal(t, 3.0, updates.Teams[entityModelKey{EntityID: "team1", Model: "model1"}])
	assert.Equal(t, 4.0, updates.Orgs[entityModelKey{EntityID: "org1", Model: "model1"}])
	assert.Equal(t, 5.0, updates.TeamMembers[teamMemberKey{TeamID: "team1", UserID: "user1"}])
	assert.Equal(t, 5.5, updates.OrganizationMembers[organizationMemberKey{OrganizationID: "org1", UserID: "user1"}])
	assert.Equal(t, 6.0, updates.EndUsers["end-user1"])
}

// TestSpendUpdates_Empty verifies empty SpendUpdates
func TestSpendUpdates_Empty(t *testing.T) {
	updates := &SpendUpdates{}

	assert.Nil(t, updates.Tokens)
	assert.Nil(t, updates.Users)
	assert.Nil(t, updates.Teams)
	assert.Nil(t, updates.Orgs)
	assert.Nil(t, updates.TeamMembers)
	assert.Nil(t, updates.OrganizationMembers)
	assert.Nil(t, updates.EndUsers)
}

func TestSortedSpendKeysDeterministicAcrossTypedAggregateKeys(t *testing.T) {
	first := map[entityModelKey]float64{
		{EntityID: "team-b", Model: "model-z"}: 1,
		{EntityID: "team-a", Model: "model-z"}: 2,
		{EntityID: "team-a", Model: "model-a"}: 3,
	}
	second := map[entityModelKey]float64{
		{EntityID: "team-a", Model: "model-a"}: 7,
		{EntityID: "team-b", Model: "model-z"}: 8,
		{EntityID: "team-a", Model: "model-z"}: 9,
	}
	want := []entityModelKey{
		{EntityID: "team-a", Model: "model-a"},
		{EntityID: "team-a", Model: "model-z"},
		{EntityID: "team-b", Model: "model-z"},
	}

	for range 20 {
		assert.Equal(t, want, sortedSpendKeys(first, compareEntityModelKey))
		assert.Equal(t, want, sortedSpendKeys(second, compareEntityModelKey))
	}
}

type recordedSpendUpdateCall struct {
	query string
	args  []interface{}
}

type recordingSpendUpdateExecer struct {
	calls []recordedSpendUpdateCall
}

func (r *recordingSpendUpdateExecer) Exec(_ context.Context, query string, args ...interface{}) (pgconn.CommandTag, error) {
	r.calls = append(r.calls, recordedSpendUpdateCall{query: query, args: append([]interface{}(nil), args...)})
	return pgconn.CommandTag{}, nil
}

type endUserSpendExecer struct {
	recordingSpendUpdateExecer
	spendByUser map[string]float64
}

func (e *endUserSpendExecer) Exec(ctx context.Context, query string, args ...interface{}) (pgconn.CommandTag, error) {
	tag, err := e.recordingSpendUpdateExecer.Exec(ctx, query, args...)
	if err != nil {
		return tag, err
	}
	endUserID := args[0].(string)
	amount := args[1].(float64)
	e.spendByUser[endUserID] += amount
	return tag, nil
}

func TestUpdateEndUsersInsertsNewEndUser(t *testing.T) {
	execer := &endUserSpendExecer{spendByUser: make(map[string]float64)}

	require.NoError(t, updateEndUsers(context.Background(), execer, map[string]float64{
		"new-end-user": 1.25,
	}))

	assert.Equal(t, 1.25, execer.spendByUser["new-end-user"])
	assertEndUserUpsert(t, execer.calls, "new-end-user", 1.25)
}

func TestUpdateEndUsersIncrementsExistingEndUser(t *testing.T) {
	execer := &endUserSpendExecer{spendByUser: map[string]float64{"existing-end-user": 3.5}}

	require.NoError(t, updateEndUsers(context.Background(), execer, map[string]float64{
		"existing-end-user": 0.75,
	}))

	assert.Equal(t, 4.25, execer.spendByUser["existing-end-user"])
	assertEndUserUpsert(t, execer.calls, "existing-end-user", 0.75)
}

func assertEndUserUpsert(t *testing.T, calls []recordedSpendUpdateCall, endUserID string, amount float64) {
	t.Helper()
	require.Len(t, calls, 1)
	call := calls[0]
	assert.Contains(t, call.query, `INSERT INTO "LiteLLM_EndUserTable" (user_id, spend)`)
	assert.Contains(t, call.query, "ON CONFLICT (user_id) DO UPDATE")
	assert.Contains(t, call.query, `COALESCE("LiteLLM_EndUserTable".spend, 0) + EXCLUDED.spend`)
	assert.Equal(t, []interface{}{endUserID, amount}, call.args)
}

func TestAggregateSpendUpdatesPreservesZeroSpendModelAndCompositeIDs(t *testing.T) {
	entry := &models.SpendLogEntry{
		APIKey:         "token:west",
		UserID:         "user:east",
		TeamID:         "team:blue",
		BillingTeamID:  "team:blue",
		OrganizationID: "org:green",
		Model:          "provider:model:v1",
		Spend:          0,
	}

	updates := aggregateSpendUpdates([]*models.SpendLogEntry{entry})

	for _, present := range []bool{
		hasEntityModelKey(updates.Tokens, entityModelKey{EntityID: entry.APIKey, Model: entry.Model}),
		hasEntityModelKey(updates.Teams, entityModelKey{EntityID: entry.BillingTeamID, Model: entry.Model}),
		hasEntityModelKey(updates.Orgs, entityModelKey{EntityID: entry.OrganizationID, Model: entry.Model}),
	} {
		assert.True(t, present, "zero-spend rows must still create their model counter key")
	}
	assert.True(t, hasEntityModelKey(updates.Users, entityModelKey{EntityID: entry.UserID, Model: entry.Model}))
	_, teamMemberPresent := updates.TeamMembers[teamMemberKey{TeamID: entry.BillingTeamID, UserID: entry.UserID}]
	assert.True(t, teamMemberPresent)
	_, organizationMemberPresent := updates.OrganizationMembers[organizationMemberKey{
		OrganizationID: entry.OrganizationID,
		UserID:         entry.UserID,
	}]
	assert.True(t, organizationMemberPresent)
}

// TestAggregateSpendUpdates_BillingTeamIDDrivesRouting reproduces the vsellm
// incident and its neighboring cases: credential_name_as_team_id substitutes
// the provider credential name into TeamID for log/DailyTeamSpend attribution
// on keys that have no real team. That synthetic TeamID must never be mistaken
// for a real team billing entity, in any combination with UserID/OrgID.
func TestAggregateSpendUpdates_BillingTeamIDDrivesRouting(t *testing.T) {
	tests := []struct {
		name string

		userID        string
		teamID        string // raw attribution value written to the log
		billingTeamID string // real team, drives the routing decision
		orgID         string
		spend         float64

		wantUserUpdated       bool
		wantTeamCharged       bool
		wantTeamMemberCharged bool
		wantOrgMemberCharged  bool
	}{
		{
			name:            "no team",
			userID:          "user-1",
			teamID:          "",
			billingTeamID:   "",
			spend:           10,
			wantUserUpdated: true,
		},
		{
			name:            "synthetic team attribution",
			userID:          "user-1",
			teamID:          "air-ru01", // credential_name_as_team_id fallback
			billingTeamID:   "",         // token has no real team
			spend:           10,
			wantUserUpdated: true,
		},
		{
			name:                  "real team",
			userID:                "user-1",
			teamID:                "team-1",
			billingTeamID:         "team-1",
			spend:                 10,
			wantUserUpdated:       true,
			wantTeamCharged:       true,
			wantTeamMemberCharged: true,
		},
		{
			name:                  "real team with credential attribution enabled",
			userID:                "user-1",
			teamID:                "team-1", // TokenInfo.TeamID was non-empty, so proxy_log.go never substitutes credName
			billingTeamID:         "team-1",
			spend:                 10,
			wantUserUpdated:       true,
			wantTeamCharged:       true,
			wantTeamMemberCharged: true,
		},
		{
			name:                  "real team with organization membership",
			userID:                "user-1",
			teamID:                "team-1",
			billingTeamID:         "team-1",
			orgID:                 "org-1",
			spend:                 10,
			wantUserUpdated:       true,
			wantTeamCharged:       true,
			wantTeamMemberCharged: true,
			wantOrgMemberCharged:  true,
		},
		{
			name:                 "synthetic team attribution with organization membership",
			userID:               "user-1",
			teamID:               "air-ru01",
			billingTeamID:        "",
			orgID:                "org-1",
			spend:                10,
			wantUserUpdated:      true,
			wantOrgMemberCharged: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batch := []*models.SpendLogEntry{{
				APIKey:         "token-1",
				UserID:         tt.userID,
				TeamID:         tt.teamID,
				BillingTeamID:  tt.billingTeamID,
				OrganizationID: tt.orgID,
				Model:          "model-1",
				Spend:          tt.spend,
			}}

			result := aggregateSpendUpdates(batch)

			userKey := entityModelKey{EntityID: tt.userID, Model: "model-1"}
			if tt.wantUserUpdated {
				assert.Equal(t, tt.spend, result.Users[userKey])
			} else {
				assert.NotContains(t, result.Users, userKey)
			}

			teamKey := entityModelKey{EntityID: tt.billingTeamID, Model: "model-1"}
			if tt.wantTeamCharged {
				assert.Equal(t, tt.spend, result.Teams[teamKey])
			} else {
				assert.Empty(t, result.Teams, "must never charge the synthetic credential-name team")
			}

			teamMemberKey := teamMemberKey{TeamID: tt.billingTeamID, UserID: tt.userID}
			if tt.wantTeamMemberCharged {
				assert.Equal(t, tt.spend, result.TeamMembers[teamMemberKey])
			} else {
				assert.Empty(t, result.TeamMembers, "must never create a membership row for a nonexistent team")
			}

			orgMemberKey := organizationMemberKey{OrganizationID: tt.orgID, UserID: tt.userID}
			if tt.wantOrgMemberCharged {
				assert.Equal(t, tt.spend, result.OrganizationMembers[orgMemberKey])
			} else {
				assert.Empty(t, result.OrganizationMembers)
			}
		})
	}
}

func TestAggregateSpendUpdatesUpdatesUserAndTeamCounters(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey: "token-synthetic",
			UserID: "user-shared",
			TeamID: "air-ru01",
			Model:  "model-1",
			Spend:  7.0,
		},
		{
			APIKey:        "token-team",
			UserID:        "user-shared",
			TeamID:        "team-real",
			BillingTeamID: "team-real",
			Model:         "model-1",
			Spend:         4.0,
		},
	}

	result := aggregateSpendUpdates(batch)

	assert.Equal(t, 7.0, result.Tokens[entityModelKey{EntityID: "token-synthetic", Model: "model-1"}])
	assert.Equal(t, 4.0, result.Tokens[entityModelKey{EntityID: "token-team", Model: "model-1"}])
	assert.Len(t, result.Tokens, 2)

	assert.Equal(t, 11.0, result.Users[entityModelKey{EntityID: "user-shared", Model: "model-1"}])
	assert.Len(t, result.Users, 1)

	assert.NotContains(t, result.Teams, entityModelKey{EntityID: "air-ru01", Model: "model-1"})
	assert.Equal(t, 4.0, result.Teams[entityModelKey{EntityID: "team-real", Model: "model-1"}])
	assert.Len(t, result.Teams, 1)

	assert.Equal(t, 4.0, result.TeamMembers[teamMemberKey{TeamID: "team-real", UserID: "user-shared"}])
	assert.Len(t, result.TeamMembers, 1)
}

func hasEntityModelKey(updates map[entityModelKey]float64, key entityModelKey) bool {
	_, ok := updates[key]
	return ok
}

func TestEntityUpdatesPersistSpendAndNumericModelSpendTogether(t *testing.T) {
	const (
		model  = "provider:model:{v1}"
		entity = "tenant:west"
	)

	previousTokenLastActive := schemaTokenHasLastActive.Load()
	schemaTokenHasLastActive.Store(true)
	t.Cleanup(func() { schemaTokenHasLastActive.Store(previousTokenLastActive) })

	tests := []struct {
		name      string
		table     string
		extraSQL  string
		runUpdate func(*recordingSpendUpdateExecer) error
	}{
		{
			name:     "verification token",
			table:    `"LiteLLM_VerificationToken"`,
			extraSQL: "last_active = NOW()",
			runUpdate: func(execer *recordingSpendUpdateExecer) error {
				return updateTokens(context.Background(), execer, map[entityModelKey]float64{
					{EntityID: entity, Model: model}: 0,
				})
			},
		},
		{
			name:  "user",
			table: `"LiteLLM_UserTable"`,
			runUpdate: func(execer *recordingSpendUpdateExecer) error {
				return updateUsers(context.Background(), execer, map[entityModelKey]float64{
					{EntityID: entity, Model: model}: 0,
				})
			},
		},
		{
			name:  "team",
			table: `"LiteLLM_TeamTable"`,
			runUpdate: func(execer *recordingSpendUpdateExecer) error {
				return updateTeams(context.Background(), execer, map[entityModelKey]float64{
					{EntityID: entity, Model: model}: 0,
				})
			},
		},
		{
			name:  "organization",
			table: `"LiteLLM_OrganizationTable"`,
			runUpdate: func(execer *recordingSpendUpdateExecer) error {
				return updateOrgs(context.Background(), execer, map[entityModelKey]float64{
					{EntityID: entity, Model: model}: 0,
				})
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			execer := &recordingSpendUpdateExecer{}
			require.NoError(t, tt.runUpdate(execer))
			require.Len(t, execer.calls, 1)

			call := execer.calls[0]
			assert.Contains(t, call.query, "UPDATE "+tt.table)
			assert.Contains(t, call.query, "SET spend = COALESCE(spend, 0) + $1")
			assert.Contains(t, call.query, "model_spend = jsonb_set")
			assert.Contains(t, call.query, "ARRAY[$2]::text[]")
			assert.Contains(t, call.query, "to_jsonb")
			assert.Contains(t, call.query, "updated_at = NOW()")
			if tt.extraSQL != "" {
				assert.Contains(t, call.query, tt.extraSQL)
			}
			assert.Equal(t, []interface{}{float64(0), model, entity}, call.args)
		})
	}
}

func TestMembershipUpdatesKeepColonContainingCompositeIDsSeparate(t *testing.T) {
	previousTeamTotalSpend := schemaTeamMemberHasTotalSpend.Load()
	schemaTeamMemberHasTotalSpend.Store(true)
	t.Cleanup(func() { schemaTeamMemberHasTotalSpend.Store(previousTeamTotalSpend) })

	teamExecer := &recordingSpendUpdateExecer{}
	require.NoError(t, updateTeamMembers(context.Background(), teamExecer, map[teamMemberKey]float64{
		{TeamID: "team:west", UserID: "user:east"}: 1.25,
	}))
	require.Len(t, teamExecer.calls, 1)
	assert.Equal(t, []interface{}{1.25, "team:west", "user:east"}, teamExecer.calls[0].args)

	organizationExecer := &recordingSpendUpdateExecer{}
	require.NoError(t, updateOrganizationMembers(
		context.Background(),
		organizationExecer,
		map[organizationMemberKey]float64{
			{OrganizationID: "org:west", UserID: "user:east"}: 2.5,
		},
	))
	require.Len(t, organizationExecer.calls, 1)
	call := organizationExecer.calls[0]
	assert.Contains(t, call.query, `UPDATE "LiteLLM_OrganizationMembership"`)
	assert.Contains(t, call.query, "updated_at = NOW()")
	assert.Contains(t, call.query, "organization_id = $2 AND user_id = $3")
	assert.Equal(t, []interface{}{2.5, "org:west", "user:east"}, call.args)
}

// TestFilterBatchByInsertedIDs tests filtering batch by inserted IDs
func TestFilterBatchByInsertedIDs(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{RequestID: "req-1"},
		{RequestID: "req-2"},
		{RequestID: "req-3"},
		{RequestID: "req-4"},
	}

	insertedIDs := []string{"req-1", "req-3"}

	result := filterBatchByInsertedIDs(batch, insertedIDs)

	assert.Len(t, result, 2)
	assert.Equal(t, "req-1", result[0].entry.RequestID)
	assert.Equal(t, "req-1", result[0].requestID)
	assert.Equal(t, "req-3", result[1].entry.RequestID)
	assert.Equal(t, "req-3", result[1].requestID)
}

// TestFilterBatchByInsertedIDs_Empty tests empty inserted IDs
func TestFilterBatchByInsertedIDs_Empty(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{RequestID: "req-1"},
		{RequestID: "req-2"},
	}

	result := filterBatchByInsertedIDs(batch, []string{})

	assert.Nil(t, result)
}

// TestFilterBatchByInsertedIDs_NilBatch tests nil batch - function doesn't handle nil
func TestFilterBatchByInsertedIDs_NilBatch(t *testing.T) {
	// Note: function will panic on nil batch, so this test verifies it returns empty for empty batch
	result := filterBatchByInsertedIDs([]*models.SpendLogEntry{}, []string{"req-1"})
	assert.Empty(t, result)
}

// TestFilterBatchByInsertedIDs_AllMatch tests when all IDs match
func TestFilterBatchByInsertedIDs_AllMatch(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{RequestID: "req-1"},
		{RequestID: "req-2"},
	}

	insertedIDs := []string{"req-1", "req-2", "req-3"}

	result := filterBatchByInsertedIDs(batch, insertedIDs)

	assert.Len(t, result, 2)
}

// TestFilterBatchByInsertedIDs_NoMatch tests when no IDs match
func TestFilterBatchByInsertedIDs_NoMatch(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{RequestID: "req-1"},
		{RequestID: "req-2"},
	}

	insertedIDs := []string{"req-99", "req-100"}

	result := filterBatchByInsertedIDs(batch, insertedIDs)

	assert.Len(t, result, 0)
}

// TestSortedKeys_DeterministicAcrossRuns pins the fix for the multi-pod
// deadlock: two concurrent transactions updating the same set of rows (e.g.
// two teams) must always take their row locks in the same order, regardless
// of which process built the map or Go's randomized map iteration. If this
// ever regresses to `for k := range m`, this test becomes flaky (different
// order per run) instead of failing outright - run with -count=20 to catch
// that class of regression.
func TestSortedKeys_DeterministicAcrossRuns(t *testing.T) {
	m := map[string]float64{
		"team-zebra": 1,
		"team-alpha": 2,
		"team-mike":  3,
		"team-echo":  4,
	}

	want := []string{"team-alpha", "team-echo", "team-mike", "team-zebra"}

	for range 20 {
		assert.Equal(t, want, sortedKeys(m))
	}
}

// TestSortedKeys_MatchesAcrossIndependentMaps simulates two "pods" that
// aggregated overlapping entities from different batches (different Go map
// instances, different insertion order). Both must still update in the same
// global order, which is what actually prevents the lock-order deadlock.
func TestSortedKeys_MatchesAcrossIndependentMaps(t *testing.T) {
	podA := map[string]float64{"team-b": 10, "team-a": 5, "team-c": 1}
	podB := map[string]float64{"team-c": 2, "team-a": 7, "team-b": 3}

	assert.Equal(t, sortedKeys(podA), sortedKeys(podB))
}

func TestSortedKeys_Empty(t *testing.T) {
	assert.Empty(t, sortedKeys(nil))
	assert.Empty(t, sortedKeys(map[string]float64{}))
}
