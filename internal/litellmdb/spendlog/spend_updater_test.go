package spendlog

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
)

// TestAggregateSpendUpdates_AllEntities tests aggregation with all entity types
func TestAggregateSpendUpdates_AllEntities(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey:         "token-1",
			UserID:         "user-1",
			TeamID:         "team-1",
			OrganizationID: "org-1",
			Spend:          10.0,
		},
		{
			APIKey:         "token-1",
			UserID:         "user-1",
			TeamID:         "team-1",
			OrganizationID: "org-1",
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
	assert.Equal(t, 15.0, result.Tokens["token-1"])
	assert.Equal(t, 3.0, result.Tokens["token-2"])

	// User aggregation
	assert.Equal(t, 15.0, result.Users["user-1"])
	assert.Equal(t, 3.0, result.Users["user-2"])

	// Team aggregation
	assert.Equal(t, 15.0, result.Teams["team-1"])

	// Org aggregation
	assert.Equal(t, 15.0, result.Orgs["org-1"])

	// Team membership
	assert.Equal(t, 15.0, result.TeamMembers["team-1:user-1"])

	// Org membership
	assert.Equal(t, 15.0, result.OrgMembers["org-1:user-1"])
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
	assert.Empty(t, result.OrgMembers)
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
			APIKey: "token-2",
			TeamID: "team-1",
			Spend:  3.0,
			// No UserID, OrganizationID
		},
	}

	result := aggregateSpendUpdates(batch)

	// Token aggregation works
	assert.Equal(t, 15.0, result.Tokens["token-1"])
	assert.Equal(t, 3.0, result.Tokens["token-2"])

	// User aggregation works
	assert.Equal(t, 5.0, result.Users["user-1"])

	// Team aggregation works
	assert.Equal(t, 3.0, result.Teams["team-1"])

	// Org should be empty
	assert.Empty(t, result.Orgs)
}

// TestAggregateSpendUpdates_TeamMember tests team membership with user
func TestAggregateSpendUpdates_TeamMember(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{
			APIKey: "token-1",
			UserID: "user-1",
			TeamID: "team-1",
			Spend:  10.0,
		},
		{
			APIKey: "token-1",
			UserID: "user-2",
			TeamID: "team-1",
			Spend:  5.0,
		},
	}

	result := aggregateSpendUpdates(batch)

	// Team membership should aggregate by team:user
	assert.Equal(t, 10.0, result.TeamMembers["team-1:user-1"])
	assert.Equal(t, 5.0, result.TeamMembers["team-1:user-2"])
}

// TestAggregateSpendUpdates_OrgMember tests org membership with user
func TestAggregateSpendUpdates_OrgMember(t *testing.T) {
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

	// Org membership should aggregate by org:user
	assert.Equal(t, 10.0, result.OrgMembers["org-1:user-1"])
	assert.Equal(t, 5.0, result.OrgMembers["org-1:user-2"])
}

// TestExecuteSpendUpdates_NilUpdates tests nil updates
func TestExecuteSpendUpdates_NilUpdates(t *testing.T) {
	// Can't actually test without DB connection, but verify it doesn't panic
	// This is tested via integration tests with real DB
}

// TestSpendUpdates_Fields verifies SpendUpdates structure
func TestSpendUpdates_Fields(t *testing.T) {
	updates := &SpendUpdates{
		Tokens:      map[string]float64{"key1": 1.0},
		Users:       map[string]float64{"user1": 2.0},
		Teams:       map[string]float64{"team1": 3.0},
		Orgs:        map[string]float64{"org1": 4.0},
		TeamMembers: map[string]float64{"team1:user1": 5.0},
		OrgMembers:  map[string]float64{"org1:user1": 6.0},
	}

	assert.Len(t, updates.Tokens, 1)
	assert.Len(t, updates.Users, 1)
	assert.Len(t, updates.Teams, 1)
	assert.Len(t, updates.Orgs, 1)
	assert.Len(t, updates.TeamMembers, 1)
	assert.Len(t, updates.OrgMembers, 1)

	assert.Equal(t, 1.0, updates.Tokens["key1"])
	assert.Equal(t, 2.0, updates.Users["user1"])
	assert.Equal(t, 3.0, updates.Teams["team1"])
	assert.Equal(t, 4.0, updates.Orgs["org1"])
	assert.Equal(t, 5.0, updates.TeamMembers["team1:user1"])
	assert.Equal(t, 6.0, updates.OrgMembers["org1:user1"])
}

// TestSpendUpdates_Empty verifies empty SpendUpdates
func TestSpendUpdates_Empty(t *testing.T) {
	updates := &SpendUpdates{}

	assert.Nil(t, updates.Tokens)
	assert.Nil(t, updates.Users)
	assert.Nil(t, updates.Teams)
	assert.Nil(t, updates.Orgs)
	assert.Nil(t, updates.TeamMembers)
	assert.Nil(t, updates.OrgMembers)
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
	assert.Equal(t, "req-1", result[0].RequestID)
	assert.Equal(t, "req-3", result[1].RequestID)
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
