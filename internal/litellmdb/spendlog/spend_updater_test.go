package spendlog

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
)

func TestAggregateSpendUpdates(t *testing.T) {
	t.Run("aggregates tokens by api key", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "key-a", Spend: 10.0},
			{APIKey: "key-a", Spend: 5.0},
			{APIKey: "key-b", Spend: 3.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 15.0, updates.Tokens["key-a"])
		assert.Equal(t, 3.0, updates.Tokens["key-b"])
		assert.Len(t, updates.Tokens, 2)
	})

	t.Run("aggregates users by user id", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", UserID: "user-1", Spend: 8.0},
			{APIKey: "k2", UserID: "user-1", Spend: 7.0},
			{APIKey: "k3", UserID: "user-2", Spend: 4.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 15.0, updates.Users["user-1"])
		assert.Equal(t, 4.0, updates.Users["user-2"])
		assert.Len(t, updates.Users, 2)
	})

	t.Run("skips users when user id is empty", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", UserID: "", Spend: 10.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Len(t, updates.Users, 0)
		assert.Equal(t, 10.0, updates.Tokens["k1"])
	})

	t.Run("aggregates teams by team id", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", TeamID: "team-1", Spend: 5.0},
			{APIKey: "k2", TeamID: "team-1", Spend: 3.0},
			{APIKey: "k3", TeamID: "team-2", Spend: 2.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 8.0, updates.Teams["team-1"])
		assert.Equal(t, 2.0, updates.Teams["team-2"])
	})

	t.Run("aggregates organizations by org id", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", OrganizationID: "org-1", Spend: 6.0},
			{APIKey: "k2", OrganizationID: "org-1", Spend: 4.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 10.0, updates.Orgs["org-1"])
		assert.Len(t, updates.Orgs, 1)
	})

	t.Run("aggregates team members when user and team present", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", UserID: "user-1", TeamID: "team-1", Spend: 5.0},
			{APIKey: "k2", UserID: "user-1", TeamID: "team-1", Spend: 3.0},
			{APIKey: "k3", UserID: "user-2", TeamID: "team-1", Spend: 2.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 8.0, updates.TeamMembers["team-1:user-1"])
		assert.Equal(t, 2.0, updates.TeamMembers["team-1:user-2"])
		assert.Len(t, updates.TeamMembers, 2)
	})

	t.Run("aggregates org members when user and org present", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", UserID: "user-1", OrganizationID: "org-1", Spend: 4.0},
			{APIKey: "k2", UserID: "user-1", OrganizationID: "org-1", Spend: 6.0},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 10.0, updates.OrgMembers["org-1:user-1"])
		assert.Len(t, updates.OrgMembers, 1)
	})

	t.Run("no team members when user or team missing", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{APIKey: "k1", UserID: "user-1", Spend: 5.0},        // no team
			{APIKey: "k2", TeamID: "team-1", Spend: 3.0},        // no user
			{APIKey: "k3", OrganizationID: "org-1", Spend: 2.0}, // no user, no team
		}

		updates := aggregateSpendUpdates(batch)

		assert.Len(t, updates.TeamMembers, 0)
		assert.Len(t, updates.OrgMembers, 0)
	})

	t.Run("empty batch returns empty maps", func(t *testing.T) {
		updates := aggregateSpendUpdates([]*models.SpendLogEntry{})

		assert.Len(t, updates.Tokens, 0)
		assert.Len(t, updates.Users, 0)
		assert.Len(t, updates.Teams, 0)
		assert.Len(t, updates.Orgs, 0)
		assert.Len(t, updates.TeamMembers, 0)
		assert.Len(t, updates.OrgMembers, 0)
	})

	t.Run("full hierarchy entry populates all maps", func(t *testing.T) {
		batch := []*models.SpendLogEntry{
			{
				APIKey:         "key-1",
				UserID:         "user-1",
				TeamID:         "team-1",
				OrganizationID: "org-1",
				Spend:          12.5,
			},
		}

		updates := aggregateSpendUpdates(batch)

		assert.Equal(t, 12.5, updates.Tokens["key-1"])
		assert.Equal(t, 12.5, updates.Users["user-1"])
		assert.Equal(t, 12.5, updates.Teams["team-1"])
		assert.Equal(t, 12.5, updates.Orgs["org-1"])
		assert.Equal(t, 12.5, updates.TeamMembers["team-1:user-1"])
		assert.Equal(t, 12.5, updates.OrgMembers["org-1:user-1"])
	})
}
