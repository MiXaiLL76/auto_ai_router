package yandexsearch

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valkey-io/valkey-go"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

// reserverForTest mirrors internal/litellmdb/budget's own integration-test
// convention: a real Reserver over VALKEY_ADDR, skipped when unset.
func reserverForTest(t *testing.T, prefix string) *budget.Reserver {
	t.Helper()
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	client, err := valkey.NewClient(valkey.ClientOption{
		InitAddress:       []string{addr},
		ForceSingleClient: true,
	})
	require.NoError(t, err)
	t.Cleanup(client.Close)
	return budget.New(client, prefix+strconv.FormatInt(time.Now().UnixNano(), 10)+":", time.Minute)
}

func float64p(v float64) *float64 { return &v }

func TestBudgetLevelsForBuildsFullHierarchy(t *testing.T) {
	info := &dbmodels.TokenInfo{
		Token:          "tok1",
		MaxBudget:      float64p(10),
		Spend:          1,
		UserID:         "user1",
		TeamID:         "team1",
		OrganizationID: "org1",
		UserMaxBudget:  float64p(20),
		TeamMaxBudget:  float64p(30),
		OrgMaxBudget:   float64p(40),
	}
	levels := budgetLevelsFor(info)

	entities := make(map[string]budgetLevel, len(levels))
	for _, l := range levels {
		entities[l.entity] = l
	}
	assert.Contains(t, entities, "token:tok1")
	assert.Contains(t, entities, "team:team1")
	assert.Contains(t, entities, "org:org1")
	assert.Contains(t, entities, "teammember:team1:user1")
	assert.Contains(t, entities, "orgmember:org1:user1")
	// "user:" level is only added when TeamID == "" — here it's set, so a
	// bare "user:" entity must NOT appear (matches budget_enforcement.go).
	assert.NotContains(t, entities, "user:user1")
}

func TestBudgetLevelsForPersonalKeyWithoutTeam(t *testing.T) {
	info := &dbmodels.TokenInfo{
		Token:         "tok1",
		UserID:        "user1",
		UserMaxBudget: float64p(20),
	}
	levels := budgetLevelsFor(info)
	entities := make(map[string]bool, len(levels))
	for _, l := range levels {
		entities[l.entity] = true
	}
	assert.True(t, entities["user:user1"])
}

func TestBudgetReserveAllowsUnderBudgetAndReconcilesExactMatch(t *testing.T) {
	r := reserverForTest(t, "test:yandexsearch:ok:")
	ctx := context.Background()
	info := &dbmodels.TokenInfo{Token: "tok-ok", MaxBudget: float64p(100), Spend: 0}

	res, err := budgetReserve(ctx, r, info, 0.01, nil)
	require.NoError(t, err)
	require.NotNil(t, res)

	// Settling at the same estimate (the common flat-price case) must not error.
	res.settle(ctx, r, 0.01, nil)

	// A second settle call must be a no-op (idempotency guard).
	res.settle(ctx, r, 99, nil)
}

func TestBudgetReserveRejectsOverBudgetAndRollsBack(t *testing.T) {
	r := reserverForTest(t, "test:yandexsearch:reject:")
	ctx := context.Background()
	info := &dbmodels.TokenInfo{Token: "tok-reject", MaxBudget: float64p(0.005), Spend: 0.003}

	res, err := budgetReserve(ctx, r, info, 0.01, nil)
	require.ErrorIs(t, err, dbmodels.ErrBudgetExceeded)
	refusal, ok := dbmodels.AsBudgetExceeded(err)
	require.True(t, ok)
	assert.Equal(t, dbmodels.BudgetLevelKey, refusal.Level)
	assert.Equal(t, "key budget exceeded", budgetExceededMessage(err))
	assert.Nil(t, res)
}

// The key's budget windows are reserved against like in the LLM path, on the
// same counters (internal/proxy budgetWindowLevels names them identically).
func TestBudgetLevelsForIncludeTheKeyBudgetWindows(t *testing.T) {
	start := time.Date(2026, 3, 10, 21, 0, 0, 0, time.UTC)
	levels := budgetLevelsFor(&dbmodels.TokenInfo{
		Token: "tok",
		BudgetWindows: []dbmodels.BudgetWindow{
			{BudgetLimit: dbmodels.BudgetLimit{Duration: "1mo", MaxBudget: 3}, Start: start, Spend: 1},
			{BudgetLimit: dbmodels.BudgetLimit{Duration: "2d", MaxBudget: 1}},
		},
	})
	require.Len(t, levels, 2)
	assert.Equal(t, "token:tok:window:1mo:"+strconv.FormatInt(start.Unix(), 10), levels[1].entity)
	assert.Equal(t, "1mo", levels[1].window)
	assert.Equal(t, 3.0, *levels[1].maxBudget)
	assert.Equal(t, 1.0, levels[1].dbSpend)
}

func TestBudgetReserveRefusesOnAKeyBudgetWindow(t *testing.T) {
	r := reserverForTest(t, "test:yandexsearch:window:")
	info := &dbmodels.TokenInfo{Token: "tok-window", BudgetWindows: []dbmodels.BudgetWindow{{
		BudgetLimit: dbmodels.BudgetLimit{Duration: "1d", MaxBudget: 0.005},
		Start:       time.Date(2026, 3, 10, 21, 0, 0, 0, time.UTC),
		Spend:       0.003,
	}}}

	res, err := budgetReserve(context.Background(), r, info, 0.01, nil)
	require.ErrorIs(t, err, dbmodels.ErrBudgetExceeded)
	assert.Equal(t, "key 1d budget exceeded", budgetExceededMessage(err))
	assert.Nil(t, res)
}

func TestBudgetReserveRollsBackTheKeyWhenAParentLevelRefuses(t *testing.T) {
	r := reserverForTest(t, "test:yandexsearch:rollback:")
	ctx := context.Background()
	info := &dbmodels.TokenInfo{
		Token: "tok-rollback", MaxBudget: float64p(0.05), Spend: 0,
		UserID: "user-rollback", UserMaxBudget: float64p(0.005), UserSpend: float64p(0.003),
	}

	res, err := budgetReserve(ctx, r, info, 0.01, nil)
	require.ErrorIs(t, err, dbmodels.ErrBudgetExceeded)
	refusal, ok := dbmodels.AsBudgetExceeded(err)
	require.True(t, ok)
	assert.Equal(t, dbmodels.BudgetLevelUser, refusal.Level)
	assert.Equal(t, "budget exceeded", budgetExceededMessage(err))
	assert.Nil(t, res)

	// The key was reserved before the user refused; the rollback returns its
	// counter to the DB spend, so the key's whole budget still fits.
	allowed, err := r.TryReserve(ctx, "token:tok-rollback", 0, 0.05, 0.05)
	require.NoError(t, err)
	assert.True(t, allowed)
}

func TestBudgetReserveSkipsUnlimitedLevels(t *testing.T) {
	r := reserverForTest(t, "test:yandexsearch:unlimited:")
	ctx := context.Background()
	// No MaxBudget set anywhere (nil) -> every level is "unlimited", so
	// reservation must trivially succeed with zero reserved entities.
	info := &dbmodels.TokenInfo{Token: "tok-unlimited"}

	res, err := budgetReserve(ctx, r, info, 0.01, nil)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Empty(t, res.entities)
}

func TestBudgetReserveNilReserverAlwaysAllows(t *testing.T) {
	ctx := context.Background()
	info := &dbmodels.TokenInfo{Token: "tok-nil", MaxBudget: float64p(0), Spend: 100}

	// A nil *budget.Reserver degrades to always-allow/no-op settle, per
	// budget.Reserver's own nil-receiver safety (TestNilReserver_NoOp).
	res, err := budgetReserve(ctx, nil, info, 0.01, nil)
	require.NoError(t, err)
	require.NotNil(t, res)
	res.settle(ctx, nil, 0.01, nil)
}

func TestReservationSettleNilReceiverIsNoop(t *testing.T) {
	var res *reservation
	res.settle(context.Background(), nil, 1, nil) // must not panic
}

func TestBudgetExceededMessageNamesOnlyTheKeyLimit(t *testing.T) {
	assert.Equal(t, "key budget exceeded",
		budgetExceededMessage(&dbmodels.BudgetExceededError{Level: dbmodels.BudgetLevelKey}))
	assert.Equal(t, "key 1d budget exceeded",
		budgetExceededMessage(&dbmodels.BudgetExceededError{Level: dbmodels.BudgetLevelKey, Window: "1d"}))
	for _, level := range []dbmodels.BudgetLevel{
		dbmodels.BudgetLevelUser, dbmodels.BudgetLevelTeam, dbmodels.BudgetLevelTeamMember,
		dbmodels.BudgetLevelOrganization, dbmodels.BudgetLevelOrganizationMember,
	} {
		assert.Equal(t, "budget exceeded", budgetExceededMessage(&dbmodels.BudgetExceededError{Level: level}))
	}
	assert.Equal(t, "budget exceeded", budgetExceededMessage(dbmodels.ErrBudgetExceeded))
}
