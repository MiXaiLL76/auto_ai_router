package proxy

import (
	"testing"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func pointerTo[T any](value T) *T { return &value }

func TestBudgetLevelsFollowLiteLLMHierarchy(t *testing.T) {
	info := &dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", TeamID: "team-1", OrganizationID: "org-1",
		MaxBudget: pointerTo(10.0), TeamMaxBudget: pointerTo(20.0), OrgMaxBudget: pointerTo(30.0),
		TeamMemberMaxBudget: pointerTo(4.0), OrgMemberMaxBudget: pointerTo(5.0),
	}
	levels := budgetLevels(info)
	require.Len(t, levels, 5)
	assert.Equal(t, []string{
		"token:token-hash", "team:team-1", "org:org-1",
		"teammember:team-1:user-1", "orgmember:org-1:user-1",
	}, []string{levels[0].entity, levels[1].entity, levels[2].entity, levels[3].entity, levels[4].entity})
}

func TestBudgetLevelsUsePersonalUserLimitsWithoutTeam(t *testing.T) {
	rpm, tpm := int64(7), int64(800)
	levels := budgetLevels(&dbmodels.TokenInfo{
		Token: "token-hash", UserID: "user-1", UserRPMLimit: &rpm, UserTPMLimit: &tpm,
	})
	require.Len(t, levels, 2)
	assert.Equal(t, "user:user-1", levels[1].entity)
	assert.Equal(t, &rpm, levels[1].rpm)
	assert.Equal(t, &tpm, levels[1].tpm)
}

func TestEstimateCompletionTokensUsesRequestLimitThenSafeDefault(t *testing.T) {
	proxy := &Proxy{defaultEstimatedCompletionTokens: 321}
	assert.Equal(t, 42, proxy.estimateCompletionTokens([]byte(`{"max_completion_tokens":42}`)))
	assert.Equal(t, 321, proxy.estimateCompletionTokens([]byte(`{"messages":[]}`)))
	assert.Equal(t, 321, proxy.estimateCompletionTokens([]byte(`not-json`)))
}
