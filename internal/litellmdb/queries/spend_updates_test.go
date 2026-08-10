package queries

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSpendUpdateQueriesInitializeNullSpend(t *testing.T) {
	for _, query := range []string{
		QueryUpdateTokenSpendWithLastActive,
		QueryUpdateTokenModelSpendWithLastActive,
		QueryUpdateTokenSpend,
		QueryUpdateTokenModelSpend,
		QueryUpdateEntitySpendTemplate,
		QueryUpdateEntityModelSpendTemplate,
		QueryUpdateTeamMemberSpendWithTotal,
		QueryUpdateTeamMemberSpend,
		QueryUpdateOrganizationMemberSpend,
	} {
		assert.Contains(t, query, "COALESCE(spend, 0) + $1")
		assert.NotContains(t, query, "spend IS NOT NULL")
	}
}
