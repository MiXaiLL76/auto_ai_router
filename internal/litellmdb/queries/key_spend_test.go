package queries

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKeySpendQueriesPreserveUnknownAndLockOnlyTransactionalRead(t *testing.T) {
	for _, query := range []string{QuerySelectKeySpend, QuerySelectKeySpendForUpdate} {
		assert.Contains(t, query, `FROM "LiteLLM_VerificationToken"`)
		assert.Contains(t, query, "token = $1")
		assert.Contains(t, query, "spend IS NOT NULL")
		assert.NotContains(t, query, "COALESCE", "NULL spend must remain unknown")
	}
	assert.NotContains(t, QuerySelectKeySpend, "FOR UPDATE")
	assert.Contains(t, QuerySelectKeySpendForUpdate, "FOR UPDATE")
}
