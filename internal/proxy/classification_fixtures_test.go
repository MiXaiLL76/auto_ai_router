package proxy

import (
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/upstreamerror"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClassificationFixtures runs upstreamerror.ClassificationFixtures
// through this package's own entry point, statusCodeFromProviderBodyError,
// wrapping each fixture body as the "error" value of an HTTP 200 response --
// the shape a provider that answers 200 with the failure in the body sends.
//
// The LiteLLM compatibility layer (internal/responsecompat/litellm) runs the
// exact same table through its own entry point, errorStatus. Both packages
// now delegate to upstreamerror.ExtractErrorSignals/ClassifyProviderErrorSignals,
// so a mismatch here would mean one of them stopped doing that -- see
// upstreamerror.ClassificationFixtures' own doc comment for why this
// shared-table test exists.
func TestClassificationFixtures(t *testing.T) {
	for _, fx := range upstreamerror.ClassificationFixtures {
		t.Run(fx.Name, func(t *testing.T) {
			body := []byte(`{"error":` + fx.Body + `}`)
			status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
			require.True(t, ok, "fixture body must be recognized as an error")
			assert.Equal(t, fx.WantStatus, status)
		})
	}
}
