package litellm

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/upstreamerror"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClassificationFixtures runs upstreamerror.ClassificationFixtures
// through this package's own entry point, errorStatus. The proxy package
// (internal/proxy) runs the exact same table through its own entry point,
// statusCodeFromProviderBodyError -- see upstreamerror.ClassificationFixtures'
// own doc comment for why one shared table backs both.
func TestClassificationFixtures(t *testing.T) {
	for _, fx := range upstreamerror.ClassificationFixtures {
		t.Run(fx.Name, func(t *testing.T) {
			var value any
			require.NoError(t, json.Unmarshal([]byte(fx.Body), &value))
			assert.Equal(t, fx.WantStatus, errorStatus(value))
		})
	}
}
