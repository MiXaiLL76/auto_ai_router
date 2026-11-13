package upstreamerror

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClassificationFixtures is the baseline: every shared fixture classifies
// correctly through ExtractErrorSignals + ClassifyProviderErrorSignals
// directly. The proxy and LiteLLM packages run the same table through their
// own entry points (statusCodeFromProviderBodyError, errorStatus) -- see
// their own TestClassificationFixtures in each package.
func TestClassificationFixtures(t *testing.T) {
	for _, fx := range ClassificationFixtures {
		t.Run(fx.Name, func(t *testing.T) {
			signals, ok := ExtractErrorSignals([]byte(fx.Body))
			require.True(t, ok, "fixture body must yield at least one signal")
			assert.Equal(t, fx.WantStatus, ClassifyProviderErrorSignals(signals))
		})
	}
}
