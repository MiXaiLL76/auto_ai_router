package balancer

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/require"
)

func TestRetryHistoryExcludedPriorityDoesNotCountAsAttempt(t *testing.T) {
	for _, excludedHasModel := range []bool{false, true} {
		for _, scoped := range []bool{false, true} {
			name := "excluded priority"
			if excludedHasModel {
				name += "/excluded credential has model"
			} else {
				name += "/excluded credential lacks model"
			}
			if scoped {
				name += "/scoped"
			} else {
				name += "/unscoped"
			}
			t.Run(name, func(t *testing.T) {
				credentials := []config.CredentialConfig{
					{Name: "cheap", Type: config.ProviderTypeOpenAI, APIKey: "test", BaseURL: "http://cheap.test", RPM: 100},
					{Name: "excluded", Type: config.ProviderTypeOpenAI, APIKey: "test", BaseURL: "http://excluded.test", RPM: 100, FallbackPriority: 150},
					{Name: "healthy", Type: config.ProviderTypeOpenAI, APIKey: "test", BaseURL: "http://healthy.test", RPM: 100, FallbackPriority: 1},
				}
				bal := New(credentials, fail2ban.New(3, 0, []int{401, 403, 500}), ratelimit.New())
				checker := NewMockModelChecker(true)
				checker.AddModel("cheap", "model")
				checker.AddModel("healthy", "model")
				if excludedHasModel {
					checker.AddModel("excluded", "model")
				}
				bal.SetModelChecker(checker)
				exclude := map[string]bool{"cheap": true, "excluded": true}
				attempted := map[string]bool{"cheap": true}
				var selected *config.CredentialConfig
				var err error
				if scoped {
					selected, err = bal.NextRetryForModelExcludingScoped("model", &credentials[0], exclude, attempted, scope.PublicContext())
				} else {
					selected, err = bal.NextRetryForModelExcluding("model", &credentials[0], exclude, attempted)
				}
				require.NoError(t, err)
				require.Equal(t, "healthy", selected.Name)
				require.Equal(t, map[string]bool{"cheap": true, "excluded": true}, exclude)
				require.Equal(t, map[string]bool{"cheap": true}, attempted)
			})
		}
	}
}

func TestRetryHistoryActuallyTriedPriorityKeepsUnprioritizedTail(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "priority-start", Type: config.ProviderTypeAnthropic, APIKey: "test", BaseURL: "http://priority.test", RPM: 100, FallbackPriority: 10},
		{Name: "cheap", Type: config.ProviderTypeOpenAI, APIKey: "test", BaseURL: "http://cheap.test", RPM: 100},
		{Name: "unused-priority", Type: config.ProviderTypeOpenAI, APIKey: "test", BaseURL: "http://unused.test", RPM: 100, FallbackPriority: 20},
		{Name: "tail", Type: config.ProviderTypeAnthropic, APIKey: "test", BaseURL: "http://tail.test", RPM: 100},
	}
	bal := New(credentials, fail2ban.New(3, 0, []int{401, 403, 500}), ratelimit.New())
	exclude := map[string]bool{"priority-start": true, "cheap": true}
	attempted := map[string]bool{"priority-start": true, "cheap": true}
	selected, err := bal.NextRetryForModelExcluding("", &credentials[1], exclude, attempted)
	require.NoError(t, err)
	require.Equal(t, "tail", selected.Name)
}
