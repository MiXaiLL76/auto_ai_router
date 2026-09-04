package models

import (
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writePolicyPrices(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(path, []byte(body), 0600))
	return path
}

func testPolicyManager() *Manager {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	credential := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI}
	manager := New(logger, 100, []config.ModelRPMConfig{
		{Name: "route-a", Credential: credential.Name},
		{Name: "route-b", Credential: credential.Name},
	})
	manager.SetModelAliases(map[string]string{"public/a": "route-a"})
	manager.SetPublicModelAliases(map[string]string{"alias/a": "public/a"})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})
	return manager
}

func validPolicyOptions() OrganizationPolicyLoadOptions {
	return OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: false,
	}
}

func TestLoadOrganizationPolicies_RequiresPostgresWriter(t *testing.T) {
	_, err := LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writePolicyPrices(t, `{"public/a":{"input_cost_per_token":0}}`),
	}}, testPolicyManager(), OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: true,
	})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "disable_spend_logs_write=false")
}

func TestLoadOrganizationPolicies_StrictTariffJSON(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "duplicate key", body: `{"public/a":{"input_cost_per_token":0},"public/a":{"output_cost_per_token":0}}`, want: "duplicate JSON key"},
		{name: "unknown field", body: `{"public/a":{"unexpected":1}}`, want: "unknown price field"},
		{name: "null row", body: `{"public/a":null}`, want: "null price row"},
		{name: "empty row", body: `{"public/a":{}}`, want: "empty price row"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
				OrganizationID:  "org-1",
				PriceProfileID:  "profile-1",
				ModelPricesLink: writePolicyPrices(t, tt.body),
				AllowlistSet:    true,
				ModelAllowlist:  []string{"public/a"},
			}}, testPolicyManager(), validPolicyOptions())

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

func TestLoadOrganizationPolicies_AllowlistRequiresExactPrice(t *testing.T) {
	_, err := LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writePolicyPrices(t, `{"route-a":{"input_cost_per_token":0.001}}`),
		AllowlistSet:    true,
		ModelAllowlist:  []string{"public/a"},
	}}, testPolicyManager(), validPolicyOptions())

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no exact profile price")
}

func TestLoadOrganizationPolicies_FreePriceAndScopedCatalog(t *testing.T) {
	manager := testPolicyManager()
	registry, err := LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: writePolicyPrices(t, `{"public/a":{"input_cost_per_token":0},"org/model":{"input_cost_per_token":0.5,"cache_read_input_tokens_free":true}}`),
		ModelMappings:   map[string]string{"org/model": "route-b"},
	}}, manager, validPolicyOptions())
	require.NoError(t, err)
	policy, ok := registry.Policy("org-1")
	require.True(t, ok)

	catalog := manager.GetAllModelsScopedForOrganization(scope.PublicContext(), policy)

	assert.Equal(t, []string{"org/model", "public/a"}, responseModelIDs(catalog))
	resolution, err := manager.ResolveOrganizationModel(policy, "org/model")
	require.NoError(t, err)
	assert.Equal(t, "org/model", resolution.PublicModelID)
	assert.Equal(t, "route-b", resolution.CanonicalModelID)
	assert.Equal(t, "route-b", resolution.ModelID)
	assert.Equal(t, "org/model", resolution.PriceModelID)
	require.NotNil(t, resolution.ModelPrice)
	assert.True(t, resolution.ModelPrice.CacheReadInputTokensFree)
}
