package config

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestCredentialRoutingIDYAML(t *testing.T) {
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte(`name: account-a
type: openai
api_key: example
base_url: https://example.com
proxy_url: http://localhost:3128
routing_id: route-01`), &cred)
	require.NoError(t, err)
	require.Equal(t, "route-01", cred.RoutingID)
	require.NoError(t, cred.validate(0))
}

func TestCredentialRoutingIDValidation(t *testing.T) {
	direct := CredentialConfig{
		Name: "account-a", Type: ProviderTypeOpenAI, APIKey: "example",
		BaseURL: "https://example.com", RPM: -1, TPM: -1, RoutingID: "route-01",
	}
	require.NoError(t, direct.validate(0))

	bad := direct
	bad.RoutingID = "provider-name"
	require.ErrorContains(t, bad.validate(0), "routing_id must be")
	for _, typ := range []ProviderType{ProviderTypeAIR, ProviderTypeProxy} {
		peer := direct
		peer.Type = typ
		require.ErrorContains(t, peer.validate(0), "routing_id requires a direct credential")
	}
	duplicate := direct
	duplicate.Name = "account-b"
	cfg := Config{Credentials: []CredentialConfig{direct, duplicate}}
	require.ErrorContains(t, cfg.validateCredentials(), "routing_id is already assigned")
}

func TestCredentialVisibleToRoutingRestriction(t *testing.T) {
	cred := CredentialConfig{Type: ProviderTypeOpenAI, RoutingID: "route-01", Scopes: []string{"team-a"}}
	team := scope.NewContext([]string{"team-a"}, nil)
	require.True(t, cred.VisibleTo(team))
	require.True(t, cred.VisibleTo(team.RestrictCredentialIDs([]string{"route-01"})))
	require.False(t, cred.VisibleTo(team.RestrictCredentialIDs([]string{"route-02"})))
	require.False(t, cred.VisibleTo(team.RestrictCredentialIDs(nil)))
	require.False(t, cred.VisibleTo(scope.PublicContext().RestrictCredentialIDs([]string{"route-01"})))

	cred.Type = ProviderTypeAIR
	require.False(t, cred.VisibleTo(team.RestrictCredentialIDs([]string{"route-01"})))
	require.True(t, cred.VisibleTo(team))
}
