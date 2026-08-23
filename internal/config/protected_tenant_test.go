package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestProtectedTenantConfigYAML(t *testing.T) {
	var cfg Config
	require.NoError(t, yaml.Unmarshal([]byte(`
protected_tenants:
  - name: cloud-ru
    organization_ids: [org-cloud]
    hostnames: [API.CLOUD.EXAMPLE.:443]
    required_scopes: [cloud-ru]
    require_model_acl: true
    require_route_acl: true
`), &cfg))
	require.Len(t, cfg.ProtectedTenants, 1)
	policy := cfg.ProtectedTenants[0]
	policy.normalize()
	assert.Equal(t, "cloud-ru", policy.Name)
	assert.Equal(t, []string{"org-cloud"}, policy.OrganizationIDs)
	assert.Equal(t, []string{"api.cloud.example"}, policy.Hostnames)
	assert.Equal(t, []string{"cloud-ru"}, policy.RequiredScopes)
	assert.True(t, policy.RequireModelACL)
	assert.True(t, policy.RequireRouteACL)
}

func TestProtectedTenantConfigValidation(t *testing.T) {
	t.Run("organization required", func(t *testing.T) {
		policy := ProtectedTenantConfig{Name: "cloud-ru"}
		assert.ErrorContains(t, policy.validate(), "organization_id")
	})

	t.Run("name required", func(t *testing.T) {
		policy := ProtectedTenantConfig{OrganizationIDs: []string{"org-cloud"}}
		assert.ErrorContains(t, policy.validate(), "name")
	})
}
