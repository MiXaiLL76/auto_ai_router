package proxy

import (
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRoutingProfileMetadataValidation(t *testing.T) {
	for name, value := range map[string]interface{}{
		"null": nil, "string": "pool-a", "object": map[string]interface{}{}, "mixed": []interface{}{"pool-a", 1},
		"empty id": []interface{}{""}, "whitespace": []string{" pool-a"}, "overlong": make([]string, 33),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := routingProfilesFromMetadata(map[string]interface{}{"air_routing_profiles": value})
			require.Error(t, err)
		})
	}
	ids, err := routingProfilesFromMetadata(nil, map[string]interface{}{"unrelated": "value"}, map[string]interface{}{"air_routing_profiles": []interface{}{"a", "b"}}, map[string]interface{}{"air_routing_profiles": []string{"b", "c"}})
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "b", "c"}, ids)
	ids, err = routingProfilesFromMetadata(map[string]interface{}{"air_routing_profiles": []interface{}{}})
	require.NoError(t, err)
	assert.Empty(t, ids)
}

func TestRoutingProfileMetadataInvalidNeverReachesUpstream(t *testing.T) {
	for name, value := range map[string]interface{}{"unknown": []string{"secret-missing"}, "null": nil, "wrong type": "pool-a"} {
		t.Run(name, func(t *testing.T) {
			f := newRoutingProfileFixture(t)
			f.proxy.LiteLLMDB.(*organizationPolicyTestDB).tokens["restricted-key"].Metadata = map[string]interface{}{"air_routing_profiles": value}
			w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-b","messages":[]}`)
			assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
			assert.NotContains(t, w.Body.String(), "secret-missing")
			assert.NotContains(t, w.Body.String(), "provider-a")
			for i := range f.calls {
				assert.Zero(t, f.calls[i].Load())
			}
		})
	}
	f := newRoutingProfileFixture(t)
	f.proxy.routingProfiles = nil
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileKeyTeamOrganizationIntersect(t *testing.T) {
	f := newRoutingProfileFixture(t)
	registry, err := routermodels.LoadRoutingProfiles([]config.RoutingProfileConfig{
		{ID: "pool-a", Rules: []config.RoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a", "provider-b"}}}},
		{ID: "team-rule", Rules: []config.RoutingRuleConfig{{Models: []string{"model-a"}, CredentialDenylist: []string{"provider-a"}}}},
		{ID: "org-rule", Rules: []config.RoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a"}}}},
	}, f.credentials, f.proxy.modelManager)
	require.NoError(t, err)
	f.proxy.routingProfiles = registry
	db := f.proxy.LiteLLMDB.(*organizationPolicyTestDB)
	info := db.tokens["restricted-key"]
	info.TeamMetadata = map[string]interface{}{"air_routing_profiles": []string{"team-rule"}}
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Zero(t, f.calls[0].Load())
	assert.Equal(t, int32(1), f.calls[1].Load())
	assert.Zero(t, f.calls[2].Load())
	info.OrganizationMetadata = map[string]interface{}{"air_routing_profiles": []string{"org-rule"}}
	w = f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[1].Load())
	// An empty key-level binding cannot remove team or organization restrictions.
	info.Metadata = map[string]interface{}{"air_routing_profiles": []string{}}
	w = f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	// A new key assigned to that team inherits the same restrictions from auth.
	newKey := info.Clone()
	newKey.Metadata = nil
	db.tokens["new-key"] = newKey
	w = f.request("new-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	// Another model and another key retain their original routes.
	w = f.request("restricted-key", "/v1/chat/completions", `{"model":"model-b","messages":[]}`)
	assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
	w = f.request("other-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(2), f.calls[2].Load())
}

func TestRoutingProfileRequestMetadataCannotSelectProfile(t *testing.T) {
	f := newRoutingProfileFixture(t)
	// The unrestricted key stays unrestricted even if a client submits a binding.
	w := f.request("other-key", "/v1/chat/completions", `{"model":"model-a","messages":[],"metadata":{"air_routing_profiles":["pool-a"]}}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[2].Load())
	w = f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[],"metadata":{"air_routing_profiles":[]}}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
}

func TestRoutingProfileUpdatedBindingRechecksSticky(t *testing.T) {
	f := newRoutingProfileFixture(t)
	body := `{"model":"model-a","messages":[],"metadata":{"session_id":"same-session"}}`
	w := f.request("restricted-key", "/v1/chat/completions", body)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	// Models the next authentication snapshot after cache expiry/invalidation.
	f.proxy.LiteLLMDB.(*organizationPolicyTestDB).tokens["restricted-key"].Metadata = map[string]interface{}{"air_routing_profiles": []string{"unknown"}}
	w = f.request("restricted-key", "/v1/chat/completions", body)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
	for _, i := range []int{1, 2, 3} {
		assert.Zero(t, f.calls[i].Load())
	}
}
