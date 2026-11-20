package proxy

import (
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRoutingPoolInvalidBindingNeverReachesUpstream(t *testing.T) {
	for name, metadata := range map[string]map[string]interface{}{
		"unknown pool":  poolPolicyMetadata([]string{"model-a"}, "private-missing"),
		"unknown model": poolPolicyMetadata([]string{"private-missing"}, "pool-a"),
		"null":          {models.RoutingModelsField: nil, models.RoutingProfilesField: []string{"pool-a"}},
		"wrong type":    {models.RoutingModelsField: "model-a", models.RoutingProfilesField: []string{"pool-a"}},
	} {
		t.Run(name, func(t *testing.T) {
			f := newRoutingPoolFixture(t)
			f.proxy.LiteLLMDB.(*organizationPolicyTestDB).tokens["restricted-key"].Metadata = metadata
			w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
			assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
			assert.NotContains(t, w.Body.String(), "private-missing")
			assert.NotContains(t, w.Body.String(), "provider-a")
			for i := range f.calls {
				assert.Zero(t, f.calls[i].Load())
			}
		})
	}
}

func TestRoutingPoolRequestMetadataCannotChangePolicy(t *testing.T) {
	f := newRoutingPoolFixture(t)
	w := f.request("other-key", "/v1/chat/completions", `{"model":"model-a","messages":[],"metadata":{"air_routing_models":["model-a"],"air_routing_profiles":["pool-a"]}}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[2].Load())
	w = f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[],"metadata":{"air_routing_models":null,"air_routing_profiles":[]}}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
}

func TestRoutingPoolMissingPoolDoesNotBlockOtherModels(t *testing.T) {
	f := newRoutingPoolFixture(t)
	f.proxy.LiteLLMDB.(*organizationPolicyTestDB).tokens["restricted-key"].Metadata = poolPolicyMetadata([]string{"model-a"}, "missing-pool")
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-b","messages":[]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[2].Load())
}

func TestRoutingPoolParentProfilesCannotWidenStrictKey(t *testing.T) {
	f := newRoutingPoolFixture(t)
	info := f.proxy.LiteLLMDB.(*organizationPolicyTestDB).tokens["restricted-key"]
	info.Metadata = poolPolicyMetadata([]string{"model-a"}, "provider-a")
	info.OrganizationMetadata = map[string]interface{}{models.RoutingProfilesField: []string{"provider-b"}}
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	require.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingPoolRebindingMovesStickyOnlyInsideNewPool(t *testing.T) {
	f := newRoutingPoolFixture(t)
	db := f.proxy.LiteLLMDB.(*organizationPolicyTestDB)
	db.tokens["restricted-key"].Metadata = poolPolicyMetadata([]string{"model-a"}, "provider-a")
	body := `{"model":"model-a","messages":[],"user":"same-session"}`
	w := f.request("restricted-key", "/v1/chat/completions", body)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	db.tokens["restricted-key"].Metadata = poolPolicyMetadata([]string{"model-a"}, "provider-b")
	w = f.request("restricted-key", "/v1/chat/completions", body)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
	assert.Equal(t, int32(1), f.calls[1].Load())
	assert.Zero(t, f.calls[2].Load())
	w = f.request("other-key", "/v1/chat/completions", body)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[2].Load())
}
