package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRoutingProfileCannotBypassUsingPerCredentialModelAliases(t *testing.T) {
	f := newRoutingProfileFixture(t)
	var routes []config.ModelRPMConfig
	for i, credential := range f.credentials[:3] {
		for _, name := range []string{"model-a", "alternate-name"} {
			routes = append(routes, config.ModelRPMConfig{Name: name, Model: fmt.Sprintf("deployment-%d", i), Credential: credential.Name})
		}
	}
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, routes)
	manager.LoadModelsFromConfig(f.credentials)
	manager.SetCredentials(f.credentials)
	f.proxy.modelManager = manager
	f.proxy.balancer.SetModelChecker(manager)
	var err error
	f.proxy.routingProfiles, err = routermodels.LoadRoutingProfiles([]config.RoutingProfileConfig{{ID: "pool-a", Rules: []config.RoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a", "provider-b"}}}}}, f.credentials, manager)
	require.NoError(t, err)
	f.proxy.priceRegistry.MergeDB(map[string]*routermodels.ModelPrice{"alternate-name": {}})
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"alternate-name","messages":[]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
	assert.Zero(t, f.calls[2].Load(), "the same model under another deployment name must stay inside the pool")
	w = f.request("other-key", "/v1/chat/completions", `{"model":"alternate-name","messages":[]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[2].Load(), "another key in the same organization keeps ordinary routing")
}

func TestRoutingProfilePreservesOrganizationMappingAndBilling(t *testing.T) {
	f := newRoutingProfileFixture(t)
	registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID: "org", PriceProfileID: "profile-a",
		ModelPricesLink: writeProxyPolicyPrices(t, `{"vendor/model-a":{"input_cost_per_token":0.001,"output_cost_per_token":0.002}}`),
		ModelMappings:   map[string]string{"vendor/model-a": "model-b"},
	}}, f.proxy.modelManager, routermodels.OrganizationPolicyLoadOptions{LiteLLMDBEnabled: true, LiteLLMDBRequired: true})
	require.NoError(t, err)
	f.proxy.organizationPolicies = registry
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"vendor/model-a","messages":[]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
	assert.Zero(t, f.calls[2].Load())
	db := f.proxy.LiteLLMDB.(*organizationPolicyTestDB)
	db.mu.Lock()
	defer db.mu.Unlock()
	require.Len(t, db.logs, 1)
	assert.Equal(t, "model-b", db.logs[0].Model)
	assert.InDelta(t, 0.02, db.logs[0].Spend, 1e-9)
	var metadata map[string]any
	require.NoError(t, json.Unmarshal([]byte(db.logs[0].Metadata), &metadata))
	spend := metadata["spend_logs_metadata"].(map[string]any)
	assert.Equal(t, "profile-a", spend["billing_profile_id"])
	assert.Equal(t, "vendor/model-a", spend["public_model_name"])
}

func TestRoutingProfileDoesNotGrantModelAccess(t *testing.T) {
	f := newRoutingProfileFixture(t)
	f.proxy.strictAllTeamModelsACL = true
	f.proxy.LiteLLMDB.(*organizationPolicyTestDB).tokens["restricted-key"].Models = []string{"model-b"}
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusForbidden, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileRestoredHistoryMayUsePermittedAccount(t *testing.T) {
	f := newRoutingProfileFixture(t)
	f.proxy.responseStore = &fakeResponseStore{entry: &responsestore.StoredEntry{
		CredentialName:   "provider-other",
		AccumulatedInput: json.RawMessage(`[{"role":"user","content":"earlier"}]`),
		ResponseJSON:     &responses.Response{ID: "previous", Model: "model-a"},
	}}
	w := f.request("restricted-key", "/v1/responses", `{"model":"model-a","input":"next","previous_response_id":"previous"}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
	assert.Zero(t, f.calls[2].Load())
}

func TestRoutingProfileMissingHistoryFailsClosed(t *testing.T) {
	f := newRoutingProfileFixture(t)
	w := f.request("restricted-key", "/v1/responses", `{"model":"model-a","input":"next","previous_response_id":"unknown"}`)
	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileConcurrentKeysRemainIsolated(t *testing.T) {
	f := newRoutingProfileFixture(t)
	var group sync.WaitGroup
	for i := 0; i < 12; i++ {
		for _, key := range []string{"restricted-key", "other-key"} {
			group.Go(func() {
				w := f.request(key, "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
				assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
			})
		}
	}
	group.Wait()
	assert.Equal(t, int32(12), f.calls[0].Load())
	assert.Equal(t, int32(12), f.calls[2].Load())
	assert.Zero(t, f.calls[1].Load())
	assert.Zero(t, f.calls[3].Load())
}
