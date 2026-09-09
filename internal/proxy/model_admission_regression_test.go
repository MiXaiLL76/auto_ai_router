package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelAdmissionDirectRouteTakesPrecedenceOverModelAlias(t *testing.T) {
	for _, master := range []bool{false, true} {
		t.Run(map[bool]string{false: "client", true: "master"}[master], func(t *testing.T) {
			cred := config.CredentialConfig{Name: "image-provider", Type: config.ProviderTypeOpenAI, RPM: -1}
			p := newAdmissionTestProxy(t, []config.CredentialConfig{cred}, []config.ModelRPMConfig{
				{Name: "openai/gpt-image-2", Model: "gpt-image-2", Credential: cred.Name},
			}, map[string]*routermodels.ModelPrice{"gpt-image-2": {}})
			p.modelManager.SetModelAliases(map[string]string{"openai/gpt-image-2": "gpt-image-2-vsellm"})
			p.modelManager.SetClientModelIDs([]string{"openai/gpt-image-2"})
			token := &dbmodels.TokenInfo{IsMasterKey: master}
			require.True(t, p.modelManager.IsClientModelIDRoutable("openai/gpt-image-2"))
			admission, err := p.resolveModelAdmission(token, scope.PublicContext(), nil, "openai/gpt-image-2", master)
			require.NoError(t, err)
			assert.Equal(t, "openai/gpt-image-2", admission.modelID)
			listed, err := p.ListModelsForToken(token, scope.PublicContext())
			require.NoError(t, err)
			require.Len(t, listed.Data, 1)
			assert.Equal(t, "openai/gpt-image-2", listed.Data[0].ID)
		})
	}
}

func TestModelAdmissionRemoteDenylist(t *testing.T) {
	for _, tc := range []struct {
		name                string
		hops                int
		withAllowedProvider bool
		otherScope          bool
	}{
		{name: "one_hop_all_denied", hops: 1},
		{name: "one_hop_allowed_alternative", hops: 1, withAllowedProvider: true},
		{name: "two_hops_all_denied", hops: 2},
		{name: "two_hops_allowed_alternative", hops: 2, withAllowedProvider: true},
		{name: "three_hops_all_denied", hops: 3},
		{name: "alternative_in_another_scope", hops: 2, withAllowedProvider: true, otherScope: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var deniedCalls, allowedCalls atomic.Int32
			provider := func(calls *atomic.Int32) *httptest.Server {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("ok", "claude-sonnet-4.6", "ok"))
				}))
				t.Cleanup(server.Close)
				return server
			}
			credentials := []config.CredentialConfig{directDenylistCredential("cometapi01", provider(&deniedCalls).URL)}
			if tc.withAllowedProvider {
				credentials = append(credentials, directDenylistCredential("requesty-claude", provider(&allowedCalls).URL))
				if tc.otherScope {
					credentials[1].Scopes = []string{"other-team"}
				}
			}
			configs := make([]config.ModelRPMConfig, 0, len(credentials))
			for _, cred := range credentials {
				configs = append(configs, config.ModelRPMConfig{Name: "claude-sonnet-4.6", Model: "claude-sonnet-4-6", Credential: cred.Name})
			}
			leaf := newAdmissionTestProxy(t, credentials, configs, map[string]*routermodels.ModelPrice{"claude-sonnet-4.6": {}})
			leafServer := serveProxyWithHealth(t, leaf)
			t.Cleanup(leafServer.Close)
			remote := config.CredentialConfig{Name: "air-ger01", Type: config.ProviderTypeAIR, BaseURL: leafServer.URL, APIKey: "master-key", RPM: -1, TPM: -1}
			for hop := 1; hop < tc.hops; hop++ {
				middle := newAdmissionTestProxy(t, []config.CredentialConfig{remote}, []config.ModelRPMConfig{
					{Name: "claude-sonnet-4.6", Credential: remote.Name},
				}, map[string]*routermodels.ModelPrice{"claude-sonnet-4.6": {}})
				// Simulate startup discovery before this router advertises its topology.
				_, err := middle.modelManager.GetRemoteModelsWithError(context.Background(), &remote)
				require.NoError(t, err)
				middleServer := serveProxyWithHealth(t, middle)
				t.Cleanup(middleServer.Close)
				remote.BaseURL = middleServer.URL
			}
			root := newAdmissionTestProxy(t, []config.CredentialConfig{remote}, []config.ModelRPMConfig{
				{Name: "claude-sonnet-4.6", Credential: remote.Name},
			}, map[string]*routermodels.ModelPrice{"claude-sonnet-4.6": {}})
			root.modelManager.SetModelAliases(map[string]string{"anthropic/claude-sonnet-4.6": "claude-sonnet-4.6"})
			root.modelManager.SetClientModelIDs([]string{"anthropic/claude-sonnet-4.6"})
			registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
				OrganizationID: "cloud-ru", PriceProfileID: "test",
				ModelPricesLink:    writeProxyPolicyPrices(t, `{"anthropic/claude-sonnet-4.6":{"input_cost_per_token":0.000005124}}`),
				CredentialDenylist: []string{"cometapi01"},
			}}, root.modelManager, routermodels.OrganizationPolicyLoadOptions{LiteLLMDBEnabled: true, LiteLLMDBRequired: true})
			require.NoError(t, err)
			root.organizationPolicies = registry
			token := &dbmodels.TokenInfo{Token: "hash", UserID: "user", OrganizationID: "cloud-ru", DirectOrganizationID: "cloud-ru"}
			token.Metadata = map[string]interface{}{"air_scopes": []string{"team-a"}}
			root.LiteLLMDB = &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": token}}
			visibility := scopeContextFromTokenInfo(token)
			listed, err := root.ListModelsForToken(token, visibility)
			require.NoError(t, err)
			_, health := root.HealthCheckScoped(visibility)
			if tc.otherScope {
				for _, model := range health.Models {
					assert.NotContains(t, model.ProviderRoutes, "requesty-claude", "scoped health must not expose invisible leaf names")
				}
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"anthropic/claude-sonnet-4.6","messages":[]}`))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Authorization", "Bearer token")
			w := httptest.NewRecorder()
			root.ProxyRequest(w, request)
			assert.Zero(t, deniedCalls.Load())
			if tc.withAllowedProvider && !tc.otherScope {
				require.Len(t, listed.Data, 1)
				assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
				assert.Equal(t, int32(1), allowedCalls.Load())
			} else {
				assert.Empty(t, listed.Data)
				assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
				assert.Zero(t, allowedCalls.Load())
			}
		})
	}
}

func TestModelAdmissionShadowedAliasACL(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		direct, strict, organization bool
	}{
		{name: "direct_route_strict", direct: true, strict: true},
		{name: "active_alias_strict", strict: true},
		{name: "direct_route_permissive", direct: true},
		{name: "organization_direct_route", direct: true, strict: true, organization: true},
		{name: "organization_active_alias", strict: true, organization: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var aCalls, bCalls atomic.Int32
			var receivedHeader atomic.Bool
			providerA := denylistProvider(t, &aCalls, &receivedHeader)
			providerB := denylistProvider(t, &bCalls, &receivedHeader)
			t.Cleanup(providerA.Close)
			t.Cleanup(providerB.Close)
			credentials := []config.CredentialConfig{
				directDenylistCredential("provider-a", providerA.URL),
				directDenylistCredential("provider-b", providerB.URL),
			}
			configs := []config.ModelRPMConfig{{Name: "B", Credential: "provider-b"}}
			if tc.direct {
				configs = append(configs, config.ModelRPMConfig{Name: "A", Credential: "provider-a"})
			}
			p := newAdmissionTestProxy(t, credentials, configs, map[string]*routermodels.ModelPrice{"A": {}, "B": {}})
			p.strictAllTeamModelsACL = tc.strict
			p.modelManager.SetModelAliases(map[string]string{"A": "B"})
			token := &dbmodels.TokenInfo{Token: "hash", UserID: "user", Models: []string{"B"}}
			if tc.organization {
				registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
					OrganizationID: "org", PriceProfileID: "test",
					ModelPricesLink: writeProxyPolicyPrices(t, `{"A":{"input_cost_per_token":0},"B":{"input_cost_per_token":0}}`),
				}}, p.modelManager, routermodels.OrganizationPolicyLoadOptions{LiteLLMDBEnabled: true, LiteLLMDBRequired: true})
				require.NoError(t, err)
				p.organizationPolicies = registry
				token.OrganizationID, token.DirectOrganizationID = "org", "org"
			}
			p.LiteLLMDB = &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": token}}
			listed, err := p.ListModelsForToken(token, scope.PublicContext())
			require.NoError(t, err)
			ids := make([]string, 0, len(listed.Data))
			for _, model := range listed.Data {
				ids = append(ids, model.ID)
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"A","messages":[]}`))
			request.Header.Set("Authorization", "Bearer token")
			request.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			p.ProxyRequest(w, request)
			if tc.direct && tc.strict {
				assert.Equal(t, []string{"B"}, ids)
				assert.Equal(t, http.StatusForbidden, w.Code)
				assert.Zero(t, aCalls.Load())
				assert.Zero(t, bCalls.Load())
			} else {
				assert.Equal(t, []string{"A", "B"}, ids)
				assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
				if tc.direct {
					assert.Equal(t, int32(1), aCalls.Load())
					assert.Zero(t, bCalls.Load())
				} else {
					assert.Zero(t, aCalls.Load())
					assert.Equal(t, int32(1), bCalls.Load())
				}
			}
		})
	}
}

func TestModelAdmissionRemoteDenylistLeafScopesDoNotExpand(t *testing.T) {
	for _, tc := range []struct {
		providers, hops int
	}{
		{16, 1}, {16, 2}, {16, 3}, {17, 1}, {17, 2}, {17, 3}, {32, 3},
	} {
		t.Run(fmt.Sprintf("providers_%d_hops_%d", tc.providers, tc.hops), func(t *testing.T) {
			var calls atomic.Int32
			var receivedHeader atomic.Bool
			provider := denylistProvider(t, &calls, &receivedHeader)
			t.Cleanup(provider.Close)
			credentials := make([]config.CredentialConfig, 0, tc.providers)
			configs := make([]config.ModelRPMConfig, 0, tc.providers)
			for i := 0; i < tc.providers; i++ {
				cred := directDenylistCredential(fmt.Sprintf("provider-%d", i), provider.URL)
				cred.Scopes = []string{fmt.Sprintf("team-%d", i)}
				credentials = append(credentials, cred)
				configs = append(configs, config.ModelRPMConfig{Name: "route", Credential: cred.Name})
			}
			leaf := newAdmissionTestProxy(t, credentials, configs, map[string]*routermodels.ModelPrice{"route": {}})
			leafServer := serveProxyWithHealth(t, leaf)
			t.Cleanup(leafServer.Close)
			remote := config.CredentialConfig{
				Name: "air-child", Type: config.ProviderTypeAIR, BaseURL: leafServer.URL,
				APIKey: "master-key", RPM: -1, TPM: -1,
				Scopes: []string{"local"}, DeniedScopes: []string{"local-blocked"},
			}
			for hop := 1; hop < tc.hops; hop++ {
				middle := newAdmissionTestProxy(t, []config.CredentialConfig{remote}, []config.ModelRPMConfig{
					{Name: "route", Credential: remote.Name},
				}, map[string]*routermodels.ModelPrice{"route": {}})
				_, err := middle.modelManager.GetRemoteModelsWithError(context.Background(), &remote)
				require.NoError(t, err)
				_, health := middle.HealthCheck()
				model := health.Models[remote.Name+":route"]
				assert.Len(t, model.ProviderRoutes, tc.providers)
				for name, path := range model.ProviderRoutes {
					require.NotNil(t, path)
					assert.Len(t, path.Alternatives, 1, "hop=%d provider=%s", hop, name)
				}
				middleServer := serveProxyWithHealth(t, middle)
				t.Cleanup(middleServer.Close)
				remote.BaseURL = middleServer.URL
			}
			root := newAdmissionTestProxy(t, []config.CredentialConfig{remote}, []config.ModelRPMConfig{
				{Name: "route", Credential: remote.Name},
			}, map[string]*routermodels.ModelPrice{"route": {}})
			pricePath := writeProxyPolicyPrices(t, `{"route":{"input_cost_per_token":0}}`)
			for _, policyCase := range []struct {
				name        string
				denylist    []string
				scopes      []string
				unavailable bool
			}{
				{name: "unrelated_denylist", denylist: []string{"unrelated-provider"}, scopes: []string{"local", "team-0"}},
				{name: "empty_denylist", scopes: []string{"local", "team-0"}},
				{name: "visible_provider_denied", denylist: []string{"provider-0"}, scopes: []string{"local", "team-0"}, unavailable: true},
				{name: "local_scope_missing", denylist: []string{"unrelated-provider"}, scopes: []string{"team-0"}, unavailable: true},
				{name: "local_scope_denied", denylist: []string{"unrelated-provider"}, scopes: []string{"local", "team-0", "local-blocked"}, unavailable: true},
			} {
				t.Run(policyCase.name, func(t *testing.T) {
					registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
						OrganizationID: "org", PriceProfileID: "test", ModelPricesLink: pricePath, CredentialDenylist: policyCase.denylist,
					}}, root.modelManager, routermodels.OrganizationPolicyLoadOptions{LiteLLMDBEnabled: true, LiteLLMDBRequired: true})
					require.NoError(t, err)
					root.organizationPolicies = registry
					token := &dbmodels.TokenInfo{Token: "hash", UserID: "user", OrganizationID: "org", DirectOrganizationID: "org"}
					token.Metadata = map[string]interface{}{"air_scopes": policyCase.scopes}
					root.LiteLLMDB = &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": token}}
					visibility := scopeContextFromTokenInfo(token)
					listed, err := root.ListModelsForToken(token, visibility)
					require.NoError(t, err)
					_, health := root.HealthCheckScoped(visibility)
					for _, model := range health.Models {
						assert.Len(t, model.ProviderRoutes, 1)
						assert.Contains(t, model.ProviderRoutes, "provider-0", "scoped health must hide inaccessible leaf names")
					}
					before := calls.Load()
					request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"route","messages":[]}`))
					request.Header.Set("Authorization", "Bearer token")
					request.Header.Set("Content-Type", "application/json")
					w := httptest.NewRecorder()
					root.ProxyRequest(w, request)
					if policyCase.unavailable {
						assert.Empty(t, listed.Data)
						assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
						assert.Equal(t, before, calls.Load())
					} else {
						assert.Len(t, listed.Data, 1)
						assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
						assert.Equal(t, before+1, calls.Load())
					}
				})
			}
		})
	}
}
