package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type routingProfileFixture struct {
	proxy       *Proxy
	credentials []config.CredentialConfig
	calls       [4]atomic.Int32
	statuses    [4]atomic.Int32
}

func newRoutingProfileFixture(t *testing.T) *routingProfileFixture {
	t.Helper()
	f := &routingProfileFixture{}
	for i, name := range []string{"provider-a", "provider-b", "provider-other", "fallback-router"} {
		i := i
		server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			f.calls[i].Add(1)
			w.Header().Set("Content-Type", "application/json")
			if status := f.statuses[i].Load(); status != 0 {
				w.WriteHeader(int(status))
				_, _ = w.Write([]byte(`{"error":{"message":"temporary upstream failure"}}`))
				return
			}
			var body struct {
				Stream bool `json:"stream"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
				return
			}
			if r.URL.Path == "/v1/responses" {
				_, _ = w.Write([]byte(`{"id":"resp-test","object":"response","status":"completed","model":"model-a","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
				return
			}
			_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("test", "model-a", "ok"))
		}))
		t.Cleanup(server.Close)
		credential := directDenylistCredential(name, server.URL)
		credential.Priority = []int{10, 20, 0, 0}[i]
		if i == 3 {
			credential.Type, credential.IsFallback = config.ProviderTypeProxy, true
		}
		f.credentials = append(f.credentials, credential)
	}
	var models []config.ModelRPMConfig
	for _, credential := range f.credentials {
		for _, model := range []string{"model-a", "model-b"} {
			models = append(models, config.ModelRPMConfig{Name: model, Credential: credential.Name})
		}
	}
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, models)
	manager.LoadModelsFromConfig(f.credentials)
	manager.SetCredentials(f.credentials)
	manager.SetModelAliases(map[string]string{"vendor/model-a": "model-a"})
	manager.SetAcceptedModelAliases(map[string]string{"friendly-name": "vendor/model-a"})

	policies, err := routermodels.LoadRoutingProfiles([]config.RoutingProfileConfig{{ID: "pool-a", Rules: []config.RoutingRuleConfig{{Models: []string{"vendor/model-a"}, CredentialAllowlist: []string{"provider-a", "provider-b"}}}}}, f.credentials, manager)
	require.NoError(t, err)
	builder := NewTestProxyBuilder().WithCredentials(f.credentials...).WithSessionSticky(5 * time.Minute)
	builder.config.ModelManager = manager
	builder.config.MaxProviderRetries = 3
	f.proxy = builder.Build()
	f.proxy.routingProfiles = policies
	f.proxy.LiteLLMDB = &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"restricted-key": {Metadata: map[string]interface{}{"air_routing_profiles": []string{"pool-a"}}, Token: strings.Repeat("a", 64), UserID: "user", OrganizationID: "org", DirectOrganizationID: "org"},
		"other-key":      {Token: strings.Repeat("b", 64), UserID: "user", OrganizationID: "org", DirectOrganizationID: "org"},
	}}
	f.proxy.priceRegistry = routermodels.NewModelPriceRegistry()
	f.proxy.priceRegistry.Update(map[string]*routermodels.ModelPrice{"model-a": {}, "model-b": {}})
	return f
}

func (f *routingProfileFixture) request(key, endpoint, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, endpoint, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	// Client-controlled values must never widen a server-owned restriction.
	req.Header.Set("Air-Key-Routing-Policy", "unrestricted")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Set(HeaderAIRCredentialDenylist, `[]`)
	w := httptest.NewRecorder()
	f.proxy.ProxyRequest(w, req)
	return w
}

func TestRoutingProfileIsolatesKeyModelAndAliases(t *testing.T) {
	f := newRoutingProfileFixture(t)
	for _, model := range []string{"model-a", "vendor/model-a", "friendly-name"} {
		w := f.request("restricted-key", "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[],"metadata":{"air_routing_profile":"unrestricted"}}`, model))
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	}
	assert.Equal(t, int32(3), f.calls[0].Load())
	assert.Zero(t, f.calls[2].Load())
	assert.Zero(t, f.calls[3].Load())
	for _, test := range []struct{ key, model string }{{"other-key", "model-a"}, {"restricted-key", "model-b"}} {
		w := f.request(test.key, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[]}`, test.model))
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	}
	assert.Equal(t, int32(2), f.calls[2].Load())
}

func TestRoutingProfileSurvivesRetriesAndExhaustion(t *testing.T) {
	for _, exhaust := range []bool{false, true} {
		t.Run(fmt.Sprint(exhaust), func(t *testing.T) {
			f := newRoutingProfileFixture(t)
			f.statuses[0].Store(http.StatusInternalServerError)
			if exhaust {
				f.statuses[1].Store(http.StatusInternalServerError)
			}
			w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
			if exhaust {
				assert.GreaterOrEqual(t, w.Code, 400)
			} else {
				assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
			}
			assert.Equal(t, int32(1), f.calls[0].Load())
			assert.Equal(t, int32(1), f.calls[1].Load())
			assert.Zero(t, f.calls[2].Load())
			assert.Zero(t, f.calls[3].Load())
		})
	}
}

func TestRoutingProfileOverridesStickyAndRetainsAllowedBinding(t *testing.T) {
	for _, name := range []string{"provider-other", "provider-b"} {
		t.Run(name, func(t *testing.T) {
			f := newRoutingProfileFixture(t)
			info, err := f.proxy.LiteLLMDB.ValidateToken(t.Context(), "restricted-key")
			require.NoError(t, err)
			key := f.proxy.stickySessionKey(&RequestLogContext{Token: "restricted-key", TokenInfo: info, SessionID: "session"})
			f.proxy.sessionStore.Set(key, "model-a", name)
			w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[],"user":"session"}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Zero(t, f.calls[2].Load())
			if name == "provider-b" {
				assert.Equal(t, int32(1), f.calls[1].Load())
			} else {
				assert.Equal(t, int32(1), f.calls[0].Load())
			}
		})
	}
}

func TestRoutingProfileBansAndOrganizationDenylistStillApply(t *testing.T) {
	f := newRoutingProfileFixture(t)
	f.proxy.balancer.AdminBan("provider-a", "model-a", time.Minute, "test")
	registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{OrganizationID: "org", CredentialDenylist: []string{"provider-b"}}}, f.proxy.modelManager, routermodels.OrganizationPolicyLoadOptions{LiteLLMDBEnabled: true, LiteLLMDBRequired: true})
	require.NoError(t, err)
	f.proxy.organizationPolicies = registry
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileResponsesAndStreaming(t *testing.T) {
	for _, request := range []struct{ endpoint, body string }{
		{"/v1/responses", `{"model":"model-a","input":"hello"}`},
		{"/v1/messages", `{"model":"model-a","max_tokens":16,"messages":[{"role":"user","content":"hello"}]}`},
		{"/v1/chat/completions", `{"model":"model-a","messages":[],"stream":true}`},
	} {
		f := newRoutingProfileFixture(t)
		w := f.request("restricted-key", request.endpoint, request.body)
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		assert.Equal(t, int32(1), f.calls[0].Load())
		assert.Zero(t, f.calls[2].Load())
		assert.Zero(t, f.calls[3].Load())
	}
}

func TestRoutingProfileRejectsOpaquePreviousResponseOutsidePool(t *testing.T) {
	f := newRoutingProfileFixture(t)
	f.proxy.responseStore = &fakeResponseStore{entry: &responsestore.StoredEntry{CredentialName: "provider-other"}}
	w := f.request("restricted-key", "/v1/responses", `{"model":"model-a","input":"hello","previous_response_id":"opaque-id"}`)
	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileFinalSendGuard(t *testing.T) {
	f := newRoutingProfileFixture(t)
	logCtx := &RequestLogContext{Scope: scope.PublicContext().RestrictCredentials([]string{"provider-a"})}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	_, err := f.proxy.executeProxyRequest(req, &f.credentials[3], "model-a", []byte(`{}`), time.Now(), logCtx)
	require.ErrorIs(t, err, errRoutingProfile)
	w := httptest.NewRecorder()
	_, done, _, retry, _, _ := f.proxy.sendToCredential(req, w, logCtx, &f.credentials[2], nil, "model-a", "model-a", nil, false, false, false, 0, time.Now(), nil, nil)
	assert.True(t, done)
	assert.False(t, retry)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileRateLimitsDoNotOpenOtherRoutes(t *testing.T) {
	f := newRoutingProfileFixture(t)
	for _, name := range []string{"provider-a", "provider-b"} {
		f.proxy.rateLimiter.AddCredential(name, 1)
		f.proxy.rateLimiter.SetCredentialCurrentUsage(name, 1, 0)
	}
	w := f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusTooManyRequests, w.Code, w.Body.String())
	for i := range f.calls {
		assert.Zero(t, f.calls[i].Load())
	}
	assert.Zero(t, f.proxy.rateLimiter.GetCurrentRPM("provider-other"))
	// The limit remains shared with ordinary traffic, not a new per-policy bucket.
	f.proxy.rateLimiter.SetCredentialCurrentUsage("provider-a", 0, 0)
	w = f.request("restricted-key", "/v1/chat/completions", `{"model":"model-a","messages":[]}`)
	assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), f.calls[0].Load())
}

func TestRoutingProfileOpaqueContinuationNeverRetriesAnotherAccount(t *testing.T) {
	f := newRoutingProfileFixture(t)
	f.proxy.responseStore = &fakeResponseStore{entry: &responsestore.StoredEntry{CredentialName: "provider-a"}}
	f.statuses[0].Store(http.StatusInternalServerError)
	w := f.request("restricted-key", "/v1/responses", `{"model":"model-a","input":"hello","previous_response_id":"opaque-id"}`)
	assert.GreaterOrEqual(t, w.Code, 400)
	assert.Equal(t, int32(1), f.calls[0].Load())
	for _, i := range []int{1, 2, 3} {
		assert.Zero(t, f.calls[i].Load())
	}
}

func TestRoutingProfileRejectsInternalVisionDescription(t *testing.T) {
	u := &visionUpstream{}
	prx, db := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	db.tokens["token"].Metadata = map[string]interface{}{"air_routing_profiles": []string{"vision-test"}}
	var err error
	prx.routingProfiles, err = routermodels.LoadRoutingProfiles([]config.RoutingProfileConfig{{ID: "vision-test", Rules: []config.RoutingRuleConfig{{Models: []string{"glm"}, CredentialAllowlist: []string{"vllm-node"}}}}}, prx.balancer.GetCredentialsSnapshot(), prx.modelManager)
	require.NoError(t, err)
	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	_, bodies := u.snapshot()
	assert.Empty(t, bodies, "an internal helper model must not receive a restricted payload")
}

func TestRoutingProfileNativeWebSocket(t *testing.T) {
	for _, change := range []string{"ban", "binding"} {
		t.Run(change, func(t *testing.T) {
			f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "", func(p *Proxy) {
				db := p.LiteLLMDB.(*nativeWSTestDB)
				db.tokens["client-key"].Metadata = map[string]interface{}{"air_routing_profiles": []string{"ws-test"}}
				var err error
				p.routingProfiles, err = routermodels.LoadRoutingProfiles([]config.RoutingProfileConfig{{ID: "ws-test", Rules: []config.RoutingRuleConfig{{Models: []string{"astra"}, CredentialAllowlist: []string{"upstream"}}}}}, p.balancer.GetCredentialsSnapshot(), p.modelManager)
				require.NoError(t, err)
			})
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"hello"}`)
			upstream := f.upstream(t)
			wsEvent(t, upstream, "response.create")
			wsWrite(t, upstream, wsCreated("resp-test"))
			wsEvent(t, f.client, "response.created")
			wsWrite(t, upstream, wsCompleted("resp-test"))
			wsEvent(t, f.client, "response.completed")
			f.entry(t)
			if change == "ban" {
				f.proxy.balancer.AdminBan("upstream", "astra", time.Minute, "test")
			} else {
				f.db.mu.Lock()
				f.db.tokens["client-key"].Metadata = map[string]interface{}{"air_routing_profiles": []string{"removed-profile"}}
				f.db.mu.Unlock()
			}
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","previous_response_id":"resp-test","input":"next"}`)
			wsEvent(t, f.client, "error")
		})
	}
}
