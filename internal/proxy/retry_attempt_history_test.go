package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/require"
)

func TestRetryIgnoresDeniedPriorityHistory(t *testing.T) {
	for _, deniedHasModel := range []bool{false, true} {
		name := "unrelated model"
		if deniedHasModel {
			name = "same model"
		}
		t.Run(name, func(t *testing.T) {
			var primaryCalls, secondaryCalls, deniedCalls atomic.Int32
			primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				primaryCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "30")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded","type":"rate_limit_error","code":"rate_limited"}}`))
			}))
			defer primary.Close()
			secondary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				secondaryCalls.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"id\":\"response-1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			}))
			defer secondary.Close()
			denied := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				deniedCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer denied.Close()
			credentials := []config.CredentialConfig{
				{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: primary.URL, APIKey: "key", RPM: 100, TPM: 10000},
				{Name: "secondary", Type: config.ProviderTypeOpenAI, BaseURL: secondary.URL, APIKey: "key", Priority: 1, RPM: 100, TPM: 10000},
				{Name: "denied", Type: config.ProviderTypeOpenAI, BaseURL: denied.URL, APIKey: "key", Priority: 150, RPM: 100, TPM: 10000},
			}
			builder := NewTestProxyBuilder().WithCredentials(credentials...).WithMaxProviderRetries(6)
			routes := []config.ModelRPMConfig{{Name: "model", Credential: "primary"}, {Name: "model", Credential: "secondary"}}
			if deniedHasModel {
				routes = append(routes, config.ModelRPMConfig{Name: "model", Credential: "denied"})
			}
			manager := routermodels.New(builder.config.Logger, 100, routes)
			manager.SetModelAliases(map[string]string{"public/model": "model"})
			manager.LoadModelsFromConfig(credentials)
			manager.SetCredentials(credentials)
			policies, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
				OrganizationID: "org-1", PriceProfileID: "profile-1",
				ModelPricesLink: writeProxyPolicyPrices(t, `{"public/model":{"input_cost_per_token":0.001,"output_cost_per_token":0.002}}`),
				ModelAllowlist:  []string{"public/model"}, CredentialDenylist: []string{"denied"},
			}}, manager, routermodels.OrganizationPolicyLoadOptions{LiteLLMDBEnabled: true, LiteLLMDBRequired: true})
			require.NoError(t, err)
			builder.config.ModelManager = manager
			builder.config.OrganizationPolicies = policies
			prx := builder.Build()
			prx.LiteLLMDB = &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
				"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-1", OrganizationID: "org-1"},
			}}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"public/model","messages":[{"role":"user","content":"OK"}],"stream":true}`))
			req.Header.Set("Authorization", "Bearer token")
			req.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			prx.ProxyRequest(response, req)
			require.Equal(t, int32(1), primaryCalls.Load())
			require.Zero(t, deniedCalls.Load())
			require.Equal(t, int32(1), secondaryCalls.Load())
			require.Equal(t, http.StatusOK, response.Code)
			require.Contains(t, response.Body.String(), "[DONE]")
		})
	}
}
