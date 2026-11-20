package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/require"
)

// Strict isolation requires an explicit model selector in key metadata.
// Deployment tags alone cannot preserve that intent after all tags are removed.
// No upstream requests are sent; selection and the real retry path are exercised.
func TestPoolIsolationReview(t *testing.T) {
	newFixture := func(t *testing.T) (*Proxy, *models.Manager, []config.ModelRPMConfig, []config.CredentialConfig) {
		t.Helper()
		var credentials []config.CredentialConfig
		var entries []config.ModelRPMConfig
		for _, name := range []string{"a", "b", "c"} {
			credentials = append(credentials, config.CredentialConfig{
				Name: name, Type: config.ProviderTypeOpenAI, RPM: 100, BaseURL: "http://127.0.0.1:1",
			})
			var pools []string
			if name == "a" {
				pools = []string{"pool-01"}
			}
			entries = append(entries,
				config.ModelRPMConfig{Name: "model-a", Model: "same-upstream-model", Credential: name, Pools: pools},
				config.ModelRPMConfig{Name: "model-alternate", Model: "same-upstream-model", Credential: name},
				config.ModelRPMConfig{Name: "model-other", Model: "different-upstream-model", Credential: name},
			)
		}
		manager := models.New(testhelpers.NewTestLogger(), 100, nil)
		manager.UpdateDBModels(entries, nil, credentials)
		manager.SetModelAliases(map[string]string{"explicit-alias": "model-a"})
		builder := NewTestProxyBuilder().WithCredentials(credentials...)
		builder.config.ModelManager = manager
		return builder.Build(), manager, entries, credentials
	}

	reachable := func(t *testing.T, proxy *Proxy, model string, bound bool) []string {
		t.Helper()
		info := &dbmodels.TokenInfo{}
		if bound {
			info.Metadata = map[string]interface{}{routingProfilesKey: []interface{}{"pool-01"}, models.RoutingModelsField: []interface{}{"model-a"}}
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"`+model+`","messages":[]}`))
		req = withTrustedClientAuth(req, "fixture-client-key", info)
		w := httptest.NewRecorder()
		logCtx := &RequestLogContext{}
		prepared, ok := proxy.orchestrateRequest(w, req, logCtx)
		if !ok {
			require.GreaterOrEqual(t, w.Code, 400, w.Body.String())
			return nil
		}
		cred := prepared.cred
		got := []string{cred.Name}
		tried := GetTried(prepared.request.Context())
		attempted := map[string]bool{cred.Name: true}
		for attempt := 1; attempt <= 3; attempt++ {
			next, _, ready, _ := proxy.nextCrossTypeRetryCredential(prepared.request, prepared, logCtx, cred, tried, attempted, attempt, true)
			if !ready {
				break
			}
			require.False(t, attempted[next.Name], "a retry must not repeat a credential")
			attempted[next.Name] = true
			got = append(got, next.Name)
			cred = next
		}
		return got
	}

	for _, tc := range []struct {
		name  string
		model string
		bound bool
		want  []string
	}{
		{"unbound key remains unrestricted", "model-a", false, []string{"a", "b", "c"}},
		{"bound model uses its pool", "model-a", true, []string{"a"}},
		{"unrelated model remains unrestricted", "model-other", true, []string{"a", "b", "c"}},
		{"explicit alias keeps the pool", "explicit-alias", true, []string{"a"}},
		{"separate name for same deployment keeps the pool", "model-alternate", true, []string{"a"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			proxy, _, _, _ := newFixture(t)
			require.ElementsMatch(t, tc.want, reachable(t, proxy, tc.model, tc.bound))
		})
	}

	t.Run("removing last model member does not reopen its cascade", func(t *testing.T) {
		proxy, manager, entries, credentials := newFixture(t)
		require.ElementsMatch(t, []string{"a"}, reachable(t, proxy, "model-a", true))
		for i := range entries {
			if entries[i].Name == "model-a" {
				entries[i].Pools = nil
			}
			// The pool remains a valid ID on a different model, so checking only
			// whether a pool exists globally cannot enforce this expectation.
			if entries[i].Name == "model-other" && entries[i].Credential == "a" {
				entries[i].Pools = []string{"pool-01"}
			}
		}
		manager.UpdateDBModels(entries, nil, credentials)
		require.Empty(t, reachable(t, proxy, "model-a", true))
	})
}
