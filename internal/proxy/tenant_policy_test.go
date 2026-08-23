package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	aimodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func protectedTenantTestProxy() *Proxy {
	credential := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI}
	manager := aimodels.New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{{
		Name: "public/model", Credential: credential.Name,
	}})
	manager.SetClientModelIDs([]string{"public/model"})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})
	return &Proxy{modelManager: manager, protectedTenants: []config.ProtectedTenantConfig{{
		Name:            "cloud-ru",
		OrganizationIDs: []string{"org-cloud"},
		Hostnames:       []string{"api.cloud.example"},
		RequiredScopes:  []string{"cloud-ru"},
		RequireModelACL: true,
		RequireRouteACL: true,
	}}}
}

func protectedTenantToken(organizationID string, modelIDs, allowedRoutes []string) *models.TokenInfo {
	return &models.TokenInfo{
		OrganizationID: organizationID,
		Models:         modelIDs,
		AllowedRoutes:  allowedRoutes,
	}
}

func TestProtectedTenantAdmission(t *testing.T) {
	tests := []struct {
		name       string
		org        string
		host       string
		scopes     []string
		models     []string
		routes     []string
		path       string
		wantStatus int
	}{
		{name: "allowed", org: "org-cloud", host: "api.cloud.example:443", scopes: []string{"cloud-ru"}, models: []string{"public/model"}, routes: []string{"POST /v1/chat/completions"}, path: "/v1/chat/completions"},
		{name: "foreign org on protected host", org: "org-other", host: "api.cloud.example", path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "foreign org with protected scope", org: "org-other", host: "router.example", scopes: []string{"cloud-ru"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "missing scope", org: "org-cloud", host: "router.example", models: []string{"public/model"}, routes: []string{"/v1/chat/completions"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "missing models", org: "org-cloud", host: "router.example", scopes: []string{"cloud-ru"}, routes: []string{"/v1/chat/completions"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "all team models is unbounded", org: "org-cloud", host: "router.example", scopes: []string{"cloud-ru"}, models: []string{"all-team-models"}, routes: []string{"/v1/chat/completions"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "wildcard models are unbounded", org: "org-cloud", host: "router.example", scopes: []string{"cloud-ru"}, models: []string{"public/*"}, routes: []string{"/v1/chat/completions"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "regex models are unbounded", org: "org-cloud", host: "router.example", scopes: []string{"cloud-ru"}, models: []string{".+"}, routes: []string{"/v1/chat/completions"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "missing routes", org: "org-cloud", host: "router.example", scopes: []string{"cloud-ru"}, models: []string{"public/model"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "denied route", org: "org-cloud", host: "router.example", scopes: []string{"cloud-ru"}, models: []string{"public/model"}, routes: []string{"/v1/embeddings"}, path: "/v1/chat/completions", wantStatus: http.StatusForbidden},
		{name: "unprotected org on ordinary host", org: "org-other", host: "router.example", path: "/v1/chat/completions"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			prx := protectedTenantTestProxy()
			req := httptest.NewRequest(http.MethodPost, "http://"+test.host+test.path, nil)
			info := protectedTenantToken(test.org, test.models, test.routes)
			logCtx := &RequestLogContext{Request: req, TokenInfo: info, Scope: scope.NewContext(test.scopes, nil)}
			response := httptest.NewRecorder()

			allowed := prx.enforceProtectedTenantAdmission(response, req, logCtx)
			if test.wantStatus == 0 {
				require.True(t, allowed)
				assert.Equal(t, http.StatusOK, response.Code)
				return
			}
			assert.False(t, allowed)
			assert.Equal(t, test.wantStatus, response.Code)
		})
	}
}

func TestProtectedTenantModelACLIsRequestScoped(t *testing.T) {
	prx := protectedTenantTestProxy()
	token := protectedTenantToken("org-cloud", []string{"public/model"}, []string{"llm_api_routes"})
	assert.True(t, prx.IsModelAllowedForRequest(nil, token, "public/model"))
	assert.False(t, prx.IsModelAllowedForRequest(nil, token, "other/model"))

	unprotected := protectedTenantToken("org-other", nil, nil)
	assert.True(t, prx.IsModelAllowedForRequest(nil, unprotected, "other/model"))
}

func TestProtectedTenantMasterKeyIsAdministrativeException(t *testing.T) {
	prx := protectedTenantTestProxy()
	req := httptest.NewRequest(http.MethodPost, "http://api.cloud.example/v1/chat/completions", nil)
	logCtx := &RequestLogContext{Request: req, TokenInfo: &models.TokenInfo{IsMasterKey: true}}
	assert.True(t, prx.enforceProtectedTenantAdmission(httptest.NewRecorder(), req, logCtx))
}

func TestProtectedTenantTrustedInternalRouteKeepsIdentityChecks(t *testing.T) {
	prx := protectedTenantTestProxy()
	req := httptest.NewRequest(http.MethodPost, "http://api.cloud.example/v1/responses", nil)
	req = withTrustedRouteAuthorization(req)
	logCtx := &RequestLogContext{
		Request: req,
		TokenInfo: protectedTenantToken(
			"org-cloud",
			[]string{"public/model"},
			[]string{"GET /v1/responses"},
		),
		Scope: scope.NewContext([]string{"cloud-ru"}, nil),
	}
	response := httptest.NewRecorder()
	assert.True(t, prx.enforceProtectedTenantAdmission(response, req, logCtx))

	logCtx.Scope = scope.NewContext([]string{"other"}, nil)
	response = httptest.NewRecorder()
	assert.False(t, prx.enforceProtectedTenantAdmission(response, req, logCtx))
	assert.Equal(t, http.StatusForbidden, response.Code)
}
