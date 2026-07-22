package models

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	manager := New(logger, 100, []config.ModelRPMConfig{})

	assert.NotNil(t, manager)
	assert.NotNil(t, manager.credentialModels)
	assert.NotNil(t, manager.allModels)
	assert.NotNil(t, manager.modelToCredentials)
}

func TestNew_WithStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, TPM: 10000},
		{Name: "gpt-3.5-turbo", RPM: 200, TPM: 20000},
	}

	manager := New(logger, 50, staticModels)

	assert.NotNil(t, manager)
	assert.True(t, manager.IsEnabled())

	// Check that static models are loaded
	models := manager.GetAllModels()
	assert.Equal(t, 2, len(models.Data))
}

func TestGetAllModels_WithStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100},
		{Name: "gpt-3.5-turbo", RPM: 200},
	}
	manager := New(logger, 100, staticModels)

	result := manager.GetAllModels()

	assert.Equal(t, "list", result.Object)
	assert.Equal(t, 2, len(result.Data))
}

func TestGetAllModelsScoped_FiltersByCredentialScope(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	staticModels := []config.ModelRPMConfig{
		{Name: "shared-model"},
		{Name: "team-a-model", Credential: "team-a-cred"},
		{Name: "team-b-model", Credential: "team-b-cred"},
	}
	manager := New(logger, 100, staticModels)
	credentials := []config.CredentialConfig{
		{Name: "shared-cred", Type: config.ProviderTypeOpenAI},
		{Name: "team-a-cred", Type: config.ProviderTypeOpenAI, Scopes: []string{"team-a"}},
		{Name: "team-b-cred", Type: config.ProviderTypeOpenAI, Scopes: []string{"team-b"}},
	}
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)

	resp := manager.GetAllModelsScoped(scope.NewContext([]string{"team-a"}, nil))
	ids := make([]string, 0, len(resp.Data))
	for _, model := range resp.Data {
		ids = append(ids, model.ID)
	}

	assert.ElementsMatch(t, []string{"shared-model", "team-a-model"}, ids)
}

func TestGetAllModelsScoped_AdminExcludesCredentialsWithoutRoute(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	staticModels := []config.ModelRPMConfig{{Name: "blocked-model", Credential: "proxy"}}
	manager := New(logger, 100, staticModels)
	credentials := []config.CredentialConfig{{
		Name:                    "proxy",
		Type:                    config.ProviderTypeProxy,
		ProviderScopeExpression: scope.FalseExpression(),
		ProviderScopeKnown:      true,
	}}
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)

	response := manager.GetAllModelsScoped(scope.AdminContext())

	assert.Empty(t, response.Data)
}

func TestGetAllModelsScoped_FetchesOnlyVisibleProxyCredentials(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	visibleRequests := 0
	visibleServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		visibleRequests++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
			Credentials: map[string]httputil.CredentialHealthStats{
				"upstream": {Type: "openai"},
			},
			Models: map[string]httputil.ModelHealthStats{
				"visible": {Credential: "upstream", Model: "visible-model"},
			},
		})
	}))
	defer visibleServer.Close()

	hiddenRequests := 0
	hiddenServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hiddenRequests++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
			Credentials: map[string]httputil.CredentialHealthStats{
				"upstream": {Type: "openai"},
			},
			Models: map[string]httputil.ModelHealthStats{
				"hidden": {Credential: "upstream", Model: "hidden-model"},
			},
		})
	}))
	defer hiddenServer.Close()

	manager := New(logger, 100, nil)
	manager.cacheExpiration = -time.Second
	manager.SetCredentials([]config.CredentialConfig{
		{Name: "visible-proxy", Type: config.ProviderTypeProxy, BaseURL: visibleServer.URL, Scopes: []string{"team-a"}},
		{Name: "hidden-proxy", Type: config.ProviderTypeProxy, BaseURL: hiddenServer.URL, Scopes: []string{"team-b"}},
	})

	resp := manager.GetAllModelsScoped(scope.NewContext([]string{"team-a"}, nil))
	ids := make([]string, 0, len(resp.Data))
	for _, model := range resp.Data {
		ids = append(ids, model.ID)
	}

	assert.Equal(t, 1, visibleRequests)
	assert.Equal(t, 0, hiddenRequests)
	assert.Contains(t, ids, "visible-model")
	assert.NotContains(t, ids, "hidden-model")

	resp = manager.GetAllModelsScoped(scope.NewContext([]string{"team-a"}, nil))
	ids = ids[:0]
	for _, model := range resp.Data {
		ids = append(ids, model.ID)
	}

	assert.Equal(t, 1, visibleRequests)
	assert.Equal(t, 0, hiddenRequests)
	assert.Contains(t, ids, "visible-model")
	assert.NotContains(t, ids, "hidden-model")
}

func TestGetAllModelsScoped_CacheIsBounded(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{{Name: "shared-model"}})
	credentials := []config.CredentialConfig{{Name: "shared", Type: config.ProviderTypeOpenAI}}
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)

	for i := 0; i < scopedAllModelsCacheSize+10; i++ {
		manager.GetAllModelsScoped(scope.NewContext([]string{fmt.Sprintf("tenant-%d", i)}, nil))
	}

	assert.Equal(t, scopedAllModelsCacheSize, manager.scopedAllModelsCache.Len())
}

func TestGetAllModelsScoped_CacheSeparatesDynamicModelScopes(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	staticModels := []config.ModelRPMConfig{
		{Name: "team-a-model", Credential: "shared"},
		{Name: "team-b-model", Credential: "shared"},
	}
	manager := New(logger, 100, staticModels)
	credentials := []config.CredentialConfig{{Name: "shared", Type: config.ProviderTypeOpenAI}}
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)
	manager.ReplaceModelScopesForCredential("shared", map[string]ScopeMetadata{
		"team-a-model": {Scopes: []string{"team-a"}},
		"team-b-model": {Scopes: []string{"team-b"}},
	})

	teamA := manager.GetAllModelsScoped(scope.NewContext([]string{"team-a"}, nil))
	teamB := manager.GetAllModelsScoped(scope.NewContext([]string{"team-b"}, nil))

	assert.Equal(t, []string{"team-a-model"}, responseModelIDs(teamA))
	assert.Equal(t, []string{"team-b-model"}, responseModelIDs(teamB))
}

func TestGetAllModelsScoped_CacheKeyDistinguishesDelimitedScopeValues(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{{Name: "model-a", Credential: "shared"}})
	credentials := []config.CredentialConfig{{Name: "shared", Type: config.ProviderTypeOpenAI}}
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)
	manager.ReplaceModelScopesForCredential("shared", map[string]ScopeMetadata{
		"model-a": {Scopes: []string{"a"}},
	})

	separate := manager.GetAllModelsScoped(scope.NewContext([]string{"a", "b"}, nil))
	combined := manager.GetAllModelsScoped(scope.NewContext([]string{"a,b"}, nil))

	assert.Equal(t, []string{"model-a"}, responseModelIDs(separate))
	assert.Empty(t, responseModelIDs(combined))
}

func TestGetAllModelsScoped_RemovesExpiredCacheEntry(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{{Name: "shared-model"}})
	credentials := []config.CredentialConfig{{Name: "shared", Type: config.ProviderTypeOpenAI}}
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)
	visibility := scope.NewContext([]string{"team-a"}, nil)

	manager.GetAllModelsScoped(visibility)
	cacheKey := manager.scopedAllModelsCacheKeyLocked(visibility)
	cached, ok := manager.scopedAllModelsCache.Peek(cacheKey)
	assert.True(t, ok)
	cached.expiresAt = time.Now().Add(-time.Second)
	manager.scopedAllModelsCache.Add(cacheKey, cached)

	_, ok = manager.getCachedScopedAllModels(visibility)

	assert.False(t, ok)
	assert.False(t, manager.scopedAllModelsCache.Contains(cacheKey))
}

func TestCurrentModelsLocked_DropsStaleFetchedModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, nil)
	manager.SetCredentials([]config.CredentialConfig{{Name: "proxy", Type: config.ProviderTypeProxy}})
	manager.ReplaceModelsForCredential("proxy", []string{"current-model"})

	manager.mu.Lock()
	response := manager.currentModelsLocked(ModelsResponse{
		Object: "list",
		Data:   []Model{{ID: "stale-model"}},
	}, map[string]bool{"proxy": true})
	manager.mu.Unlock()

	assert.Equal(t, []string{"current-model"}, responseModelIDs(response))
}

func responseModelIDs(response ModelsResponse) []string {
	ids := make([]string, 0, len(response.Data))
	for _, model := range response.Data {
		ids = append(ids, model.ID)
	}
	return ids
}

func TestGetAllModels_Empty(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	result := manager.GetAllModels()

	assert.Equal(t, "list", result.Object)
	assert.Equal(t, 0, len(result.Data))
}

func TestGetCredentialsForModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, Credential: "test1"},
		{Name: "gpt-4", RPM: 100, Credential: "test2"},
		{Name: "gpt-3.5-turbo", RPM: 200, Credential: "test1"},
	}
	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "test1"},
		{Name: "test2"},
	}
	manager.LoadModelsFromConfig(credentials)

	// Test existing model with multiple credentials
	creds := manager.GetCredentialsForModel("gpt-4")
	assert.Equal(t, 2, len(creds))
	assert.Contains(t, creds, "test1")
	assert.Contains(t, creds, "test2")

	// Test model with single credential
	creds2 := manager.GetCredentialsForModel("gpt-3.5-turbo")
	assert.Equal(t, 1, len(creds2))
	assert.Contains(t, creds2, "test1")

	// Test non-existing model
	creds3 := manager.GetCredentialsForModel("non-existing-model")
	assert.Nil(t, creds3)
}

func TestGetAliasesForModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Same provider-facing model exposed under three route aliases, mirroring
	// config.yaml.example's "Comet API Claude aliases for public model names".
	staticModels := []config.ModelRPMConfig{
		{Name: "anthropic/claude-haiku-4.5", Model: "claude-haiku-4-5-20251001", Credential: "comet"},
		{Name: "claude-haiku-4.5", Model: "claude-haiku-4-5-20251001", Credential: "comet"},
		{Name: "claude-haiku-4-5-20251001", Credential: "comet"},
		{Name: "gpt-4o", Credential: "openai"},
	}
	manager := New(logger, 100, staticModels)
	credentials := []config.CredentialConfig{{Name: "comet"}, {Name: "openai"}}
	manager.LoadModelsFromConfig(credentials)

	aliases := manager.GetAliasesForModel("claude-haiku-4.5", "claude-haiku-4-5-20251001")
	assert.ElementsMatch(t, []string{"claude-haiku-4.5", "anthropic/claude-haiku-4.5", "claude-haiku-4-5-20251001"}, aliases)

	// A model with no other aliases just returns itself.
	aliases2 := manager.GetAliasesForModel("gpt-4o", "gpt-4o")
	assert.Equal(t, []string{"gpt-4o"}, aliases2)

	// Empty realModelID (e.g. no alias configured) - no lookup possible, returns modelID only.
	aliases3 := manager.GetAliasesForModel("unknown-model", "")
	assert.Equal(t, []string{"unknown-model"}, aliases3)
}

func TestHasModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, Credential: "test1"},
		{Name: "gpt-3.5-turbo", RPM: 200, Credential: "test1"},
		{Name: "claude-3", RPM: 150, Credential: "test2"},
	}
	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "test1"},
		{Name: "test2"},
	}
	manager.LoadModelsFromConfig(credentials)

	// Test credential has model
	assert.True(t, manager.HasModel("test1", "gpt-4"))
	assert.True(t, manager.HasModel("test1", "gpt-3.5-turbo"))

	// Test credential doesn't have model
	assert.False(t, manager.HasModel("test1", "claude-3"))

	// Test different credential
	assert.True(t, manager.HasModel("test2", "claude-3"))
	assert.False(t, manager.HasModel("test2", "gpt-4"))

	// Test non-existing credential with configured model (should return false - model exists but not for this cred)
	assert.False(t, manager.HasModel("non-existing", "gpt-4"))

	// Test non-existing credential with non-configured model (fallback - allow)
	assert.True(t, manager.HasModel("non-existing", "some-unknown-model"))
}

func TestHasModel_NoStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// Should return true when no models configured (allow all)
	assert.True(t, manager.HasModel("test1", "gpt-4"))
	assert.True(t, manager.HasModel("test1", "any-model"))
}

func TestIsEnabled(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Test 1: No static models -> IsEnabled=false
	manager1 := New(logger, 100, []config.ModelRPMConfig{})
	assert.False(t, manager1.IsEnabled(), "Should be disabled when no static models configured")

	// Test 2: With static models -> IsEnabled=true
	manager2 := New(logger, 100, []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100},
	})
	assert.True(t, manager2.IsEnabled(), "Should be enabled when static models are configured")
}

func TestGetModelRPM(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100},
		{Name: "gpt-3.5-turbo", RPM: 200},
	}
	manager := New(logger, 50, staticModels)

	// Test existing model in config
	rpm1 := manager.GetModelRPM("gpt-4")
	assert.Equal(t, 100, rpm1)

	rpm2 := manager.GetModelRPM("gpt-3.5-turbo")
	assert.Equal(t, 200, rpm2)

	// Test non-existing model (should return default)
	rpm3 := manager.GetModelRPM("non-existing-model")
	assert.Equal(t, 50, rpm3)
}

func TestGetModelRPM_NoStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 75, []config.ModelRPMConfig{})

	// Should return default RPM when no models configured
	rpm := manager.GetModelRPM("any-model")
	assert.Equal(t, 75, rpm)
}

func TestGetModelTPM(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, TPM: 10000},
		{Name: "gpt-3.5-turbo", RPM: 200, TPM: 20000},
	}
	manager := New(logger, 50, staticModels)

	// Test existing model in config
	tpm1 := manager.GetModelTPM("gpt-4")
	assert.Equal(t, 10000, tpm1)

	tpm2 := manager.GetModelTPM("gpt-3.5-turbo")
	assert.Equal(t, 20000, tpm2)

	// Test non-existing model (should return default -1)
	tpm3 := manager.GetModelTPM("non-existing-model")
	assert.Equal(t, -1, tpm3)
}

func TestGetModelTPM_NoStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 75, []config.ModelRPMConfig{})

	// Should return -1 (unlimited) when no models configured
	tpm := manager.GetModelTPM("any-model")
	assert.Equal(t, -1, tpm)
}

func TestGetModelTPM_ZeroValue(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, TPM: 0}, // TPM not set
	}
	manager := New(logger, 50, staticModels)

	// Should return -1 (unlimited) when TPM is 0
	tpm := manager.GetModelTPM("gpt-4")
	assert.Equal(t, -1, tpm)
}

func TestGetModelRPMForCredential(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", Credential: "cred1", RPM: 100},
		{Name: "gpt-4", Credential: "cred2", RPM: 200},
		{Name: "gpt-3.5-turbo", Credential: "cred1", RPM: 150},
	}
	manager := New(logger, 50, staticModels)

	// Test existing model with specific credential
	rpm1 := manager.GetModelRPMForCredential("gpt-4", "cred1")
	assert.Equal(t, 100, rpm1)

	// Test same model with different credential
	rpm2 := manager.GetModelRPMForCredential("gpt-4", "cred2")
	assert.Equal(t, 200, rpm2)

	// Test model with non-existent credential (should return default)
	rpm3 := manager.GetModelRPMForCredential("gpt-4", "cred3")
	assert.Equal(t, 50, rpm3)

	// Test non-existent model (should return default)
	rpm4 := manager.GetModelRPMForCredential("non-existing", "cred1")
	assert.Equal(t, 50, rpm4)
}

func TestGetModelRPMForCredential_NoStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 75, []config.ModelRPMConfig{})

	// Should return default RPM when no models configured
	rpm := manager.GetModelRPMForCredential("any-model", "any-cred")
	assert.Equal(t, 75, rpm)
}

func TestGetModelTPMForCredential(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", Credential: "cred1", TPM: 10000},
		{Name: "gpt-4", Credential: "cred2", TPM: 20000},
		{Name: "gpt-3.5-turbo", Credential: "cred1", TPM: 0}, // 0 means unlimited
	}
	manager := New(logger, 50, staticModels)

	// Test existing model with specific credential
	tpm1 := manager.GetModelTPMForCredential("gpt-4", "cred1")
	assert.Equal(t, 10000, tpm1)

	// Test same model with different credential
	tpm2 := manager.GetModelTPMForCredential("gpt-4", "cred2")
	assert.Equal(t, 20000, tpm2)

	// Test model with TPM = 0 (should return -1 for unlimited)
	tpm3 := manager.GetModelTPMForCredential("gpt-3.5-turbo", "cred1")
	assert.Equal(t, -1, tpm3)

	// Test model with non-existent credential (should return default)
	tpm4 := manager.GetModelTPMForCredential("gpt-4", "cred3")
	assert.Equal(t, -1, tpm4)

	// Test non-existent model (should return default)
	tpm5 := manager.GetModelTPMForCredential("non-existing", "cred1")
	assert.Equal(t, -1, tpm5)
}

func TestGetModelTPMForCredential_NoStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 75, []config.ModelRPMConfig{})

	// Should return -1 (unlimited) when no models configured
	tpm := manager.GetModelTPMForCredential("any-model", "any-cred")
	assert.Equal(t, -1, tpm)
}

func TestGetModelsForCredential(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, Credential: "test1"},
		{Name: "gpt-3.5-turbo", RPM: 200, Credential: "test1"},
		{Name: "claude-3", RPM: 150, Credential: "test2"},
		{Name: "gemini-pro", RPM: 80}, // Global model
	}
	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "test1"},
		{Name: "test2"},
	}
	manager.LoadModelsFromConfig(credentials)

	// Test credential with multiple models
	models1 := manager.GetModelsForCredential("test1")
	assert.Equal(t, 3, len(models1), "test1 should have 3 models (2 specific + 1 global)")

	modelIDs1 := make(map[string]bool)
	for _, model := range models1 {
		modelIDs1[model.ID] = true
	}
	assert.True(t, modelIDs1["gpt-4"], "test1 should have gpt-4")
	assert.True(t, modelIDs1["gpt-3.5-turbo"], "test1 should have gpt-3.5-turbo")
	assert.True(t, modelIDs1["gemini-pro"], "test1 should have gemini-pro (global)")

	// Test credential with one specific model + global
	models2 := manager.GetModelsForCredential("test2")
	assert.Equal(t, 2, len(models2), "test2 should have 2 models (1 specific + 1 global)")

	modelIDs2 := make(map[string]bool)
	for _, model := range models2 {
		modelIDs2[model.ID] = true
	}
	assert.True(t, modelIDs2["claude-3"], "test2 should have claude-3")
	assert.True(t, modelIDs2["gemini-pro"], "test2 should have gemini-pro (global)")

	// Test non-existent credential - should still get global models
	models3 := manager.GetModelsForCredential("non-existent")
	assert.Equal(t, 1, len(models3), "non-existent credential should have 1 global model")
	assert.Equal(t, "gemini-pro", models3[0].ID, "should have gemini-pro (global)")
}

func TestGetModelsForCredential_NoStaticModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// Should return empty list when no models configured
	models := manager.GetModelsForCredential("any-cred")
	assert.Equal(t, 0, len(models))
}

func TestGetModelsForCredential_GlobalModelsOnly(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "global-1", RPM: 100},
		{Name: "global-2", RPM: 200},
	}
	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "test1"},
		{Name: "test2"},
	}
	manager.LoadModelsFromConfig(credentials)

	// Both credentials should have all global models
	models1 := manager.GetModelsForCredential("test1")
	assert.Equal(t, 2, len(models1))

	models2 := manager.GetModelsForCredential("test2")
	assert.Equal(t, 2, len(models2))
}

func TestAddModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// Test adding a new model for a credential
	manager.AddModel("gateway02", "gpt-oss-120b")

	// Verify the model appears in credentialModels
	models := manager.GetModelsForCredential("gateway02")
	assert.Len(t, models, 1)
	assert.Equal(t, "gpt-oss-120b", models[0].ID)

	// Verify HasModel returns true
	assert.True(t, manager.HasModel("gateway02", "gpt-oss-120b"))

	// Test adding the same model again (should not duplicate)
	manager.AddModel("gateway02", "gpt-oss-120b")
	models = manager.GetModelsForCredential("gateway02")
	assert.Len(t, models, 1, "Should not create duplicate model entry")
}

// TestConcurrentGetAllModels tests concurrent access to GetAllModels
func TestConcurrentGetAllModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100},
		{Name: "gpt-3.5-turbo", RPM: 200},
	}
	manager := New(logger, 50, staticModels)

	// Run concurrent reads
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			result := manager.GetAllModels()
			assert.Equal(t, "list", result.Object)
			assert.Equal(t, 2, len(result.Data))
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}

// TestConcurrentAddModelAndGetCredentialsForModel tests concurrent AddModel and GetCredentialsForModel
func TestConcurrentAddModelAndGetCredentialsForModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	done := make(chan bool, 20)

	// 10 goroutines adding models
	for i := 0; i < 10; i++ {
		go func(idx int) {
			modelName := "model-" + string(rune(idx+'0'))
			manager.AddModel("cred1", modelName)
			done <- true
		}(i)
	}

	// 10 goroutines reading models
	for i := 0; i < 10; i++ {
		go func() {
			creds := manager.GetCredentialsForModel("model-0")
			_ = creds // Just check it doesn't panic
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}
}

// TestConcurrentSetCredentialsAndGetAllModels tests SetCredentials concurrent with GetAllModels
func TestConcurrentSetCredentialsAndGetAllModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	done := make(chan bool, 20)

	// 10 goroutines setting credentials
	for i := 0; i < 10; i++ {
		go func() {
			creds := []config.CredentialConfig{
				{Name: "cred1"},
				{Name: "cred2"},
			}
			manager.SetCredentials(creds)
			done <- true
		}()
	}

	// 10 goroutines calling GetAllModels
	for i := 0; i < 10; i++ {
		go func() {
			_ = manager.GetAllModels()
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}
}

// TestConcurrentHasModelAndAddModel tests HasModel concurrent with AddModel
func TestConcurrentHasModelAndAddModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	done := make(chan bool, 20)

	// 10 goroutines adding models
	for i := 0; i < 10; i++ {
		go func(idx int) {
			modelName := "model-" + string(rune(idx+'0'))
			manager.AddModel("cred1", modelName)
			done <- true
		}(i)
	}

	// 10 goroutines checking if models exist
	for i := 0; i < 10; i++ {
		go func(idx int) {
			modelName := "model-" + string(rune(idx+'0'))
			_ = manager.HasModel("cred1", modelName)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}
}

// TestGetAllModels_CacheExpiryRace tests concurrent access to GetAllModels with cache expiry
// This test is designed to catch TOCTOU race conditions when cache expires
func TestGetAllModels_CacheExpiryRace(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100},
		{Name: "gpt-3.5-turbo", RPM: 200},
	}
	manager := New(logger, 50, staticModels)

	// Run concurrent reads to populate cache
	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func() {
			resp := manager.GetAllModels()
			if len(resp.Data) != 2 {
				t.Errorf("Expected 2 models, got %d", len(resp.Data))
			}
			done <- true
		}()
	}

	for i := 0; i < 100; i++ {
		<-done
	}
}

func TestSetModelAliases(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	aliases := map[string]string{
		"gpt-4":  "gpt-4o",
		"claude": "claude-sonnet-4-20250514",
		"gemini": "gemini-2.5-flash",
	}
	manager.SetModelAliases(aliases)

	// Verify aliases are set
	resolved, isAlias := manager.ResolveAlias("gpt-4")
	assert.True(t, isAlias)
	assert.Equal(t, "gpt-4o", resolved)

	resolved, isAlias = manager.ResolveAlias("claude")
	assert.True(t, isAlias)
	assert.Equal(t, "claude-sonnet-4-20250514", resolved)

	resolved, isAlias = manager.ResolveAlias("gemini")
	assert.True(t, isAlias)
	assert.Equal(t, "gemini-2.5-flash", resolved)
}

func TestResolveAlias_NotAnAlias(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	aliases := map[string]string{
		"gpt-4": "gpt-4o",
	}
	manager.SetModelAliases(aliases)

	// Non-alias model should return as-is
	resolved, isAlias := manager.ResolveAlias("gpt-4o")
	assert.False(t, isAlias)
	assert.Equal(t, "gpt-4o", resolved)

	resolved, isAlias = manager.ResolveAlias("unknown-model")
	assert.False(t, isAlias)
	assert.Equal(t, "unknown-model", resolved)
}

func TestResolveAlias_EmptyAliases(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// No aliases set
	resolved, isAlias := manager.ResolveAlias("gpt-4")
	assert.False(t, isAlias)
	assert.Equal(t, "gpt-4", resolved)
}

func TestSetModelAliases_SkipsSelfAlias(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	aliases := map[string]string{
		"gpt-4":  "gpt-4", // self-reference, should be skipped
		"claude": "claude-sonnet-4-20250514",
	}
	manager.SetModelAliases(aliases)

	// Self-alias should not resolve
	resolved, isAlias := manager.ResolveAlias("gpt-4")
	assert.False(t, isAlias)
	assert.Equal(t, "gpt-4", resolved)

	// Normal alias should work
	resolved, isAlias = manager.ResolveAlias("claude")
	assert.True(t, isAlias)
	assert.Equal(t, "claude-sonnet-4-20250514", resolved)
}

func TestSetModelAliases_Overwrite(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// Set initial aliases
	manager.SetModelAliases(map[string]string{"gpt-4": "gpt-4o"})
	resolved, _ := manager.ResolveAlias("gpt-4")
	assert.Equal(t, "gpt-4o", resolved)

	// Overwrite with new aliases
	manager.SetModelAliases(map[string]string{"gpt-4": "gpt-4o-mini"})
	resolved, _ = manager.ResolveAlias("gpt-4")
	assert.Equal(t, "gpt-4o-mini", resolved)
}

// TestGetRemoteModels_CacheExpiryRace tests concurrent access to GetRemoteModels with cache expiry
// This test is designed to catch TOCTOU race conditions when cache expires
func TestNewModelPriceRegistry(t *testing.T) {
	registry := NewModelPriceRegistry()

	assert.NotNil(t, registry)
	assert.Equal(t, 0, registry.Count())
	assert.True(t, registry.LastUpdate().IsZero(), "LastUpdate should be zero for a new registry")
}

func TestModelPriceRegistry_UpdateAndGetPrice(t *testing.T) {
	registry := NewModelPriceRegistry()

	prices := map[string]*ModelPrice{
		"gpt-4": {
			InputCostPerToken:  0.00003,
			OutputCostPerToken: 0.00006,
		},
		"claude-3-opus": {
			InputCostPerToken:  0.000015,
			OutputCostPerToken: 0.000075,
		},
		"gemini-1.5-pro": {
			InputCostPerToken:           0.0000035,
			OutputCostPerToken:          0.0000105,
			OutputCostPerReasoningToken: 0.000014,
		},
	}

	registry.Update(prices)

	// Verify Count matches
	assert.Equal(t, 3, registry.Count())

	// Verify LastUpdate is recent
	assert.False(t, registry.LastUpdate().IsZero(), "LastUpdate should not be zero after Update")
	assert.WithinDuration(t, time.Now().UTC(), registry.LastUpdate(), 5*time.Second)

	// Verify GetPrice returns correct values
	gpt4Price := registry.GetPrice("gpt-4")
	assert.NotNil(t, gpt4Price)
	assert.Equal(t, 0.00003, gpt4Price.InputCostPerToken)
	assert.Equal(t, 0.00006, gpt4Price.OutputCostPerToken)

	claudePrice := registry.GetPrice("claude-3-opus")
	assert.NotNil(t, claudePrice)
	assert.Equal(t, 0.000015, claudePrice.InputCostPerToken)
	assert.Equal(t, 0.000075, claudePrice.OutputCostPerToken)

	geminiPrice := registry.GetPrice("gemini-1.5-pro")
	assert.NotNil(t, geminiPrice)
	assert.Equal(t, 0.000014, geminiPrice.OutputCostPerReasoningToken)
}

func TestModelPriceRegistry_MergeDB(t *testing.T) {
	registry := NewModelPriceRegistry()

	initial := map[string]*ModelPrice{
		"gpt-4": {
			InputCostPerToken:  0.00003,
			OutputCostPerToken: 0.00006,
		},
		"claude-3-opus": {
			InputCostPerToken:  0.000015,
			OutputCostPerToken: 0.000075,
		},
	}
	registry.Update(initial)
	prevUpdate := registry.LastUpdate()

	dbPrices := map[string]*ModelPrice{
		"gpt-4": {
			InputCostPerToken:  0.000031,
			OutputCostPerToken: 0.000061,
		},
		"gemini-1.5-pro": {
			InputCostPerToken:  0.0000035,
			OutputCostPerToken: 0.0000105,
		},
	}
	registry.MergeDB(dbPrices)

	assert.Equal(t, 3, registry.Count())
	assert.WithinDuration(t, time.Now().UTC(), registry.LastUpdate(), 5*time.Second)
	assert.True(t, registry.LastUpdate().After(prevUpdate) || registry.LastUpdate().Equal(prevUpdate))

	// DB prices should override existing entries.
	updated := registry.GetPrice("gpt-4")
	assert.NotNil(t, updated)
	assert.Equal(t, 0.000031, updated.InputCostPerToken)
	assert.Equal(t, 0.000061, updated.OutputCostPerToken)

	// Existing non-DB entries should remain.
	claude := registry.GetPrice("claude-3-opus")
	assert.NotNil(t, claude)
	assert.Equal(t, 0.000015, claude.InputCostPerToken)

	// New DB entries should be added.
	gemini := registry.GetPrice("gemini-1.5-pro")
	assert.NotNil(t, gemini)
}

func TestModelPriceRegistry_GetPrice_NotFound(t *testing.T) {
	registry := NewModelPriceRegistry()

	// Empty registry
	result := registry.GetPrice("nonexistent-model")
	assert.Nil(t, result)

	// After adding some prices, lookup a model that doesn't exist
	registry.Update(map[string]*ModelPrice{
		"gpt-4": {InputCostPerToken: 0.00003},
	})

	result = registry.GetPrice("claude-3-opus")
	assert.Nil(t, result, "GetPrice should return nil for a model not in the registry")
}

func TestUpdateDBModels_PreservesStaticAndMapsDB(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "static-global", RPM: 10},
		{Name: "static-specific", Credential: "yaml-1", RPM: 20},
		{Name: "static-real", Model: "real-static", RPM: 30},
	}
	manager := New(logger, 100, staticModels)

	staticCreds := []config.CredentialConfig{
		{Name: "yaml-1"},
		{Name: "yaml-2"},
	}
	manager.LoadModelsFromConfig(staticCreds)

	dbModels := []config.ModelRPMConfig{
		{Name: "db-global", RPM: 5},
		{Name: "db-specific", Credential: "db-cred-1", RPM: 7, TPM: 9, Model: "real-db"},
		{Name: "db-unknown", Credential: "missing", RPM: 11},
	}
	dbCreds := []config.CredentialConfig{
		{Name: "db-cred-1"},
		{Name: "db-model-foo"},
	}
	allCreds := append(append([]config.CredentialConfig(nil), staticCreds...), dbCreds...)

	manager.UpdateDBModels(dbModels, staticCreds, allCreds)

	assert.ElementsMatch(t, []string{"yaml-1", "yaml-2"}, manager.GetCredentialsForModel("static-global"))
	assert.ElementsMatch(t, []string{"yaml-1", "yaml-2"}, manager.GetCredentialsForModel("db-global"))
	assert.ElementsMatch(t, []string{"db-cred-1"}, manager.GetCredentialsForModel("db-specific"))
	assert.Nil(t, manager.GetCredentialsForModel("db-unknown"))

	// DB model with a specific credential goes into the per-credential map.
	real, ok := manager.GetRealModelNameForCredential("db-specific", "db-cred-1")
	assert.True(t, ok)
	assert.Equal(t, "real-db", real)

	// Global GetRealModelName should NOT find it (it has a credential).
	_, ok = manager.GetRealModelName("db-specific")
	assert.False(t, ok)

	real, ok = manager.GetRealModelName("static-real")
	assert.True(t, ok)
	assert.Equal(t, "real-static", real)
}

// TestUpdateDBModels_StaticRealNameNotOverriddenByDB verifies that a static
// models[].model mapping (e.g. "anthropic/claude-opus-4.7" → "global.anthropic.claude-opus-4-7")
// is never replaced by a conflicting entry from the LiteLLM DB sync.
// Regression test: without the fix, UpdateDBModels would overwrite staticModelRealNames
// with DB values, causing requests to be forwarded with the wrong model name and
// returning empty responses with 0 tokens after the first sync cycle.
func TestUpdateDBModels_StaticRealNameNotOverriddenByDB(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "anthropic/claude-opus-4.7", Model: "global.anthropic.claude-opus-4-7", RPM: 1000},
		{Name: "z-ai/glm-4.7-flash", Model: "zai.glm-4.7-flash", RPM: 1000},
	}
	m := New(logger, 100, staticModels)

	staticCreds := []config.CredentialConfig{{Name: "cred-1"}}
	m.LoadModelsFromConfig(staticCreds)

	// Simulate DB sync where LiteLLM has a conflicting model field
	// (e.g. the DB stores "claude-opus-4" instead of the correct "global.anthropic.claude-opus-4-7")
	dbModels := []config.ModelRPMConfig{
		{Name: "anthropic/claude-opus-4.7", Model: "claude-opus-4", RPM: 500, Credential: "db-cred"},
		{Name: "z-ai/glm-4.7-flash", Model: "wrong-real-name", RPM: 500, Credential: "db-cred"},
	}
	dbCreds := []config.CredentialConfig{{Name: "db-cred"}}
	allCreds := append(append([]config.CredentialConfig(nil), staticCreds...), dbCreds...)

	m.UpdateDBModels(dbModels, staticCreds, allCreds)

	// Static real names must survive the DB sync unchanged
	real, ok := m.GetRealModelName("anthropic/claude-opus-4.7")
	assert.True(t, ok)
	assert.Equal(t, "global.anthropic.claude-opus-4-7", real,
		"DB sync must not overwrite static models[].model mapping")

	real, ok = m.GetRealModelName("z-ai/glm-4.7-flash")
	assert.True(t, ok)
	assert.Equal(t, "zai.glm-4.7-flash", real,
		"DB sync must not overwrite static models[].model mapping")
}

func TestUpdateDBModels_DBOnlyGlobalMapping(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	manager := New(logger, 100, []config.ModelRPMConfig{})

	staticCreds := []config.CredentialConfig{}
	dbCreds := []config.CredentialConfig{
		{Name: "db-cred-1"},
		{Name: "db-model-foo"},
	}
	dbModels := []config.ModelRPMConfig{
		{Name: "db-global", RPM: 5},
	}

	manager.UpdateDBModels(dbModels, staticCreds, dbCreds)

	creds := manager.GetCredentialsForModel("db-global")
	assert.ElementsMatch(t, []string{"db-cred-1"}, creds)
}

// TestGetRealModelNameForCredential_SameAliasMultipleProviders verifies that the same model
// alias (e.g. "claude-haiku-4.5") resolves to the correct real name for each credential,
// even when Bedrock and OpenRouter both expose it under the same name but with different
// provider-specific identifiers.
func TestGetRealModelNameForCredential_SameAliasMultipleProviders(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{
			Name:       "claude-haiku-4.5",
			Model:      "global.anthropic.claude-haiku-4-5-20251001-v1:0",
			Credential: "bedrock_aws",
			RPM:        100,
		},
		{
			Name:       "claude-haiku-4.5",
			Model:      "anthropic/claude-haiku-4.5",
			Credential: "openrouter",
			RPM:        100,
		},
	}
	m := New(logger, 100, staticModels)

	// Each credential gets its own correct real name.
	real, ok := m.GetRealModelNameForCredential("claude-haiku-4.5", "bedrock_aws")
	assert.True(t, ok)
	assert.Equal(t, "global.anthropic.claude-haiku-4-5-20251001-v1:0", real)

	real, ok = m.GetRealModelNameForCredential("claude-haiku-4.5", "openrouter")
	assert.True(t, ok)
	assert.Equal(t, "anthropic/claude-haiku-4.5", real)

	// Global lookup finds nothing (all entries are credential-specific).
	_, ok = m.GetRealModelName("claude-haiku-4.5")
	assert.False(t, ok)

	// Credential that has no mapping falls through to global (nothing here).
	_, ok = m.GetRealModelNameForCredential("claude-haiku-4.5", "unknown-cred")
	assert.False(t, ok)
}

// TestGetRealModelNameForCredential_FallbackToGlobal verifies that when a model has a
// global real name (no credential in config) and is routed to any credential, the global
// real name is used as fallback.
func TestGetRealModelNameForCredential_FallbackToGlobal(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "my-model", Model: "real-provider-name", RPM: 100},
	}
	m := New(logger, 100, staticModels)

	// Any credential gets the global real name.
	real, ok := m.GetRealModelNameForCredential("my-model", "any-cred")
	assert.True(t, ok)
	assert.Equal(t, "real-provider-name", real)

	// Global lookup also works.
	real, ok = m.GetRealModelName("my-model")
	assert.True(t, ok)
	assert.Equal(t, "real-provider-name", real)
}

func TestGetRemoteModels_CacheExpiryRace(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// Create a mock HTTP server that responds with a models list
	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			models := map[string]interface{}{
				"object": "list",
				"data": []map[string]string{
					{"id": "gpt-4", "object": "model", "owned_by": "openai"},
					{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
				},
			}
			err := json.NewEncoder(w).Encode(models)
			assert.Nil(t, err)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test-proxy",
		Type:    config.ProviderTypeProxy,
		BaseURL: server.URL,
		APIKey:  "test-key",
	}

	// Run concurrent reads to test cache logic under concurrency
	// Note: Using 10 goroutines instead of 100 because:
	// - httputil has minProxyFetchInterval = 100ms rate limiting per credential
	// - 100 goroutines * 100ms = 10 seconds minimum (exceeds 5s default timeout)
	// - 10 goroutines * 100ms = 1 second (fits within timeout)
	// This still thoroughly tests concurrent access and caching behavior
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			models := manager.GetRemoteModels(cred)
			if len(models) > 0 {
				// Successfully fetched models from cache/server
				assert.Equal(t, 2, len(models))
				assert.Equal(t, "gpt-4", models[0].ID)
			}
			done <- true
		}()
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestManager_GetModelWeightForCredential(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4o", Credential: "ours", Weight: 200},
		{Name: "gpt-4o", Credential: "azure"}, // no weight → unset
		{Name: "shared", Weight: 7},           // global entry (no credential)
	}
	m := New(logger, 50, staticModels)

	assert.Equal(t, 200, m.GetModelWeightForCredential("gpt-4o", "ours"), "credential-specific override")
	assert.Equal(t, 0, m.GetModelWeightForCredential("gpt-4o", "azure"), "unset model weight is 0")
	assert.Equal(t, 7, m.GetModelWeightForCredential("shared", "anyone"), "falls back to global entry")
	assert.Equal(t, 0, m.GetModelWeightForCredential("unknown-model", "ours"), "untracked model is 0")
}

func TestManager_GetModelWeightForCredential_DBSpecificUnsetFallsBackToStaticGlobal(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "shared", Weight: 7},
	}
	m := New(logger, 50, staticModels)

	staticCreds := []config.CredentialConfig{{Name: "yaml-cred"}}
	dbCreds := []config.CredentialConfig{{Name: "db-cred"}}
	allCreds := append(append([]config.CredentialConfig(nil), staticCreds...), dbCreds...)
	dbModels := []config.ModelRPMConfig{
		{Name: "shared", Credential: "db-cred"},
	}

	m.UpdateDBModels(dbModels, staticCreds, allCreds)

	assert.Equal(t, 7, m.GetModelWeightForCredential("shared", "db-cred"),
		"DB credential-specific unset weight must not block the global YAML weight")
}
