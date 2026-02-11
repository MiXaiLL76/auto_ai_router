package models

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
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

// TestGetRemoteModels_CacheExpiryRace tests concurrent access to GetRemoteModels with cache expiry
// This test is designed to catch TOCTOU race conditions when cache expires
func TestGetRemoteModels_CacheExpiryRace(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, 100, []config.ModelRPMConfig{})

	// Create a mock HTTP server that responds with a models list
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
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

	for i := 0; i < 100; i++ {
		<-done
	}
}
