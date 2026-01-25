package models

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	manager := New(logger, true, 100, "/tmp/models_test.yaml")

	assert.NotNil(t, manager)
	assert.True(t, manager.enabled)
	assert.Equal(t, 100, manager.defaultModelsRPM)
	assert.NotNil(t, manager.credentialModels)
	assert.NotNil(t, manager.allModels)
	assert.NotNil(t, manager.modelToCredentials)
}

func TestNew_Disabled(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	manager := New(logger, false, 50, "/tmp/models_test.yaml")

	assert.NotNil(t, manager)
	assert.False(t, manager.enabled)
	assert.Equal(t, 50, manager.defaultModelsRPM)
}

func TestFetchModels_Success(t *testing.T) {
	// Create mock server that returns models
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/models", r.URL.Path)
		assert.Equal(t, "GET", r.Method)
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

		response := ModelsResponse{
			Object: "list",
			Data: []Model{
				{ID: "gpt-4", Object: "model", Created: 1234567890},
				{ID: "gpt-3.5-turbo", Object: "model", Created: 1234567891},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_fetch.yaml")

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServer.URL, RPM: 100},
	}

	manager.FetchModels(credentials, 5*time.Second)

	// Verify models were fetched
	assert.Equal(t, 2, len(manager.allModels))
	assert.Equal(t, 2, len(manager.credentialModels["test1"]))
	assert.Contains(t, manager.modelToCredentials["gpt-4"], "test1")
	assert.Contains(t, manager.modelToCredentials["gpt-3.5-turbo"], "test1")
}

func TestFetchModels_MultipleCredentials(t *testing.T) {
	// Create mock server 1
	mockServer1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := ModelsResponse{
			Object: "list",
			Data: []Model{
				{ID: "gpt-4", Object: "model"},
				{ID: "gpt-3.5-turbo", Object: "model"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer mockServer1.Close()

	// Create mock server 2
	mockServer2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := ModelsResponse{
			Object: "list",
			Data: []Model{
				{ID: "gpt-4", Object: "model"}, // Duplicate
				{ID: "claude-3", Object: "model"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer mockServer2.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_multi.yaml")

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServer1.URL, RPM: 100},
		{Name: "test2", APIKey: "key2", BaseURL: mockServer2.URL, RPM: 100},
	}

	manager.FetchModels(credentials, 5*time.Second)

	// Should have 3 unique models
	assert.Equal(t, 3, len(manager.allModels))

	// gpt-4 should be available from both credentials
	assert.Contains(t, manager.modelToCredentials["gpt-4"], "test1")
	assert.Contains(t, manager.modelToCredentials["gpt-4"], "test2")
	assert.Equal(t, 2, len(manager.modelToCredentials["gpt-4"]))

	// gpt-3.5-turbo should only be from test1
	assert.Contains(t, manager.modelToCredentials["gpt-3.5-turbo"], "test1")
	assert.Equal(t, 1, len(manager.modelToCredentials["gpt-3.5-turbo"]))

	// claude-3 should only be from test2
	assert.Contains(t, manager.modelToCredentials["claude-3"], "test2")
	assert.Equal(t, 1, len(manager.modelToCredentials["claude-3"]))
}

func TestFetchModels_PartialFailure(t *testing.T) {
	// Create a working server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := ModelsResponse{
			Object: "list",
			Data:   []Model{{ID: "gpt-4", Object: "model"}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_partial.yaml")

	credentials := []config.CredentialConfig{
		{Name: "working", APIKey: "key1", BaseURL: mockServer.URL, RPM: 100},
		{Name: "failing", APIKey: "key2", BaseURL: "http://invalid-url-that-does-not-exist.com", RPM: 100},
	}

	manager.FetchModels(credentials, 2*time.Second)

	// Should still have models from working credential
	assert.Equal(t, 1, len(manager.allModels))
	assert.Equal(t, 1, len(manager.credentialModels["working"]))
	assert.Equal(t, 0, len(manager.credentialModels["failing"]))
}

func TestFetchModels_Disabled(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, false, 100, "/tmp/models_test_disabled.yaml")

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test.com", RPM: 100},
	}

	manager.FetchModels(credentials, 5*time.Second)

	// Should not fetch when disabled
	assert.Equal(t, 0, len(manager.allModels))
	assert.Equal(t, 0, len(manager.credentialModels))
}

func TestFetchModels_ErrorResponse(t *testing.T) {
	// Create server that returns error
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "invalid API key"}`))
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_error.yaml")

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "invalid", BaseURL: mockServer.URL, RPM: 100},
	}

	manager.FetchModels(credentials, 5*time.Second)

	// Should not have any models
	assert.Equal(t, 0, len(manager.allModels))
	assert.Equal(t, 0, len(manager.credentialModels["test1"]))
}

func TestGetAllModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_getall.yaml")

	// Manually add some models
	manager.allModels = []Model{
		{ID: "gpt-4", Object: "model"},
		{ID: "gpt-3.5-turbo", Object: "model"},
	}

	result := manager.GetAllModels()

	assert.Equal(t, "list", result.Object)
	assert.Equal(t, 2, len(result.Data))
	assert.Equal(t, "gpt-4", result.Data[0].ID)
	assert.Equal(t, "gpt-3.5-turbo", result.Data[1].ID)
}

func TestGetAllModels_Empty(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_empty.yaml")

	result := manager.GetAllModels()

	assert.Equal(t, "list", result.Object)
	assert.Equal(t, 0, len(result.Data))
}

func TestGetCredentialsForModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_getcreds.yaml")

	manager.modelToCredentials["gpt-4"] = []string{"test1", "test2"}
	manager.modelToCredentials["gpt-3.5-turbo"] = []string{"test1"}

	// Test existing model
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

func TestGetCredentialsForModel_Disabled(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, false, 100, "/tmp/models_test_disabled2.yaml")

	manager.modelToCredentials["gpt-4"] = []string{"test1"}

	// Should return nil when disabled
	creds := manager.GetCredentialsForModel("gpt-4")
	assert.Nil(t, creds)
}

func TestHasModel(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/models_test_hasmodel.yaml")

	manager.credentialModels["test1"] = []string{"gpt-4", "gpt-3.5-turbo"}
	manager.credentialModels["test2"] = []string{"claude-3"}

	// Test credential has model
	assert.True(t, manager.HasModel("test1", "gpt-4"))
	assert.True(t, manager.HasModel("test1", "gpt-3.5-turbo"))

	// Test credential doesn't have model
	assert.False(t, manager.HasModel("test1", "claude-3"))

	// Test different credential
	assert.True(t, manager.HasModel("test2", "claude-3"))
	assert.False(t, manager.HasModel("test2", "gpt-4"))

	// Test non-existing credential (should return true - allow when no data)
	assert.True(t, manager.HasModel("non-existing", "gpt-4"))
}

func TestHasModel_Disabled(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, false, 100, "/tmp/models_test_disabled3.yaml")

	manager.credentialModels["test1"] = []string{"gpt-4"}

	// Should return true when disabled (allow all)
	assert.True(t, manager.HasModel("test1", "gpt-4"))
	assert.True(t, manager.HasModel("test1", "any-model"))
}

func TestIsEnabled(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	manager1 := New(logger, true, 100, "/tmp/test.yaml")
	assert.True(t, manager1.IsEnabled())

	manager2 := New(logger, false, 100, "/tmp/test.yaml")
	assert.False(t, manager2.IsEnabled())
}

func TestGetModelRPM(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 50, "/tmp/models_test_getrpm.yaml")

	// Mock models config
	manager.modelsConfig = &config.ModelsConfig{
		Models: []config.ModelRPMConfig{
			{Name: "gpt-4", RPM: 100},
			{Name: "gpt-3.5-turbo", RPM: 200},
		},
	}

	// Test existing model in config
	rpm1 := manager.GetModelRPM("gpt-4")
	assert.Equal(t, 100, rpm1)

	rpm2 := manager.GetModelRPM("gpt-3.5-turbo")
	assert.Equal(t, 200, rpm2)

	// Test non-existing model (should return default)
	rpm3 := manager.GetModelRPM("non-existing-model")
	assert.Equal(t, 50, rpm3)
}

func TestGetModelRPM_NilConfig(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 75, "/tmp/models_test_nilconfig.yaml")

	manager.modelsConfig = nil

	// Should return default RPM when config is nil
	rpm := manager.GetModelRPM("any-model")
	assert.Equal(t, 75, rpm)
}

func TestFetchModelsFromCredential_InvalidJSON(t *testing.T) {
	// Create server that returns invalid JSON
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`invalid json`))
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/test.yaml")

	client := &http.Client{Timeout: 5 * time.Second}
	cred := config.CredentialConfig{
		Name:    "test",
		APIKey:  "key",
		BaseURL: mockServer.URL,
		RPM:     100,
	}

	models, err := manager.fetchModelsFromCredential(client, cred)

	assert.Error(t, err)
	assert.Nil(t, models)
	assert.Contains(t, err.Error(), "failed to decode response")
}

func TestFetchModelsFromCredential_Timeout(t *testing.T) {
	// Create server that never responds
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Second) // Sleep longer than timeout
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/test.yaml")

	client := &http.Client{Timeout: 100 * time.Millisecond} // Short timeout
	cred := config.CredentialConfig{
		Name:    "test",
		APIKey:  "key",
		BaseURL: mockServer.URL,
		RPM:     100,
	}

	models, err := manager.fetchModelsFromCredential(client, cred)

	assert.Error(t, err)
	assert.Nil(t, models)
}

func TestFetchModelsFromCredential_BaseURLWithSlash(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/models", r.URL.Path)

		response := ModelsResponse{
			Object: "list",
			Data:   []Model{{ID: "test-model", Object: "model"}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	manager := New(logger, true, 100, "/tmp/test.yaml")

	client := &http.Client{Timeout: 5 * time.Second}
	cred := config.CredentialConfig{
		Name:    "test",
		APIKey:  "key",
		BaseURL: mockServer.URL + "/", // With trailing slash
		RPM:     100,
	}

	models, err := manager.fetchModelsFromCredential(client, cred)

	assert.NoError(t, err)
	assert.Equal(t, 1, len(models))
	assert.Equal(t, "test-model", models[0].ID)
}
