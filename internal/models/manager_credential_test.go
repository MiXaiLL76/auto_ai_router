package models

import (
	"log/slog"
	"os"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

// TestHasModel_CredentialSpecific tests credential-specific model assignment
func TestHasModel_CredentialSpecific(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Create manager with credential-specific models in config
	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, Credential: "openai-1"},         // Only for openai-1
		{Name: "gpt-3.5-turbo", RPM: 200, Credential: "openai-2"}, // Only for openai-2
		{Name: "gemini-pro", RPM: 60, Credential: "vertex-1"},     // Only for vertex-1
		{Name: "claude-3", RPM: 50},                               // Global - for all credentials
	}

	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "openai-1"},
		{Name: "openai-2"},
		{Name: "vertex-1"},
	}

	// Load models from config
	manager.LoadModelsFromConfig(credentials)

	// Test credential-specific models
	// gpt-4 should only be available for openai-1
	assert.True(t, manager.HasModel("openai-1", "gpt-4"), "gpt-4 should be available for openai-1")
	assert.False(t, manager.HasModel("openai-2", "gpt-4"), "gpt-4 should NOT be available for openai-2")
	assert.False(t, manager.HasModel("vertex-1", "gpt-4"), "gpt-4 should NOT be available for vertex-1")

	// gpt-3.5-turbo should only be available for openai-2
	assert.False(t, manager.HasModel("openai-1", "gpt-3.5-turbo"), "gpt-3.5-turbo should NOT be available for openai-1")
	assert.True(t, manager.HasModel("openai-2", "gpt-3.5-turbo"), "gpt-3.5-turbo should be available for openai-2")
	assert.False(t, manager.HasModel("vertex-1", "gpt-3.5-turbo"), "gpt-3.5-turbo should NOT be available for vertex-1")

	// gemini-pro should only be available for vertex-1
	assert.False(t, manager.HasModel("openai-1", "gemini-pro"), "gemini-pro should NOT be available for openai-1")
	assert.False(t, manager.HasModel("openai-2", "gemini-pro"), "gemini-pro should NOT be available for openai-2")
	assert.True(t, manager.HasModel("vertex-1", "gemini-pro"), "gemini-pro should be available for vertex-1")

	// claude-3 is global - should be available for all
	assert.True(t, manager.HasModel("openai-1", "claude-3"), "claude-3 should be available for openai-1")
	assert.True(t, manager.HasModel("openai-2", "claude-3"), "claude-3 should be available for openai-2")
	assert.True(t, manager.HasModel("vertex-1", "claude-3"), "claude-3 should be available for vertex-1")
}

// TestHasModel_MixedCredentialModels tests mixing credential-specific and global models
func TestHasModel_MixedCredentialModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Scenario: 3 credentials with different model sets
	// - openai-1: has gpt-4, gpt-3.5-turbo (specific) + global models
	// - vertex-1: has gemini-pro, gemini-flash (specific) + global models
	// - anthropic-1: has claude-3-opus (specific) + global models
	// - All have: gpt-4-turbo (global)

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, Credential: "openai-1"},
		{Name: "gpt-3.5-turbo", RPM: 200, Credential: "openai-1"},
		{Name: "gemini-pro", RPM: 60, Credential: "vertex-1"},
		{Name: "gemini-flash", RPM: 100, Credential: "vertex-1"},
		{Name: "claude-3-opus", RPM: 50, Credential: "anthropic-1"},
		{Name: "gpt-4-turbo", RPM: 80}, // Global
	}

	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "openai-1"},
		{Name: "vertex-1"},
		{Name: "anthropic-1"},
	}

	manager.LoadModelsFromConfig(credentials)

	// Test openai-1 models
	assert.True(t, manager.HasModel("openai-1", "gpt-4"))
	assert.True(t, manager.HasModel("openai-1", "gpt-3.5-turbo"))
	assert.True(t, manager.HasModel("openai-1", "gpt-4-turbo"))    // Global
	assert.False(t, manager.HasModel("openai-1", "gemini-pro"))    // vertex-1 only
	assert.False(t, manager.HasModel("openai-1", "claude-3-opus")) // anthropic-1 only

	// Test vertex-1 models
	assert.False(t, manager.HasModel("vertex-1", "gpt-4")) // openai-1 only
	assert.True(t, manager.HasModel("vertex-1", "gemini-pro"))
	assert.True(t, manager.HasModel("vertex-1", "gemini-flash"))
	assert.True(t, manager.HasModel("vertex-1", "gpt-4-turbo"))    // Global
	assert.False(t, manager.HasModel("vertex-1", "claude-3-opus")) // anthropic-1 only

	// Test anthropic-1 models
	assert.False(t, manager.HasModel("anthropic-1", "gpt-4"))      // openai-1 only
	assert.False(t, manager.HasModel("anthropic-1", "gemini-pro")) // vertex-1 only
	assert.True(t, manager.HasModel("anthropic-1", "claude-3-opus"))
	assert.True(t, manager.HasModel("anthropic-1", "gpt-4-turbo")) // Global
}

// TestHasModel_InvalidCredentialInConfig tests behavior when config references non-existent credential
func TestHasModel_InvalidCredentialInConfig(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 100, Credential: "openai-1"},
		{Name: "invalid-model", RPM: 50, Credential: "non-existent-cred"}, // Invalid credential
	}

	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "openai-1"},
	}

	manager.LoadModelsFromConfig(credentials)

	// Valid model should work
	assert.True(t, manager.HasModel("openai-1", "gpt-4"))

	// Invalid model should not be accessible from configured credential
	assert.False(t, manager.HasModel("openai-1", "invalid-model"))

	// For non-existent credential, fallback to allow (no data available)
	// This is the expected behavior when credential doesn't exist
	assert.True(t, manager.HasModel("non-existent-cred", "invalid-model"))
}

// TestHasModel_EmptyCredentialField tests models without credential field (global models)
func TestHasModel_EmptyCredentialField(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	staticModels := []config.ModelRPMConfig{
		{Name: "global-model-1", RPM: 100},                      // No credential = global
		{Name: "global-model-2", RPM: 200, Credential: ""},      // Empty string = global
		{Name: "specific-model", RPM: 50, Credential: "cred-1"}, // Credential-specific
	}

	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "cred-1"},
		{Name: "cred-2"},
		{Name: "cred-3"},
	}

	manager.LoadModelsFromConfig(credentials)

	// Global models should be available for all credentials
	for _, credName := range []string{"cred-1", "cred-2", "cred-3"} {
		assert.True(t, manager.HasModel(credName, "global-model-1"), "global-model-1 should be available for "+credName)
		assert.True(t, manager.HasModel(credName, "global-model-2"), "global-model-2 should be available for "+credName)
	}

	// Credential-specific model should only be for cred-1
	assert.True(t, manager.HasModel("cred-1", "specific-model"))
	assert.False(t, manager.HasModel("cred-2", "specific-model"))
	assert.False(t, manager.HasModel("cred-3", "specific-model"))
}

// TestRealWorldScenario tests a realistic multi-provider setup
func TestRealWorldScenario(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Real-world scenario:
	// - openai-prod: GPT-4, GPT-3.5-turbo
	// - vertex-gemini: Gemini 2.0 Flash, Gemini 1.5 Pro
	// - vertex-claude: Claude 3.5 Sonnet, Claude 3 Opus (via Vertex AI)
	// - All providers: text-embedding-ada-002 (global embedding model)

	staticModels := []config.ModelRPMConfig{
		// OpenAI models
		{Name: "gpt-4", RPM: 100, TPM: 50000, Credential: "openai-prod"},
		{Name: "gpt-3.5-turbo", RPM: 500, TPM: 200000, Credential: "openai-prod"},

		// Vertex AI - Gemini models
		{Name: "gemini-2.0-flash-exp", RPM: 60, TPM: 30000, Credential: "vertex-gemini"},
		{Name: "gemini-1.5-pro", RPM: 30, TPM: 15000, Credential: "vertex-gemini"},

		// Vertex AI - Claude models
		{Name: "claude-3-5-sonnet@20240620", RPM: 50, TPM: 25000, Credential: "vertex-claude"},
		{Name: "claude-3-opus@20240229", RPM: 20, TPM: 10000, Credential: "vertex-claude"},

		// Global embedding model
		{Name: "text-embedding-ada-002", RPM: 1000, TPM: 500000},
	}

	manager := New(logger, 100, staticModels)

	credentials := []config.CredentialConfig{
		{Name: "openai-prod"},
		{Name: "vertex-gemini"},
		{Name: "vertex-claude"},
	}

	manager.LoadModelsFromConfig(credentials)

	// Test OpenAI credential
	t.Run("OpenAI credential", func(t *testing.T) {
		assert.True(t, manager.HasModel("openai-prod", "gpt-4"))
		assert.True(t, manager.HasModel("openai-prod", "gpt-3.5-turbo"))
		assert.True(t, manager.HasModel("openai-prod", "text-embedding-ada-002"))
		assert.False(t, manager.HasModel("openai-prod", "gemini-2.0-flash-exp"))
		assert.False(t, manager.HasModel("openai-prod", "claude-3-5-sonnet@20240620"))
	})

	// Test Vertex Gemini credential
	t.Run("Vertex Gemini credential", func(t *testing.T) {
		assert.False(t, manager.HasModel("vertex-gemini", "gpt-4"))
		assert.True(t, manager.HasModel("vertex-gemini", "gemini-2.0-flash-exp"))
		assert.True(t, manager.HasModel("vertex-gemini", "gemini-1.5-pro"))
		assert.True(t, manager.HasModel("vertex-gemini", "text-embedding-ada-002"))
		assert.False(t, manager.HasModel("vertex-gemini", "claude-3-5-sonnet@20240620"))
	})

	// Test Vertex Claude credential
	t.Run("Vertex Claude credential", func(t *testing.T) {
		assert.False(t, manager.HasModel("vertex-claude", "gpt-4"))
		assert.False(t, manager.HasModel("vertex-claude", "gemini-2.0-flash-exp"))
		assert.True(t, manager.HasModel("vertex-claude", "claude-3-5-sonnet@20240620"))
		assert.True(t, manager.HasModel("vertex-claude", "claude-3-opus@20240229"))
		assert.True(t, manager.HasModel("vertex-claude", "text-embedding-ada-002"))
	})

	// Test that a non-configured model returns false for all
	t.Run("Non-configured model", func(t *testing.T) {
		assert.False(t, manager.HasModel("openai-prod", "non-existent-model"))
		assert.False(t, manager.HasModel("vertex-gemini", "non-existent-model"))
		assert.False(t, manager.HasModel("vertex-claude", "non-existent-model"))
	})
}
