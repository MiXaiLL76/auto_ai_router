package main

import (
	"log/slog"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/modelupdate"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSplitCredentialModel(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "standard format",
			input:    "openai_main:gpt-4o",
			expected: []string{"openai_main", "gpt-4o"},
		},
		{
			name:     "with multiple colons in model name",
			input:    "openai_main:gpt-4o:turbo",
			expected: []string{"openai_main", "gpt-4o:turbo"},
		},
		{
			name:     "simple names",
			input:    "cred1:model1",
			expected: []string{"cred1", "model1"},
		},
		{
			name:     "with dashes and underscores",
			input:    "openai_backup:gpt-3.5-turbo",
			expected: []string{"openai_backup", "gpt-3.5-turbo"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := modelupdate.SplitCredentialModel(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestInitializeModelManagerKeepsBackendRateLimitsBehindClientSurface(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{DefaultModelsRPM: -1},
		Fail2Ban: config.Fail2BanConfig{
			MaxAttempts: 3,
			BanDuration: time.Minute,
			ErrorCodes:  []int{429},
		},
		Credentials: []config.CredentialConfig{{
			Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1, TPM: -1,
		}},
		Models: []config.ModelRPMConfig{{
			Name: "backend-chat", Credential: "provider", RPM: 7, TPM: 70,
		}},
		ModelAlias:     map[string]string{"public/chat": "backend-chat"},
		ClientModelIDs: []string{"public/chat"},
	}
	logger := slog.New(slog.DiscardHandler)
	_, limiter, bal := initializeBalancer(cfg, logger, nil)
	manager := initializeModelManager(logger, cfg, limiter, bal)

	assert.Equal(t, 7, limiter.GetModelLimitRPM("provider", "backend-chat"))
	assert.Equal(t, 70, limiter.GetModelLimitTPM("provider", "backend-chat"))
	assert.Equal(t, -1, limiter.GetModelLimitRPM("provider", "public/chat"))
	models := manager.GetClientModels()
	if assert.Len(t, models.Data, 1) {
		assert.Equal(t, "public/chat", models.Data[0].ID)
	}
}

func TestInitializeModelManagerRegistersAcceptedModelAliases(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{DefaultModelsRPM: -1},
		Credentials: []config.CredentialConfig{{
			Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1, TPM: -1,
		}},
		Models: []config.ModelRPMConfig{{
			Name: "backend-chat", Credential: "provider", RPM: 7, TPM: 70,
		}},
		ModelAlias:         map[string]string{"public/chat": "backend-chat"},
		ClientModelIDs:     []string{"public/chat"},
		AcceptedModelAlias: map[string]string{"legacy-chat": "public/chat"},
	}
	logger := slog.New(slog.DiscardHandler)
	_, limiter, bal := initializeBalancer(cfg, logger, nil)
	manager := initializeModelManager(logger, cfg, limiter, bal)

	assert.True(t, manager.IsClientModelIDRoutable("legacy-chat"))
	canonical, alias, err := manager.ResolvePublicModelAlias("legacy-chat")
	require.NoError(t, err)
	assert.True(t, alias)
	assert.Equal(t, "public/chat", canonical)

	models := manager.GetClientModels()
	require.Len(t, models.Data, 1)
	assert.Equal(t, "public/chat", models.Data[0].ID)
}
