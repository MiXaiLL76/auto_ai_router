package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadModelsConfig_FileExists(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "models.yaml")

	configContent := `
models:
  - name: gpt-4o
    rpm: 50
  - name: gpt-4o-mini
    rpm: 100
  - name: gpt-3.5-turbo
    rpm: 150
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := LoadModelsConfig(configPath)
	require.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.Len(t, cfg.Models, 3)

	assert.Equal(t, "gpt-4o", cfg.Models[0].Name)
	assert.Equal(t, 50, cfg.Models[0].RPM)

	assert.Equal(t, "gpt-4o-mini", cfg.Models[1].Name)
	assert.Equal(t, 100, cfg.Models[1].RPM)

	assert.Equal(t, "gpt-3.5-turbo", cfg.Models[2].Name)
	assert.Equal(t, 150, cfg.Models[2].RPM)
}

func TestLoadModelsConfig_FileNotExists(t *testing.T) {
	cfg, err := LoadModelsConfig("/non/existent/path.yaml")
	require.NoError(t, err) // Should not error, returns empty config
	assert.NotNil(t, cfg)
	assert.Len(t, cfg.Models, 0)
}

func TestLoadModelsConfig_InvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "invalid.yaml")

	invalidContent := `
models:
  - name: gpt-4
    rpm: invalid_rpm
    - this is not valid
`
	err := os.WriteFile(configPath, []byte(invalidContent), 0644)
	require.NoError(t, err)

	_, err = LoadModelsConfig(configPath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse models config file")
}

func TestSaveModelsConfig_Success(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "models.yaml")

	cfg := &ModelsConfig{
		Models: []ModelRPMConfig{
			{Name: "gpt-4o", RPM: 50},
			{Name: "gpt-4o-mini", RPM: 100},
		},
	}

	err := SaveModelsConfig(configPath, cfg)
	require.NoError(t, err)

	// Verify file was created
	_, err = os.Stat(configPath)
	assert.NoError(t, err)

	// Load and verify content
	loadedCfg, err := LoadModelsConfig(configPath)
	require.NoError(t, err)
	assert.Len(t, loadedCfg.Models, 2)
	assert.Equal(t, "gpt-4o", loadedCfg.Models[0].Name)
	assert.Equal(t, 50, loadedCfg.Models[0].RPM)
}

func TestModelsConfig_GetModelRPM_Exists(t *testing.T) {
	cfg := &ModelsConfig{
		Models: []ModelRPMConfig{
			{Name: "gpt-4o", RPM: 50},
			{Name: "gpt-4o-mini", RPM: 100},
		},
	}

	rpm := cfg.GetModelRPM("gpt-4o", 200)
	assert.Equal(t, 50, rpm)

	rpm = cfg.GetModelRPM("gpt-4o-mini", 200)
	assert.Equal(t, 100, rpm)
}

func TestModelsConfig_GetModelRPM_NotExists(t *testing.T) {
	cfg := &ModelsConfig{
		Models: []ModelRPMConfig{
			{Name: "gpt-4o", RPM: 50},
		},
	}

	rpm := cfg.GetModelRPM("non-existent-model", 200)
	assert.Equal(t, 200, rpm) // Should return default
}

func TestModelsConfig_UpdateOrAddModel_Update(t *testing.T) {
	cfg := &ModelsConfig{
		Models: []ModelRPMConfig{
			{Name: "gpt-4o", RPM: 50},
			{Name: "gpt-4o-mini", RPM: 100},
		},
	}

	// Update existing model
	cfg.UpdateOrAddModel("gpt-4o", 75)

	assert.Len(t, cfg.Models, 2)
	assert.Equal(t, "gpt-4o", cfg.Models[0].Name)
	assert.Equal(t, 75, cfg.Models[0].RPM) // Updated
	assert.Equal(t, "gpt-4o-mini", cfg.Models[1].Name)
	assert.Equal(t, 100, cfg.Models[1].RPM) // Unchanged
}

func TestModelsConfig_UpdateOrAddModel_Add(t *testing.T) {
	cfg := &ModelsConfig{
		Models: []ModelRPMConfig{
			{Name: "gpt-4o", RPM: 50},
		},
	}

	// Add new model
	cfg.UpdateOrAddModel("gpt-4o-mini", 100)

	assert.Len(t, cfg.Models, 2)
	assert.Equal(t, "gpt-4o", cfg.Models[0].Name)
	assert.Equal(t, 50, cfg.Models[0].RPM)
	assert.Equal(t, "gpt-4o-mini", cfg.Models[1].Name)
	assert.Equal(t, 100, cfg.Models[1].RPM) // New model
}

func TestModelsConfig_GetModelRPM_EmptyConfig(t *testing.T) {
	cfg := &ModelsConfig{
		Models: []ModelRPMConfig{},
	}

	rpm := cfg.GetModelRPM("any-model", 150)
	assert.Equal(t, 150, rpm) // Should return default
}
