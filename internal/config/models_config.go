package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// ModelRPMConfig represents RPM and TPM limits for a specific model
type ModelRPMConfig struct {
	Name string `yaml:"name"`
	RPM  int    `yaml:"rpm"`
	TPM  int    `yaml:"tpm"`
}

// ModelsConfig represents the models.yaml file structure
type ModelsConfig struct {
	Models []ModelRPMConfig `yaml:"models"`
}

// LoadModelsConfig loads models.yaml file
// Returns nil if file doesn't exist (will be created later)
func LoadModelsConfig(path string) (*ModelsConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// File doesn't exist - return empty config
			return &ModelsConfig{Models: []ModelRPMConfig{}}, nil
		}
		return nil, fmt.Errorf("failed to read models config file: %w", err)
	}

	var cfg ModelsConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse models config file: %w", err)
	}

	return &cfg, nil
}

// SaveModelsConfig saves models config to YAML file
func SaveModelsConfig(path string, cfg *ModelsConfig) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("failed to marshal models config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write models config file: %w", err)
	}

	return nil
}

// GetModelRPM returns RPM limit for a specific model, or defaultRPM if not found
func (c *ModelsConfig) GetModelRPM(modelName string, defaultRPM int) int {
	for _, model := range c.Models {
		if model.Name == modelName {
			return model.RPM
		}
	}
	return defaultRPM
}

// GetModelTPM returns TPM limit for a specific model, or defaultTPM if not found
func (c *ModelsConfig) GetModelTPM(modelName string, defaultTPM int) int {
	for _, model := range c.Models {
		if model.Name == modelName {
			// If TPM is 0 (not set), return default
			if model.TPM == 0 {
				return defaultTPM
			}
			return model.TPM
		}
	}
	return defaultTPM
}

// UpdateOrAddModel updates RPM for existing model or adds new model
func (c *ModelsConfig) UpdateOrAddModel(modelName string, rpm int) {
	c.UpdateOrAddModelWithTPM(modelName, rpm, -1)
}

// UpdateOrAddModelWithTPM updates RPM and TPM for existing model or adds new model
func (c *ModelsConfig) UpdateOrAddModelWithTPM(modelName string, rpm int, tpm int) {
	for i := range c.Models {
		if c.Models[i].Name == modelName {
			c.Models[i].RPM = rpm
			c.Models[i].TPM = tpm
			return
		}
	}
	// Model not found, add it
	c.Models = append(c.Models, ModelRPMConfig{
		Name: modelName,
		RPM:  rpm,
		TPM:  tpm,
	})
}
