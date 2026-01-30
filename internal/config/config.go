package config

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// ProviderType represents the type of AI provider
type ProviderType string

const (
	ProviderTypeOpenAI    ProviderType = "openai"
	ProviderTypeVertexAI  ProviderType = "vertex-ai"
	ProviderTypeAnthropic ProviderType = "anthropic"
)

// IsValid checks if the provider type is valid
func (p ProviderType) IsValid() bool {
	switch p {
	case ProviderTypeOpenAI, ProviderTypeVertexAI, ProviderTypeAnthropic:
		return true
	}
	return false
}

// ModelRPMConfig represents RPM and TPM limits for a specific model
type ModelRPMConfig struct {
	Name       string `yaml:"name"`
	RPM        int    `yaml:"rpm"`
	TPM        int    `yaml:"tpm"`
	Credential string `yaml:"credential,omitempty"` // If set, model is only available for this credential
}

type Config struct {
	Server      ServerConfig       `yaml:"server"`
	Fail2Ban    Fail2BanConfig     `yaml:"fail2ban"`
	Credentials []CredentialConfig `yaml:"credentials"`
	Monitoring  MonitoringConfig   `yaml:"monitoring"`
	Models      []ModelRPMConfig   `yaml:"models,omitempty"`
}

type ServerConfig struct {
	Port             int           `yaml:"port"`
	MaxBodySizeMB    int           `yaml:"max_body_size_mb"`
	RequestTimeout   time.Duration `yaml:"request_timeout"`
	LoggingLevel     string        `yaml:"logging_level"`
	MasterKey        string        `yaml:"master_key"`
	DefaultModelsRPM int           `yaml:"default_models_rpm"`
}

type Fail2BanConfig struct {
	MaxAttempts int           `yaml:"max_attempts"`
	BanDuration time.Duration `yaml:"ban_duration"`
	ErrorCodes  []int         `yaml:"error_codes"`
}

// UnmarshalYAML implements custom unmarshaling for ServerConfig with env variable support
func (s *ServerConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with all string fields
	type tempConfig struct {
		Port             string `yaml:"port"`
		MaxBodySizeMB    string `yaml:"max_body_size_mb"`
		RequestTimeout   string `yaml:"request_timeout"`
		LoggingLevel     string `yaml:"logging_level"`
		MasterKey        string `yaml:"master_key"`
		DefaultModelsRPM string `yaml:"default_models_rpm"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Resolve and parse each field
	var err error

	// Port
	if temp.Port != "" {
		s.Port, err = resolveEnvInt(temp.Port, 8080)
		if err != nil {
			return fmt.Errorf("invalid port: %w", err)
		}
	}

	// MaxBodySizeMB
	if temp.MaxBodySizeMB != "" {
		s.MaxBodySizeMB, err = resolveEnvInt(temp.MaxBodySizeMB, 10)
		if err != nil {
			return fmt.Errorf("invalid max_body_size_mb: %w", err)
		}
	}

	// RequestTimeout
	if temp.RequestTimeout != "" {
		s.RequestTimeout, err = resolveEnvDuration(temp.RequestTimeout, 30*time.Second)
		if err != nil {
			return fmt.Errorf("invalid request_timeout: %w", err)
		}
	}

	// LoggingLevel
	s.LoggingLevel = resolveEnvString(temp.LoggingLevel)

	// MasterKey
	s.MasterKey = resolveEnvString(temp.MasterKey)

	// DefaultModelsRPM
	if temp.DefaultModelsRPM != "" {
		s.DefaultModelsRPM, err = resolveEnvInt(temp.DefaultModelsRPM, 50)
		if err != nil {
			return fmt.Errorf("invalid default_models_rpm: %w", err)
		}
	}

	return nil
}

type CredentialConfig struct {
	Name    string       `yaml:"name"`
	Type    ProviderType `yaml:"type"`
	APIKey  string       `yaml:"api_key"`
	BaseURL string       `yaml:"base_url"`
	RPM     int          `yaml:"rpm"`
	TPM     int          `yaml:"tpm"`

	// Vertex AI specific fields
	ProjectID       string `yaml:"project_id,omitempty"`
	Location        string `yaml:"location,omitempty"`
	CredentialsFile string `yaml:"credentials_file,omitempty"`
	CredentialsJSON string `yaml:"credentials_json,omitempty"`
}

// UnmarshalYAML implements custom unmarshaling for CredentialConfig with env variable support
func (c *CredentialConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with all string fields
	type tempConfig struct {
		Name            string `yaml:"name"`
		Type            string `yaml:"type"`
		APIKey          string `yaml:"api_key"`
		BaseURL         string `yaml:"base_url"`
		RPM             string `yaml:"rpm"`
		TPM             string `yaml:"tpm"`
		ProjectID       string `yaml:"project_id,omitempty"`
		Location        string `yaml:"location,omitempty"`
		CredentialsFile string `yaml:"credentials_file,omitempty"`
		CredentialsJSON string `yaml:"credentials_json,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Resolve string fields
	c.Name = resolveEnvString(temp.Name)
	c.Type = ProviderType(resolveEnvString(temp.Type))
	c.APIKey = resolveEnvString(temp.APIKey)
	c.BaseURL = resolveEnvString(temp.BaseURL)

	// Resolve Vertex AI specific fields
	c.ProjectID = resolveEnvString(temp.ProjectID)
	c.Location = resolveEnvString(temp.Location)
	c.CredentialsFile = resolveEnvString(temp.CredentialsFile)
	c.CredentialsJSON = resolveEnvString(temp.CredentialsJSON)

	// Resolve and parse RPM
	var err error
	if temp.RPM != "" {
		c.RPM, err = resolveEnvInt(temp.RPM, -1)
		if err != nil {
			return fmt.Errorf("invalid rpm for credential '%s': %w", c.Name, err)
		}
	} else {
		c.RPM = -1 // Default to unlimited
	}

	// Resolve and parse TPM
	if temp.TPM != "" {
		c.TPM, err = resolveEnvInt(temp.TPM, -1)
		if err != nil {
			return fmt.Errorf("invalid tpm for credential '%s': %w", c.Name, err)
		}
	} else {
		c.TPM = -1 // Default to unlimited
	}

	return nil
}

type MonitoringConfig struct {
	PrometheusEnabled bool   `yaml:"prometheus_enabled"`
	HealthCheckPath   string `yaml:"health_check_path"`
	LogErrors         bool   `yaml:"log_errors,omitempty"`
	ErrorsLogPath     string `yaml:"errors_log_path,omitempty"`
}

// UnmarshalYAML implements custom unmarshaling for MonitoringConfig with env variable support
func (m *MonitoringConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with all string fields
	type tempConfig struct {
		PrometheusEnabled string `yaml:"prometheus_enabled"`
		HealthCheckPath   string `yaml:"health_check_path"`
		LogErrors         string `yaml:"log_errors,omitempty"`
		ErrorsLogPath     string `yaml:"errors_log_path,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Resolve and parse PrometheusEnabled
	var err error
	if temp.PrometheusEnabled != "" {
		m.PrometheusEnabled, err = resolveEnvBool(temp.PrometheusEnabled, false)
		if err != nil {
			return fmt.Errorf("invalid prometheus_enabled: %w", err)
		}
	}

	// Resolve HealthCheckPath
	m.HealthCheckPath = resolveEnvString(temp.HealthCheckPath)

	if temp.LogErrors != "" {
		m.LogErrors, err = resolveEnvBool(temp.LogErrors, false)
		if err != nil {
			return fmt.Errorf("invalid log_errors: %w", err)
		}
	}
	m.ErrorsLogPath = resolveEnvString(temp.ErrorsLogPath)

	return nil
}

// resolveEnvString resolves environment variable if value is in format "os.environ/VAR_NAME"
func resolveEnvString(value string) string {
	const prefix = "os.environ/"
	if strings.HasPrefix(value, prefix) {
		envVar := strings.TrimPrefix(value, prefix)
		if envValue := os.Getenv(envVar); envValue != "" {
			return envValue
		}
	}
	return value
}

// resolveEnvInt resolves environment variable and converts to int
func resolveEnvInt(value string, defaultValue int) (int, error) {
	resolved := resolveEnvString(value)
	if resolved == value && value == "" {
		return defaultValue, nil
	}

	intValue, err := strconv.Atoi(resolved)
	if err != nil {
		return defaultValue, fmt.Errorf("failed to parse int from '%s': %w", resolved, err)
	}
	return intValue, nil
}

// resolveEnvBool resolves environment variable and converts to bool
func resolveEnvBool(value string, defaultValue bool) (bool, error) {
	resolved := resolveEnvString(value)
	if resolved == value && value == "" {
		return defaultValue, nil
	}

	boolValue, err := strconv.ParseBool(resolved)
	if err != nil {
		return defaultValue, fmt.Errorf("failed to parse bool from '%s': %w", resolved, err)
	}
	return boolValue, nil
}

// resolveEnvDuration resolves environment variable and converts to duration
func resolveEnvDuration(value string, defaultValue time.Duration) (time.Duration, error) {
	resolved := resolveEnvString(value)
	if resolved == value && value == "" {
		return defaultValue, nil
	}

	duration, err := time.ParseDuration(resolved)
	if err != nil {
		return defaultValue, fmt.Errorf("failed to parse duration from '%s': %w", resolved, err)
	}
	return duration, nil
}

// UnmarshalYAML implements custom unmarshaling for Fail2BanConfig
func (f *Fail2BanConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with string ban_duration
	type tempConfig struct {
		MaxAttempts int    `yaml:"max_attempts"`
		BanDuration string `yaml:"ban_duration"`
		ErrorCodes  []int  `yaml:"error_codes"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	f.MaxAttempts = temp.MaxAttempts
	f.ErrorCodes = temp.ErrorCodes

	// Parse ban_duration
	if temp.BanDuration == "permanent" || temp.BanDuration == "" {
		f.BanDuration = 0 // 0 means permanent ban
	} else {
		duration, err := time.ParseDuration(temp.BanDuration)
		if err != nil {
			return fmt.Errorf("invalid ban_duration: %w", err)
		}
		f.BanDuration = duration
	}

	return nil
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Normalize credentials
	cfg.Normalize()

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &cfg, nil
}

// Normalize cleans up configuration values
func (c *Config) Normalize() {
	// Remove /v1 suffix from base_url to avoid duplication
	for i := range c.Credentials {
		if len(c.Credentials[i].BaseURL) > 3 && c.Credentials[i].BaseURL[len(c.Credentials[i].BaseURL)-3:] == "/v1" {
			c.Credentials[i].BaseURL = c.Credentials[i].BaseURL[:len(c.Credentials[i].BaseURL)-3]
		}
	}
}

func (c *Config) Validate() error {
	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Server.Port)
	}

	if c.Server.MaxBodySizeMB <= 0 {
		return fmt.Errorf("invalid max_body_size_mb: %d", c.Server.MaxBodySizeMB)
	}

	// -1 means unlimited timeout
	if c.Server.RequestTimeout <= 0 && c.Server.RequestTimeout != -1 {
		return fmt.Errorf("invalid request_timeout: %v", c.Server.RequestTimeout)
	}

	// Validate logging level
	if c.Server.LoggingLevel != "" {
		validLevels := map[string]bool{"info": true, "debug": true, "error": true}
		if !validLevels[c.Server.LoggingLevel] {
			return fmt.Errorf("invalid logging_level: %s (must be info, debug, or error)", c.Server.LoggingLevel)
		}
	} else {
		c.Server.LoggingLevel = "info" // Default to info
	}

	// Validate master_key
	if c.Server.MasterKey == "" {
		return fmt.Errorf("master_key is required")
	}

	// Set default for default_models_rpm if not specified
	// -1 means unlimited RPM
	if c.Server.DefaultModelsRPM == 0 {
		c.Server.DefaultModelsRPM = 50 // Default value
	} else if c.Server.DefaultModelsRPM < -1 {
		return fmt.Errorf("invalid default_models_rpm: %d (must be -1 for unlimited or positive number)", c.Server.DefaultModelsRPM)
	}

	if c.Fail2Ban.MaxAttempts <= 0 {
		return fmt.Errorf("invalid max_attempts: %d", c.Fail2Ban.MaxAttempts)
	}

	if len(c.Credentials) == 0 {
		return fmt.Errorf("no credentials configured")
	}

	for i, cred := range c.Credentials {
		if cred.Name == "" {
			return fmt.Errorf("credential %d: name is required", i)
		}

		// Validate provider type
		if !cred.Type.IsValid() {
			return fmt.Errorf("credential %s: invalid type: %s (must be 'openai' or 'vertex-ai')", cred.Name, cred.Type)
		}

		// Vertex AI specific validation
		if cred.Type == ProviderTypeVertexAI {
			// For Vertex AI, project_id and location are required
			if cred.ProjectID == "" {
				return fmt.Errorf("credential %s: project_id is required for vertex-ai type", cred.Name)
			}
			if cred.Location == "" {
				return fmt.Errorf("credential %s: location is required for vertex-ai type", cred.Name)
			}
			// API Key is required for Vertex AI (Express Mode)
			if cred.APIKey == "" && cred.CredentialsFile == "" && cred.CredentialsJSON == "" {
				return fmt.Errorf("credential %s: api_key, credentials_file, or credentials_json is required for vertex-ai type", cred.Name)
			}
			// base_url is optional for Vertex AI (will be constructed dynamically)
		} else {
			// For non-Vertex AI credentials, require APIKey and BaseURL
			if cred.APIKey == "" {
				return fmt.Errorf("credential %s: api_key is required", cred.Name)
			}
			if cred.BaseURL == "" {
				return fmt.Errorf("credential %s: base_url is required", cred.Name)
			}
			// Validate base_url is a valid URL
			parsedURL, err := url.Parse(cred.BaseURL)
			if err != nil {
				return fmt.Errorf("credential %s: invalid base_url: %w", cred.Name, err)
			}
			if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
				return fmt.Errorf("credential %s: base_url must use http or https scheme, got: %s", cred.Name, parsedURL.Scheme)
			}
			if parsedURL.Host == "" {
				return fmt.Errorf("credential %s: base_url must have a host", cred.Name)
			}
		}

		// -1 means unlimited RPM
		if cred.RPM <= 0 && cred.RPM != -1 {
			return fmt.Errorf("credential %s: invalid rpm: %d (must be -1 for unlimited or positive number)", cred.Name, cred.RPM)
		}
		// TPM: 0 or -1 means unlimited, positive means limited
		if cred.TPM < -1 {
			return fmt.Errorf("credential %s: invalid tpm: %d (must be -1 or 0 for unlimited, or positive number)", cred.Name, cred.TPM)
		}
	}

	return nil
}
