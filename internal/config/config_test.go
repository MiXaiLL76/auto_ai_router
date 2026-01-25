package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoad_ValidConfig(t *testing.T) {
	// Create temporary config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 100
  request_timeout: 30s
  logging_level: info
  replace_v1_models: true
  master_key: "sk-test-master-key"
  default_models_rpm: 50

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 429, 500, 502, 503, 504]

credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60

  - name: "provider_2"
    type: "openai"
    api_key: "sk-yyyy"
    base_url: "https://api.custom-provider.com"
    rpm: 120

monitoring:
  prometheus_enabled: true
  health_check_path: "/health"
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Validate server config
	assert.Equal(t, 8080, cfg.Server.Port)
	assert.Equal(t, 100, cfg.Server.MaxBodySizeMB)
	assert.Equal(t, 30*time.Second, cfg.Server.RequestTimeout)
	assert.Equal(t, "info", cfg.Server.LoggingLevel)
	assert.True(t, cfg.Server.ReplaceV1Models)
	assert.Equal(t, "sk-test-master-key", cfg.Server.MasterKey)
	assert.Equal(t, 50, cfg.Server.DefaultModelsRPM)

	// Validate fail2ban config
	assert.Equal(t, 3, cfg.Fail2Ban.MaxAttempts)
	assert.Equal(t, time.Duration(0), cfg.Fail2Ban.BanDuration)
	assert.Equal(t, []int{401, 403, 429, 500, 502, 503, 504}, cfg.Fail2Ban.ErrorCodes)

	// Validate credentials
	assert.Len(t, cfg.Credentials, 2)
	assert.Equal(t, "provider_1", cfg.Credentials[0].Name)
	assert.Equal(t, 60, cfg.Credentials[0].RPM)

	// Validate monitoring
	assert.True(t, cfg.Monitoring.PrometheusEnabled)
	assert.Equal(t, "/health", cfg.Monitoring.HealthCheckPath)
}

func TestLoad_FileNotFound(t *testing.T) {
	_, err := Load("/non/existent/path.yaml")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to read config file")
}

func TestLoad_InvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "invalid.yaml")

	invalidContent := `
server:
  port: invalid_port
  - this is not valid yaml
`
	err := os.WriteFile(configPath, []byte(invalidContent), 0644)
	require.NoError(t, err)

	_, err = Load(configPath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse config file")
}

func TestConfig_Validate_InvalidPort(t *testing.T) {
	tests := []struct {
		name    string
		port    int
		wantErr bool
	}{
		{"valid port", 8080, false},
		{"min valid port", 1, false},
		{"max valid port", 65535, false},
		{"port zero", 0, true},
		{"negative port", -1, true},
		{"port too high", 70000, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Server: ServerConfig{
					Port:           tt.port,
					MaxBodySizeMB:  10,
					MasterKey:      "test-key",
					RequestTimeout: 30 * time.Second,
				},
				Credentials: []CredentialConfig{
					{Name: "test", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Validate_NoCredentials(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{},
		Fail2Ban:    Fail2BanConfig{MaxAttempts: 3},
	}

	err := cfg.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no credentials configured")
}

func TestConfig_Validate_MissingMasterKey(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{
			{Name: "test", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
	}

	err := cfg.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "master_key is required")
}

func TestConfig_Validate_InvalidBaseURL(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		wantErr bool
	}{
		{"valid https", "https://api.openai.com", false},
		{"valid http", "http://localhost:8080", false},
		{"invalid scheme", "ftp://test.com", true},
		{"no scheme", "api.openai.com", true},
		{"no host", "https://", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Server: ServerConfig{
					Port:           8080,
					MaxBodySizeMB:  10,
					MasterKey:      "test-key",
					RequestTimeout: 30 * time.Second,
				},
				Credentials: []CredentialConfig{
					{Name: "test", APIKey: "key", BaseURL: tt.baseURL, RPM: 10},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Validate_InvalidRPM(t *testing.T) {
	tests := []struct {
		name    string
		rpm     int
		wantErr bool
	}{
		{"valid rpm", 100, false},
		{"unlimited rpm", -1, false},
		{"zero rpm", 0, true},
		{"negative rpm (not -1)", -5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Server: ServerConfig{
					Port:           8080,
					MaxBodySizeMB:  10,
					MasterKey:      "test-key",
					RequestTimeout: 30 * time.Second,
				},
				Credentials: []CredentialConfig{
					{Name: "test", APIKey: "key", BaseURL: "http://test.com", RPM: tt.rpm},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Normalize_RemovesV1Suffix(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{
			{Name: "test1", APIKey: "key1", BaseURL: "https://api.openai.com/v1", RPM: 10},
			{Name: "test2", APIKey: "key2", BaseURL: "https://api.custom.com", RPM: 10},
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
	}

	cfg.Normalize()

	assert.Equal(t, "https://api.openai.com", cfg.Credentials[0].BaseURL)
	assert.Equal(t, "https://api.custom.com", cfg.Credentials[1].BaseURL)
}

func TestFail2BanConfig_UnmarshalYAML_Permanent(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  master_key: "test-key"
  request_timeout: 30s

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403]

credentials:
  - name: "test"
    api_key: "key"
    base_url: "http://test.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.Equal(t, time.Duration(0), cfg.Fail2Ban.BanDuration)
}

func TestFail2BanConfig_UnmarshalYAML_Duration(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  master_key: "test-key"
  request_timeout: 30s

fail2ban:
  max_attempts: 3
  ban_duration: 5m
  error_codes: [401, 403]

credentials:
  - name: "test"
    api_key: "key"
    base_url: "http://test.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.Equal(t, 5*time.Minute, cfg.Fail2Ban.BanDuration)
}

func TestFail2BanConfig_UnmarshalYAML_InvalidDuration(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  master_key: "test-key"

fail2ban:
  max_attempts: 3
  ban_duration: invalid_duration
  error_codes: [401, 403]

credentials:
  - name: "test"
    api_key: "key"
    base_url: "http://test.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	_, err = Load(configPath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid ban_duration")
}

func TestConfig_Validate_LoggingLevel(t *testing.T) {
	tests := []struct {
		name         string
		loggingLevel string
		wantErr      bool
		expected     string
	}{
		{"valid info", "info", false, "info"},
		{"valid debug", "debug", false, "debug"},
		{"valid error", "error", false, "error"},
		{"invalid level", "warning", true, ""},
		{"empty defaults to info", "", false, "info"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Server: ServerConfig{
					Port:           8080,
					MaxBodySizeMB:  10,
					MasterKey:      "test-key",
					LoggingLevel:   tt.loggingLevel,
					RequestTimeout: 30 * time.Second,
				},
				Credentials: []CredentialConfig{
					{Name: "test", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, cfg.Server.LoggingLevel)
			}
		})
	}
}

func TestConfig_Validate_DefaultModelsRPM(t *testing.T) {
	tests := []struct {
		name             string
		defaultModelsRPM int
		wantErr          bool
		expected         int
	}{
		{"valid rpm", 100, false, 100},
		{"unlimited rpm", -1, false, -1},
		{"zero defaults to 50", 0, false, 50},
		{"negative (not -1)", -5, true, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Server: ServerConfig{
					Port:             8080,
					MaxBodySizeMB:    10,
					MasterKey:        "test-key",
					DefaultModelsRPM: tt.defaultModelsRPM,
					RequestTimeout:   30 * time.Second,
				},
				Credentials: []CredentialConfig{
					{Name: "test", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, cfg.Server.DefaultModelsRPM)
			}
		})
	}
}
