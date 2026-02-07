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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: tt.baseURL, RPM: 10},
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: tt.rpm},
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
			{Name: "test1", Type: "openai", APIKey: "key1", BaseURL: "https://api.openai.com/v1", RPM: 10},
			{Name: "test2", Type: "openai", APIKey: "key2", BaseURL: "https://api.custom.com", RPM: 10},
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
    type: "openai"
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
    type: "openai"
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
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

func TestLoad_EnvVariables(t *testing.T) {
	// Set environment variables for testing
	require.NoError(t, os.Setenv("TEST_PORT", "9090"))
	require.NoError(t, os.Setenv("TEST_MAX_BODY_SIZE", "200"))
	require.NoError(t, os.Setenv("TEST_REQUEST_TIMEOUT", "60s"))
	require.NoError(t, os.Setenv("TEST_LOGGING_LEVEL", "error"))
	require.NoError(t, os.Setenv("TEST_MASTER_KEY", "sk-env-master-key"))
	require.NoError(t, os.Setenv("TEST_DEFAULT_MODELS_RPM", "100"))
	require.NoError(t, os.Setenv("TEST_CRED_NAME", "env_credential"))
	require.NoError(t, os.Setenv("TEST_CRED_TYPE", "openai"))
	require.NoError(t, os.Setenv("TEST_CRED_API_KEY", "sk-env-api-key"))
	require.NoError(t, os.Setenv("TEST_CRED_BASE_URL", "https://env.example.com"))
	require.NoError(t, os.Setenv("TEST_CRED_RPM", "150"))
	require.NoError(t, os.Setenv("TEST_CRED_TPM", "50000"))
	require.NoError(t, os.Setenv("TEST_PROMETHEUS_ENABLED", "false"))
	require.NoError(t, os.Setenv("TEST_HEALTH_CHECK_PATH", "/status"))

	defer func() {
		// Cleanup
		_ = os.Unsetenv("TEST_PORT")
		_ = os.Unsetenv("TEST_MAX_BODY_SIZE")
		_ = os.Unsetenv("TEST_REQUEST_TIMEOUT")
		_ = os.Unsetenv("TEST_LOGGING_LEVEL")
		_ = os.Unsetenv("TEST_MASTER_KEY")
		_ = os.Unsetenv("TEST_DEFAULT_MODELS_RPM")
		_ = os.Unsetenv("TEST_CRED_NAME")
		_ = os.Unsetenv("TEST_CRED_TYPE")
		_ = os.Unsetenv("TEST_CRED_API_KEY")
		_ = os.Unsetenv("TEST_CRED_BASE_URL")
		_ = os.Unsetenv("TEST_CRED_RPM")
		_ = os.Unsetenv("TEST_CRED_TPM")
		_ = os.Unsetenv("TEST_PROMETHEUS_ENABLED")
		_ = os.Unsetenv("TEST_HEALTH_CHECK_PATH")
	}()

	// Create temporary config file with env variables
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: os.environ/TEST_PORT
  max_body_size_mb: os.environ/TEST_MAX_BODY_SIZE
  request_timeout: os.environ/TEST_REQUEST_TIMEOUT
  logging_level: os.environ/TEST_LOGGING_LEVEL
  master_key: os.environ/TEST_MASTER_KEY
  default_models_rpm: os.environ/TEST_DEFAULT_MODELS_RPM

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 500]

credentials:
  - name: os.environ/TEST_CRED_NAME
    type: os.environ/TEST_CRED_TYPE
    api_key: os.environ/TEST_CRED_API_KEY
    base_url: os.environ/TEST_CRED_BASE_URL
    rpm: os.environ/TEST_CRED_RPM
    tpm: os.environ/TEST_CRED_TPM

monitoring:
  prometheus_enabled: os.environ/TEST_PROMETHEUS_ENABLED
  health_check_path: os.environ/TEST_HEALTH_CHECK_PATH
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	// Load config
	cfg, err := Load(configPath)
	require.NoError(t, err)

	// Verify server config
	assert.Equal(t, 9090, cfg.Server.Port)
	assert.Equal(t, 200, cfg.Server.MaxBodySizeMB)
	assert.Equal(t, 60*time.Second, cfg.Server.RequestTimeout)
	assert.Equal(t, "error", cfg.Server.LoggingLevel)
	assert.Equal(t, "sk-env-master-key", cfg.Server.MasterKey)
	assert.Equal(t, 100, cfg.Server.DefaultModelsRPM)

	// Verify credential config
	require.Len(t, cfg.Credentials, 1)
	assert.Equal(t, "env_credential", cfg.Credentials[0].Name)
	assert.Equal(t, ProviderType("openai"), cfg.Credentials[0].Type)
	assert.Equal(t, "sk-env-api-key", cfg.Credentials[0].APIKey)
	assert.Equal(t, "https://env.example.com", cfg.Credentials[0].BaseURL)
	assert.Equal(t, 150, cfg.Credentials[0].RPM)
	assert.Equal(t, 50000, cfg.Credentials[0].TPM)

	// Verify monitoring config
	assert.Equal(t, false, cfg.Monitoring.PrometheusEnabled)
	assert.Equal(t, "/status", cfg.Monitoring.HealthCheckPath)
}

func TestLoad_EnvVariables_Mixed(t *testing.T) {
	// Set only some environment variables
	require.NoError(t, os.Setenv("TEST_MASTER_KEY", "sk-from-env"))
	require.NoError(t, os.Setenv("TEST_CRED_API_KEY", "sk-cred-from-env"))

	defer func() {
		_ = os.Unsetenv("TEST_MASTER_KEY")
		_ = os.Unsetenv("TEST_CRED_API_KEY")
	}()

	// Create temporary config file mixing env variables and direct values
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 100
  request_timeout: 30s
  logging_level: info
  master_key: os.environ/TEST_MASTER_KEY
  default_models_rpm: 50

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 500]

credentials:
  - name: "static_provider"
    type: "openai"
    api_key: os.environ/TEST_CRED_API_KEY
    base_url: "https://api.openai.com"
    rpm: 60

monitoring:
  prometheus_enabled: true
  health_check_path: "/health"
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	// Load config
	cfg, err := Load(configPath)
	require.NoError(t, err)

	// Verify mixed config
	assert.Equal(t, 8080, cfg.Server.Port)
	assert.Equal(t, "sk-from-env", cfg.Server.MasterKey)
	assert.Equal(t, "sk-cred-from-env", cfg.Credentials[0].APIKey)
	assert.Equal(t, "static_provider", cfg.Credentials[0].Name)
}

func TestProviderType_IsValid(t *testing.T) {
	tests := []struct {
		name     string
		provider ProviderType
		valid    bool
	}{
		{"openai", ProviderTypeOpenAI, true},
		{"vertex-ai", ProviderTypeVertexAI, true},
		{"invalid", ProviderType("azure"), false},
		{"empty", ProviderType(""), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.valid, tt.provider.IsValid())
		})
	}
}

func TestConfig_Validate_VertexAI(t *testing.T) {
	tests := []struct {
		name      string
		projectID string
		location  string
		apiKey    string
		wantErr   bool
		errMsg    string
	}{
		{
			name:      "valid with api_key",
			projectID: "proj-123",
			location:  "us-central1",
			apiKey:    "sk-vertex-key",
			wantErr:   false,
		},
		{
			name:      "missing project_id",
			projectID: "",
			location:  "us-central1",
			apiKey:    "sk-vertex-key",
			wantErr:   true,
			errMsg:    "project_id is required",
		},
		{
			name:      "missing location",
			projectID: "proj-123",
			location:  "",
			apiKey:    "sk-vertex-key",
			wantErr:   true,
			errMsg:    "location is required",
		},
		{
			name:      "missing all credentials",
			projectID: "proj-123",
			location:  "us-central1",
			apiKey:    "",
			wantErr:   true,
			errMsg:    "api_key, credentials_file, or credentials_json is required",
		},
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
					{
						Name:      "vertex",
						Type:      ProviderTypeVertexAI,
						ProjectID: tt.projectID,
						Location:  tt.location,
						APIKey:    tt.apiKey,
						RPM:       10,
					},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Validate_TPM(t *testing.T) {
	tests := []struct {
		name    string
		tpm     int
		wantErr bool
	}{
		{"valid tpm", 1000, false},
		{"zero (unlimited)", 0, false},
		{"unlimited", -1, false},
		{"negative invalid", -5, true},
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10, TPM: tt.tpm},
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

// Fallback Configuration Tests

func TestConfig_ValidateFallback_AllowsFallbackOnAnyType(t *testing.T) {
	// Valid: is_fallback can be set on any credential type
	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{
			{Name: "openai-primary", Type: "openai", BaseURL: "http://a.com", APIKey: "key", RPM: 10, IsFallback: false},
			{Name: "openai-fallback", Type: "openai", BaseURL: "http://b.com", APIKey: "key", RPM: 10, IsFallback: true},
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
	}

	err := cfg.Validate()
	assert.NoError(t, err, "is_fallback should be allowed on any credential type")
}

func TestConfig_UnmarshalYAML_ModelPricesLink(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Set environment variable
	if err := os.Setenv("MODEL_PRICES_URL", "https://example.com/prices.json"); err != nil {
		t.Fatalf("Failed to set env var: %v", err)
	}
	defer func() {
		_ = os.Unsetenv("MODEL_PRICES_URL")
	}()

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  logging_level: info
  master_key: "sk-test"
  model_prices_link: os.environ/MODEL_PRICES_URL

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"

monitoring:
  prometheus_enabled: false
  health_check_path: "/health"
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Verify that ModelPricesLink was properly resolved from environment variable
	assert.Equal(t, "https://example.com/prices.json", cfg.Server.ModelPricesLink)
}

func TestConfig_UnmarshalYAML_ModelPricesLink_Direct(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  logging_level: info
  master_key: "sk-test"
  model_prices_link: "/path/to/prices.json"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"

monitoring:
  prometheus_enabled: false
  health_check_path: "/health"
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Verify that ModelPricesLink was set directly
	assert.Equal(t, "/path/to/prices.json", cfg.Server.ModelPricesLink)
}
