package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
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
    auth_type: "bearer"
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
	assert.Equal(t, "native", cfg.Server.ResponseCompatibility)
	assert.Equal(t, ResponseHeaderModePassthrough, cfg.Server.ResponseHeaders.Mode)

	// Validate fail2ban config
	assert.Equal(t, 3, cfg.Fail2Ban.MaxAttempts)
	assert.Equal(t, time.Duration(0), cfg.Fail2Ban.BanDuration)
	assert.Equal(t, []int{401, 403, 429, 500, 502, 503, 504}, cfg.Fail2Ban.ErrorCodes)

	// Validate credentials
	assert.Len(t, cfg.Credentials, 2)
	assert.Equal(t, "provider_1", cfg.Credentials[0].Name)
	assert.Equal(t, 60, cfg.Credentials[0].RPM)
	assert.Equal(t, "bearer", cfg.Credentials[1].AuthType)

	// Validate monitoring
	assert.True(t, cfg.Monitoring.PrometheusEnabled)
	assert.Equal(t, "/health", cfg.Monitoring.HealthCheckPath)
}

func TestLoad_ResponseHeadersAllowlist(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configContent := `
server:
  master_key: sk-test
  response_headers:
    mode: allowlist
credentials:
  - name: provider
    type: openai
    api_key: sk-provider
    base_url: https://api.openai.com
    rpm: 10
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0600))

	cfg, err := Load(configPath)

	require.NoError(t, err)
	assert.Equal(t, ResponseHeaderModeAllowlist, cfg.Server.ResponseHeaders.Mode)
}

func TestCredentialConfigReasoningOnly(t *testing.T) {
	t.Setenv("TEST_REASONING_ONLY", "true")

	var credential CredentialConfig
	err := yaml.Unmarshal([]byte(`
name: anthropic-reasoning
type: anthropic
api_key: key
base_url: https://api.anthropic.com
reasoning_only: os.environ/TEST_REASONING_ONLY
`), &credential)

	require.NoError(t, err)
	assert.True(t, credential.ReasoningOnly)
}

func TestLoad_ResponseHeadersModeFromEnvironment(t *testing.T) {
	t.Setenv("TEST_RESPONSE_HEADER_MODE", "ALLOWLIST")
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configContent := `
server:
  master_key: sk-test
  response_headers:
    mode: os.environ/TEST_RESPONSE_HEADER_MODE
credentials:
  - name: provider
    type: openai
    api_key: sk-provider
    base_url: https://api.openai.com
    rpm: 10
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0600))

	cfg, err := Load(configPath)

	require.NoError(t, err)
	assert.Equal(t, ResponseHeaderModeAllowlist, cfg.Server.ResponseHeaders.Mode)
}

func TestLoad_ResponseHeadersModeFromUnsetEnvironment(t *testing.T) {
	const envName = "UNSET_RESPONSE_HEADER_MODE"
	previous, existed := os.LookupEnv(envName)
	require.NoError(t, os.Unsetenv(envName))
	t.Cleanup(func() {
		if existed {
			_ = os.Setenv(envName, previous)
		}
	})
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configContent := `
server:
  master_key: sk-test
  response_headers:
    mode: os.environ/UNSET_RESPONSE_HEADER_MODE
credentials:
  - name: provider
    type: openai
    api_key: sk-provider
    base_url: https://api.openai.com
    rpm: 10
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0600))

	_, err := Load(configPath)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "UNSET_RESPONSE_HEADER_MODE")
}

func TestLoad_InvalidResponseHeadersMode(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configContent := `
server:
  master_key: sk-test
  response_headers:
    mode: denylist
credentials:
  - name: provider
    type: openai
    api_key: sk-provider
    base_url: https://api.openai.com
    rpm: 10
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0600))

	_, err := Load(configPath)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "response_headers.mode")
}

func TestCredentialConfig_UnmarshalScopes(t *testing.T) {
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte(`
name: provider
type: openai
api_key: sk-test
base_url: https://api.openai.com
scopes: [Team-A, team-a, " "]
denied_scopes: [premium]
forbidden_scopes: [blocked]
`), &cred)

	require.NoError(t, err)
	assert.Equal(t, []string{"team-a"}, cred.Scopes)
	assert.Equal(t, []string{"premium", "blocked"}, cred.DeniedScopes)
}

func TestCredentialConfig_UnmarshalRejectsLegacyProxyUsageFormat(t *testing.T) {
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte(`
name: proxy
type: proxy
base_url: http://proxy.example
proxy_usage_format: normalized
`), &cred)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "proxy_usage_format is no longer supported")
	assert.Contains(t, err.Error(), "type: air")
}

func TestConfig_ValidateAIRCredential(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{{
			Name:    "upstream-air",
			Type:    ProviderTypeAIR,
			BaseURL: "http://air.example",
			RPM:     -1,
			TPM:     -1,
		}},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
	}

	require.NoError(t, cfg.Validate())
}

func TestCredentialConfig_ScopeExpressionPreservesIndependentGroups(t *testing.T) {
	cred := CredentialConfig{
		Scopes:         []string{"team-a"},
		ProviderScopes: []string{"team-b"},
	}

	expression := cred.ScopeExpression()
	assert.True(t, scope.NewContext([]string{"team-a", "team-b"}, nil).AllowsExpression(expression))
	assert.False(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(expression))
}

func TestConfigResponseCompatibility(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Port:                  8080,
			MaxBodySizeMB:         10,
			MasterKey:             "test-key",
			RequestTimeout:        30 * time.Second,
			ResponseCompatibility: "litellm",
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
		Credentials: []CredentialConfig{
			{Name: "openai", Type: ProviderTypeOpenAI, APIKey: "key", BaseURL: "https://api.openai.com", RPM: 1},
		},
	}
	require.NoError(t, cfg.Validate())
	assert.Equal(t, "litellm", cfg.Server.ResponseCompatibility)

	cfg.Server.ResponseCompatibility = "unknown"
	err := cfg.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "response_compatibility")
}

func TestServerConfigUnmarshalResponseCompatibility(t *testing.T) {
	var server ServerConfig
	require.NoError(t, yaml.Unmarshal([]byte("response_compatibility: LiteLLM\n"), &server))
	assert.Equal(t, "litellm", server.ResponseCompatibility)
}

func TestConfig_Validate_InvalidAuthType(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{
			{Name: "test", Type: "anthropic", APIKey: "key", BaseURL: "http://test.com", AuthType: "basic", RPM: 10},
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
	}

	err := cfg.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid auth_type")
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
		{"zero defaults to unlimited", 0, false, -1},
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
	assert.Equal(t, "/health", cfg.Monitoring.HealthCheckPath)
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
		{"cometapi", ProviderTypeCometAPI, true},
		{"air", ProviderTypeAIR, true},
		{"proman", ProviderTypeProMan, true},
		{"invalid", ProviderType("azure"), false},
		{"empty", ProviderType(""), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.valid, tt.provider.IsValid())
		})
	}
}

func TestProviderType_IsProxyLike(t *testing.T) {
	assert.True(t, ProviderTypeProxy.IsProxyLike())
	assert.True(t, ProviderTypeAIR.IsProxyLike())
	assert.False(t, ProviderTypeOpenAI.IsProxyLike())
	assert.False(t, ProviderTypeVertexAI.IsProxyLike())
	assert.False(t, ProviderTypeProMan.IsProxyLike())
}

func TestIsGoogleGeminiProvider(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		want     bool
	}{
		{"gemini", "gemini", true},
		{"vertex", "vertex", true},
		{"vertex_ai", "vertex_ai", true},
		{"google", "google", true},
		{"google_ai_studio", "google_ai_studio", true},
		{"mixed_case", "GoogleAI", true},
		{"openai", "openai", false},
		{"anthropic", "anthropic", false},
		{"empty", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, IsGoogleGeminiProvider(tt.provider))
		})
	}
}

func TestCredentialConfig_NormalizeCometAPIProviderType(t *testing.T) {
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte(`
name: comet
type: comet-api
api_key: key
base_url: https://api.cometapi.com/v1
rpm: 60
`), &cred)

	require.NoError(t, err)
	assert.Equal(t, ProviderTypeCometAPI, cred.Type)
}

func TestCredentialConfig_NormalizeAIRProviderTypeAliases(t *testing.T) {
	tests := []struct {
		name string
		raw  string
	}{
		{name: "air", raw: "air"},
		{name: "aar", raw: "aar"},
		{name: "auto-ai-router", raw: "auto-ai-router"},
		{name: "auto_ai_router", raw: "auto_ai_router"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cred CredentialConfig
			err := yaml.Unmarshal([]byte(`
name: upstream
type: `+tt.raw+`
base_url: https://air.example
rpm: 60
`), &cred)

			require.NoError(t, err)
			assert.Equal(t, ProviderTypeAIR, cred.Type)
			assert.True(t, cred.IsProxyLike())
		})
	}
}

func TestCredentialConfig_NormalizeProManProviderType(t *testing.T) {
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte(`
name: proman
type: pro-man
api_key: key
base_url: https://api.proman.ai/v1
rpm: 60
`), &cred)

	require.NoError(t, err)
	assert.Equal(t, ProviderTypeProMan, cred.Type)
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

func TestConfig_ValidateFallbackPriority(t *testing.T) {
	tests := []struct {
		name       string
		priority   int
		isFallback bool
		wantErr    bool
	}{
		{name: "unset", priority: 0, wantErr: false},
		{name: "positive", priority: 10, wantErr: false},
		{name: "negative", priority: -1, wantErr: true},
		{name: "fallback with priority", priority: 10, isFallback: true, wantErr: true},
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10, IsFallback: tt.isFallback, FallbackPriority: tt.priority},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}

			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "invalid fallback_priority")
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCredentialConfig_UnmarshalYAML_FallbackPriority(t *testing.T) {
	t.Setenv("TEST_FALLBACK_PRIORITY", "20")

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  master_key: "test-key"
  request_timeout: 30s

credentials:
  - name: "cheapgpt"
    type: "anthropic"
    api_key: "key"
    base_url: "http://cheapgpt.com"
    rpm: 100
    fallback_priority: 10
  - name: "cometapi"
    type: "anthropic"
    api_key: "key2"
    base_url: "http://cometapi.com"
    rpm: 100
    fallback_priority: os.environ/TEST_FALLBACK_PRIORITY

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [429]
`

	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	require.Len(t, cfg.Credentials, 2)
	assert.Equal(t, 10, cfg.Credentials[0].FallbackPriority)
	assert.Equal(t, 20, cfg.Credentials[1].FallbackPriority)
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

func TestConfig_Validate_DatabaseURL(t *testing.T) {
	tests := []struct {
		name        string
		databaseURL string
		wantErr     bool
		errContains string
	}{
		{
			name:        "postgres:// prefix passes validation",
			databaseURL: "postgres://user:pass@localhost:5432/litellm",
			wantErr:     false,
		},
		{
			name:        "postgresql:// prefix passes validation",
			databaseURL: "postgresql://user:pass@localhost:5432/litellm",
			wantErr:     false,
		},
		{
			name:        "mysql:// prefix fails validation",
			databaseURL: "mysql://user:pass@localhost:3306/litellm",
			wantErr:     true,
			errContains: "must start with postgres:// or postgresql://",
		},
		{
			name:        "empty string when enabled fails validation",
			databaseURL: "",
			wantErr:     true,
			errContains: "database_url is required when enabled",
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
				LiteLLMDB: LiteLLMDBConfig{
					Enabled:     true,
					DatabaseURL: tt.databaseURL,
					LogWorkers:  4,
				},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Validate_MaxProviderRetries(t *testing.T) {
	tests := []struct {
		name               string
		maxProviderRetries int
		wantErr            bool
	}{
		{"default value (2)", 2, false},
		{"zero retries", 0, false},
		{"high value", 10, false},
		{"negative value", -1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Server: ServerConfig{
					Port:               8080,
					MaxBodySizeMB:      10,
					MasterKey:          "test-key",
					RequestTimeout:     30 * time.Second,
					MaxProviderRetries: tt.maxProviderRetries,
				},
				Credentials: []CredentialConfig{
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
				},
				Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
			}
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "invalid max_provider_retries")
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestLoad_MaxProviderRetries_Default(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.Equal(t, 2, cfg.Server.MaxProviderRetries, "Default MaxProviderRetries should be 2")
}

func TestLoad_MaxProviderRetries_Custom(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"
  max_provider_retries: 5

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.Equal(t, 5, cfg.Server.MaxProviderRetries, "Custom MaxProviderRetries should be 5")
}

func TestLoad_SessionStickyDefaults(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.True(t, cfg.Server.SessionStickyEnabled)
	assert.Equal(t, 0, cfg.Server.SessionStickyTTL)
}

func TestLoad_SessionStickyCustom(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"
  session_sticky_enabled: false
  session_sticky_ttl_minutes: 15

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.False(t, cfg.Server.SessionStickyEnabled)
	assert.Equal(t, 15, cfg.Server.SessionStickyTTL)
}

func TestLoad_TiktokenEnabledDefault(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.True(t, cfg.Server.TiktokenEnabled)
}

func TestLoad_TiktokenEnabledExplicitFalse(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"
  tiktoken_enabled: false

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.False(t, cfg.Server.TiktokenEnabled)
}

func TestServerConfigStrictAllTeamModelsACL(t *testing.T) {
	var defaults ServerConfig
	require.NoError(t, yaml.Unmarshal([]byte("{}"), &defaults))
	assert.False(t, defaults.StrictAllTeamModelsACL)

	var strict ServerConfig
	require.NoError(t, yaml.Unmarshal([]byte("strict_all_team_models_acl: true"), &strict))
	assert.True(t, strict.StrictAllTeamModelsACL)
}

func TestLoad_ModelAlias(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false

model_alias:
  gpt-4: gpt-4o
  claude: claude-sonnet-4-20250514
  gemini: gemini-2.5-flash
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.NotNil(t, cfg)

	// Verify aliases are loaded
	assert.Len(t, cfg.ModelAlias, 3)
	assert.Equal(t, "gpt-4o", cfg.ModelAlias["gpt-4"])
	assert.Equal(t, "claude-sonnet-4-20250514", cfg.ModelAlias["claude"])
	assert.Equal(t, "gemini-2.5-flash", cfg.ModelAlias["gemini"])
}

func TestLoad_ModelAlias_WithEnvVars(t *testing.T) {
	require.NoError(t, os.Setenv("TEST_ALIAS_TARGET", "gpt-4o-mini"))
	defer func() { _ = os.Unsetenv("TEST_ALIAS_TARGET") }()

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false

model_alias:
  gpt-4: os.environ/TEST_ALIAS_TARGET
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)

	assert.Len(t, cfg.ModelAlias, 1)
	assert.Equal(t, "gpt-4o-mini", cfg.ModelAlias["gpt-4"])
}

func TestLoad_ModelAlias_Empty(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

credentials:
  - name: "test"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 10

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)

	// No model_alias section → nil map
	assert.Equal(t, 0, len(cfg.ModelAlias))
}

// ---------------------------------------------------------------------------
// YAML anchor / alias tests
// ---------------------------------------------------------------------------

// TestLoad_ModelAnchors_ListInCredentials verifies that a YAML list anchor
// referenced via *alias inside a credential's models: field is correctly
// decoded and expanded into cfg.Models after Load.
func TestLoad_ModelAnchors_ListInCredentials(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

x-model-templates:
  base-models: &base-models
    - name: gemini-2.5-flash
      rpm: 100
      tpm: 50000
    - name: gemini-2.5-pro
      rpm: 50
      tpm: 100000

credentials:
  - name: "vertex_v1"
    type: "vertex-ai"
    project_id: "proj-123"
    location: "us-central1"
    credentials_json: '{"type":"service_account"}'
    rpm: 100
    models: *base-models

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)

	// Both models should be extracted from the credential and added to cfg.Models
	require.Len(t, cfg.Models, 2)
	assert.Equal(t, "gemini-2.5-flash", cfg.Models[0].Name)
	assert.Equal(t, 100, cfg.Models[0].RPM)
	assert.Equal(t, 50000, cfg.Models[0].TPM)
	assert.Equal(t, "vertex_v1", cfg.Models[0].Credential)

	assert.Equal(t, "gemini-2.5-pro", cfg.Models[1].Name)
	assert.Equal(t, 50, cfg.Models[1].RPM)
	assert.Equal(t, "vertex_v1", cfg.Models[1].Credential)
}

// TestLoad_ModelAnchors_SameListMultipleCredentials verifies that the same
// list anchor can be referenced from multiple credentials and each model copy
// gets the correct credential name.
func TestLoad_ModelAnchors_SameListMultipleCredentials(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

x-model-templates:
  vertex-models: &vertex-models
    - name: gemini-2.5-flash
      rpm: 100
      tpm: 50000

credentials:
  - name: "vertex_v1"
    type: "vertex-ai"
    project_id: "proj-1"
    location: "us-central1"
    credentials_json: '{"type":"service_account"}'
    rpm: 100
    models: *vertex-models

  - name: "vertex_v2"
    type: "vertex-ai"
    project_id: "proj-2"
    location: "us-central1"
    credentials_json: '{"type":"service_account"}'
    rpm: 100
    models: *vertex-models

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)

	require.Len(t, cfg.Models, 2)
	assert.Equal(t, "gemini-2.5-flash", cfg.Models[0].Name)
	assert.Equal(t, "vertex_v1", cfg.Models[0].Credential)
	assert.Equal(t, "gemini-2.5-flash", cfg.Models[1].Name)
	assert.Equal(t, "vertex_v2", cfg.Models[1].Credential)
}

// TestLoad_ModelAnchors_FlattenInTopLevelModels verifies that a list anchor
// used as an element inside the top-level models: sequence is flattened into
// the parent sequence (i.e. "- *list-anchor" expands all items).
func TestLoad_ModelAnchors_FlattenInTopLevelModels(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

x-model-templates:
  shared-models: &shared-models
    - name: gemini-2.5-flash
      credential: vertex_v1
      rpm: 100
      tpm: 50000
    - name: gemini-2.5-pro
      credential: vertex_v1
      rpm: 50
      tpm: 100000

credentials:
  - name: "vertex_v1"
    type: "vertex-ai"
    project_id: "proj-1"
    location: "us-central1"
    credentials_json: '{"type":"service_account"}'
    rpm: 100

monitoring:
  prometheus_enabled: false

models:
  - *shared-models
  - name: gpt-4
    credential: vertex_v1
    rpm: 60
    tpm: 80000
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)

	// 2 from anchor + 1 inline = 3 total
	require.Len(t, cfg.Models, 3)
	assert.Equal(t, "gemini-2.5-flash", cfg.Models[0].Name)
	assert.Equal(t, "gemini-2.5-pro", cfg.Models[1].Name)
	assert.Equal(t, "gpt-4", cfg.Models[2].Name)
	assert.Equal(t, 60, cfg.Models[2].RPM)
}

// TestLoad_ModelAnchors_SingleModelAnchor verifies that an anchor on a single
// model mapping (not a list) works when referenced in a credential's models list.
func TestLoad_ModelAnchors_SingleModelAnchor(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  request_timeout: 30s
  master_key: "sk-test"

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401]

x-model-templates:
  flash: &flash
    name: gemini-2.5-flash
    rpm: 100
    tpm: 50000

credentials:
  - name: "vertex_v1"
    type: "vertex-ai"
    project_id: "proj-1"
    location: "us-central1"
    credentials_json: '{"type":"service_account"}'
    rpm: 100
    models:
      - *flash
      - name: gemini-2.5-pro
        rpm: 50
        tpm: 100000

monitoring:
  prometheus_enabled: false
`
	err := os.WriteFile(configPath, []byte(configContent), 0644)
	require.NoError(t, err)

	cfg, err := Load(configPath)
	require.NoError(t, err)

	require.Len(t, cfg.Models, 2)
	assert.Equal(t, "gemini-2.5-flash", cfg.Models[0].Name)
	assert.Equal(t, 100, cfg.Models[0].RPM)
	assert.Equal(t, "vertex_v1", cfg.Models[0].Credential)
	assert.Equal(t, "gemini-2.5-pro", cfg.Models[1].Name)
	assert.Equal(t, "vertex_v1", cfg.Models[1].Credential)
}

func TestResolveEnvString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		envKey   string
		envValue string
		setEnv   bool
		expected string
	}{
		{
			name:     "resolves set env var",
			input:    "os.environ/TEST_RESOLVE_VAR",
			envKey:   "TEST_RESOLVE_VAR",
			envValue: "resolved-value",
			setEnv:   true,
			expected: "resolved-value",
		},
		{
			name:     "unset env var returns empty string",
			input:    "os.environ/TEST_RESOLVE_UNSET_VAR",
			envKey:   "TEST_RESOLVE_UNSET_VAR",
			envValue: "",
			setEnv:   false,
			expected: "",
		},
		{
			name:     "non-env-prefixed value passes through",
			input:    "plain-string-value",
			envKey:   "",
			envValue: "",
			setEnv:   false,
			expected: "plain-string-value",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setEnv {
				require.NoError(t, os.Setenv(tt.envKey, tt.envValue))
				defer func() { _ = os.Unsetenv(tt.envKey) }()
			} else if tt.envKey != "" {
				// Make sure it is unset
				_ = os.Unsetenv(tt.envKey)
			}

			result := resolveEnvString(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConfig_Validate_InvalidWeight(t *testing.T) {
	tests := []struct {
		name        string
		credWeight  int
		modelWeight int
		wantErr     bool
	}{
		{"default weights", 0, 0, false},
		{"positive credential weight", 100, 0, false},
		{"positive model weight", 1, 200, false},
		{"negative credential weight", -1, 0, true},
		{"negative model weight", 1, -5, true},
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
					{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10, Weight: tt.credWeight},
				},
				Models: []ModelRPMConfig{
					{Name: "gpt-4o", Credential: "test", Weight: tt.modelWeight},
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

func TestCredentialConfig_UnmarshalYAML_Weight(t *testing.T) {
	t.Setenv("TEST_MODEL_WEIGHT", "300")

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configContent := `
server:
  port: 8080
  max_body_size_mb: 10
  master_key: "test-key"
  request_timeout: 30s

credentials:
  - name: "ours"
    type: "openai"
    api_key: "key"
    base_url: "http://ours.com"
    rpm: 100
    weight: 100
    models:
      - name: "gpt-4o"
        weight: 200
      - name: "gpt-4o-mini"
        weight: os.environ/TEST_MODEL_WEIGHT
  - name: "azure"
    type: "openai"
    api_key: "key2"
    base_url: "http://azure.com"
    rpm: 100

monitoring:
  prometheus_enabled: false
`
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	cfg, err := Load(configPath)
	require.NoError(t, err)

	require.Len(t, cfg.Credentials, 2)
	assert.Equal(t, 100, cfg.Credentials[0].Weight, "credential weight parsed")
	assert.Equal(t, 0, cfg.Credentials[1].Weight, "omitted credential weight defaults to 0 (=1)")

	var found bool
	var foundEnvWeight bool
	for _, m := range cfg.Models {
		if m.Name == "gpt-4o" && m.Credential == "ours" {
			assert.Equal(t, 200, m.Weight, "per-model weight parsed and unpacked")
			found = true
		}
		if m.Name == "gpt-4o-mini" && m.Credential == "ours" {
			assert.Equal(t, 300, m.Weight, "per-model weight parsed from env and unpacked")
			foundEnvWeight = true
		}
	}
	assert.True(t, found, "model gpt-4o should be unpacked into cfg.Models")
	assert.True(t, foundEnvWeight, "model gpt-4o-mini should be unpacked into cfg.Models")
}
