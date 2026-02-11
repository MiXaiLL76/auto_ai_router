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
	ProviderTypeProxy     ProviderType = "proxy"
)

// IsValid checks if the provider type is valid
func (p ProviderType) IsValid() bool {
	switch p {
	case ProviderTypeOpenAI, ProviderTypeVertexAI, ProviderTypeAnthropic, ProviderTypeProxy:
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
	LiteLLMDB   LiteLLMDBConfig    `yaml:"litellm_db,omitempty"`
}

type ServerConfig struct {
	Port                   int           `yaml:"port"`
	MaxBodySizeMB          int           `yaml:"max_body_size_mb"`
	ResponseBodyMultiplier int           `yaml:"response_body_multiplier"` // Multiplier for response body size limit relative to max_body_size_mb (default: 10)
	RequestTimeout         time.Duration `yaml:"request_timeout"`
	LoggingLevel           string        `yaml:"logging_level"`
	MasterKey              string        `yaml:"master_key"`
	DefaultModelsRPM       int           `yaml:"default_models_rpm"`
	MaxIdleConns           int           `yaml:"max_idle_conns"`
	MaxIdleConnsPerHost    int           `yaml:"max_idle_conns_per_host"`
	IdleConnTimeout        time.Duration `yaml:"idle_conn_timeout"`
	ReadTimeout            time.Duration `yaml:"read_timeout"`                // HTTP server read timeout (default: 60s)
	WriteTimeout           time.Duration `yaml:"write_timeout"`               // HTTP server write timeout (default: 10m or 1.5*request_timeout if set)
	IdleTimeout            time.Duration `yaml:"idle_timeout"`                // HTTP server idle timeout (default: 20m or 2*write_timeout)
	ModelPricesLink        string        `yaml:"model_prices_link,omitempty"` // URL or file path to model prices JSON - supports os.environ/VAR_NAME
}

// ErrorCodeRuleConfig defines per-error-code ban rules
type ErrorCodeRuleConfig struct {
	Code        int    `yaml:"code"`
	MaxAttempts int    `yaml:"max_attempts"`
	BanDuration string `yaml:"ban_duration"`
}

type Fail2BanConfig struct {
	MaxAttempts    int                   `yaml:"max_attempts"`
	BanDuration    time.Duration         `yaml:"ban_duration"`
	ErrorCodes     []int                 `yaml:"error_codes"`
	ErrorCodeRules []ErrorCodeRuleConfig `yaml:"error_code_rules,omitempty"`
}

// UnmarshalYAML implements custom unmarshaling for ServerConfig with env variable support
func (s *ServerConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with all string fields
	type tempConfig struct {
		Port                   string `yaml:"port"`
		MaxBodySizeMB          string `yaml:"max_body_size_mb"`
		ResponseBodyMultiplier string `yaml:"response_body_multiplier"`
		RequestTimeout         string `yaml:"request_timeout"`
		LoggingLevel           string `yaml:"logging_level"`
		MasterKey              string `yaml:"master_key"`
		DefaultModelsRPM       string `yaml:"default_models_rpm"`
		MaxIdleConns           string `yaml:"max_idle_conns"`
		MaxIdleConnsPerHost    string `yaml:"max_idle_conns_per_host"`
		IdleConnTimeout        string `yaml:"idle_conn_timeout"`
		ReadTimeout            string `yaml:"read_timeout"`
		WriteTimeout           string `yaml:"write_timeout"`
		IdleTimeout            string `yaml:"idle_timeout"`
		ModelPricesLink        string `yaml:"model_prices_link,omitempty"`
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

	// ResponseBodyMultiplier
	if temp.ResponseBodyMultiplier != "" {
		s.ResponseBodyMultiplier, err = resolveEnvInt(temp.ResponseBodyMultiplier, 10)
		if err != nil {
			return fmt.Errorf("invalid response_body_multiplier: %w", err)
		}
	} else {
		s.ResponseBodyMultiplier = 10 // Default value
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

	// MaxIdleConns
	if temp.MaxIdleConns != "" {
		s.MaxIdleConns, err = resolveEnvInt(temp.MaxIdleConns, 200)
		if err != nil {
			return fmt.Errorf("invalid max_idle_conns: %w", err)
		}
	} else {
		s.MaxIdleConns = 200 // Default value
	}

	// MaxIdleConnsPerHost
	if temp.MaxIdleConnsPerHost != "" {
		s.MaxIdleConnsPerHost, err = resolveEnvInt(temp.MaxIdleConnsPerHost, 20)
		if err != nil {
			return fmt.Errorf("invalid max_idle_conns_per_host: %w", err)
		}
	} else {
		s.MaxIdleConnsPerHost = 20 // Default value
	}

	// IdleConnTimeout
	if temp.IdleConnTimeout != "" {
		s.IdleConnTimeout, err = resolveEnvDuration(temp.IdleConnTimeout, 120*time.Second)
		if err != nil {
			return fmt.Errorf("invalid idle_conn_timeout: %w", err)
		}
	} else {
		s.IdleConnTimeout = 120 * time.Second // Default value
	}

	// ReadTimeout
	if temp.ReadTimeout != "" {
		s.ReadTimeout, err = resolveEnvDuration(temp.ReadTimeout, 60*time.Second)
		if err != nil {
			return fmt.Errorf("invalid read_timeout: %w", err)
		}
	}

	// WriteTimeout
	if temp.WriteTimeout != "" {
		s.WriteTimeout, err = resolveEnvDuration(temp.WriteTimeout, 10*time.Minute)
		if err != nil {
			return fmt.Errorf("invalid write_timeout: %w", err)
		}
	}

	// IdleTimeout
	if temp.IdleTimeout != "" {
		s.IdleTimeout, err = resolveEnvDuration(temp.IdleTimeout, 20*time.Minute)
		if err != nil {
			return fmt.Errorf("invalid idle_timeout: %w", err)
		}
	}

	s.ModelPricesLink = resolveEnvString(temp.ModelPricesLink)

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

	// Proxy specific fields
	IsFallback bool `yaml:"is_fallback,omitempty"`
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
		IsFallback      string `yaml:"is_fallback,omitempty"`
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

	// Resolve and parse IsFallback
	if temp.IsFallback != "" {
		c.IsFallback, err = resolveEnvBool(temp.IsFallback, false)
		if err != nil {
			return fmt.Errorf("invalid is_fallback for credential '%s': %w", c.Name, err)
		}
	}

	// Validate base_url for proxy and other provider types that require it
	if c.BaseURL != "" {
		if err := validateBaseURL(c.Name, c.BaseURL); err != nil {
			return err
		}
	}

	return nil
}

type MonitoringConfig struct {
	PrometheusEnabled bool   `yaml:"prometheus_enabled"`
	HealthCheckPath   string `yaml:"health_check_path"`
	LogErrors         bool   `yaml:"log_errors,omitempty"`
	ErrorsLogPath     string `yaml:"errors_log_path,omitempty"`
}

// LiteLLMDBConfig holds configuration for LiteLLM database integration
type LiteLLMDBConfig struct {
	// Enable/disable module
	Enabled bool `yaml:"enabled"`

	// IsRequired specifies whether LiteLLM DB is mandatory (fail startup on error)
	// or optional (degrade to NoopManager with warning on error)
	IsRequired bool `yaml:"is_required"` // default: false

	// Database connection postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
	DatabaseURL string `yaml:"database_url"` // os.environ/LITELLM_DATABASE_URL
	MaxConns    int    `yaml:"max_conns"`    // default: 10
	MinConns    int    `yaml:"min_conns"`    // default: 2

	// Health check
	HealthCheckInterval time.Duration `yaml:"health_check_interval"` // default: 10s
	ConnectTimeout      time.Duration `yaml:"connect_timeout"`       // default: 5s

	// Auth cache
	AuthCacheTTL  time.Duration `yaml:"auth_cache_ttl"`  // default: 20s
	AuthCacheSize int           `yaml:"auth_cache_size"` // default: 10000

	// Spend logging
	LogQueueSize     int           `yaml:"log_queue_size"`     // default: 10000
	LogBatchSize     int           `yaml:"log_batch_size"`     // default: 100
	LogFlushInterval time.Duration `yaml:"log_flush_interval"` // default: 5s
	LogRetryAttempts int           `yaml:"log_retry_attempts"` // default: 3
	LogRetryDelay    time.Duration `yaml:"log_retry_delay"`    // default: 1s
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

// UnmarshalYAML implements custom unmarshaling for LiteLLMDBConfig with env variable support
func (l *LiteLLMDBConfig) UnmarshalYAML(value *yaml.Node) error {
	type tempConfig struct {
		Enabled             string `yaml:"enabled"`
		IsRequired          string `yaml:"is_required"`
		DatabaseURL         string `yaml:"database_url"`
		MaxConns            string `yaml:"max_conns"`
		MinConns            string `yaml:"min_conns"`
		HealthCheckInterval string `yaml:"health_check_interval"`
		ConnectTimeout      string `yaml:"connect_timeout"`
		AuthCacheTTL        string `yaml:"auth_cache_ttl"`
		AuthCacheSize       string `yaml:"auth_cache_size"`
		LogQueueSize        string `yaml:"log_queue_size"`
		LogBatchSize        string `yaml:"log_batch_size"`
		LogFlushInterval    string `yaml:"log_flush_interval"`
		LogRetryAttempts    string `yaml:"log_retry_attempts"`
		LogRetryDelay       string `yaml:"log_retry_delay"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Helper function to parse integer fields
	parseIntField := func(value string, defaultVal int, fieldName string) (int, error) {
		if value == "" {
			return defaultVal, nil
		}
		v, err := resolveEnvInt(value, defaultVal)
		if err != nil {
			return 0, fmt.Errorf("invalid litellm_db.%s: %w", fieldName, err)
		}
		return v, nil
	}

	// Helper function to parse duration fields
	parseDurationField := func(value string, defaultVal time.Duration, fieldName string) (time.Duration, error) {
		if value == "" {
			return defaultVal, nil
		}
		v, err := resolveEnvDuration(value, defaultVal)
		if err != nil {
			return 0, fmt.Errorf("invalid litellm_db.%s: %w", fieldName, err)
		}
		return v, nil
	}

	var err error

	l.DatabaseURL = resolveEnvString(temp.DatabaseURL)

	if temp.Enabled != "" {
		l.Enabled, err = resolveEnvBool(temp.Enabled, false)
		if err != nil {
			return fmt.Errorf("invalid litellm_db.enabled: %w", err)
		}
	}

	if temp.IsRequired != "" {
		l.IsRequired, err = resolveEnvBool(temp.IsRequired, false)
		if err != nil {
			return fmt.Errorf("invalid litellm_db.is_required: %w", err)
		}
	}

	l.MaxConns, err = parseIntField(temp.MaxConns, 10, "max_conns")
	if err != nil {
		return err
	}

	l.MinConns, err = parseIntField(temp.MinConns, 2, "min_conns")
	if err != nil {
		return err
	}

	l.AuthCacheSize, err = parseIntField(temp.AuthCacheSize, 10000, "auth_cache_size")
	if err != nil {
		return err
	}

	l.LogQueueSize, err = parseIntField(temp.LogQueueSize, 10000, "log_queue_size")
	if err != nil {
		return err
	}

	l.LogBatchSize, err = parseIntField(temp.LogBatchSize, 100, "log_batch_size")
	if err != nil {
		return err
	}

	l.LogRetryAttempts, err = parseIntField(temp.LogRetryAttempts, 3, "log_retry_attempts")
	if err != nil {
		return err
	}

	l.HealthCheckInterval, err = parseDurationField(temp.HealthCheckInterval, 10*time.Second, "health_check_interval")
	if err != nil {
		return err
	}

	l.ConnectTimeout, err = parseDurationField(temp.ConnectTimeout, 5*time.Second, "connect_timeout")
	if err != nil {
		return err
	}

	l.AuthCacheTTL, err = parseDurationField(temp.AuthCacheTTL, 20*time.Second, "auth_cache_ttl")
	if err != nil {
		return err
	}

	l.LogFlushInterval, err = parseDurationField(temp.LogFlushInterval, 5*time.Second, "log_flush_interval")
	if err != nil {
		return err
	}

	l.LogRetryDelay, err = parseDurationField(temp.LogRetryDelay, time.Second, "log_retry_delay")
	if err != nil {
		return err
	}

	return nil
}

// ApplyDefaults sets default values for all LiteLLMDBConfig fields
func (c *LiteLLMDBConfig) ApplyDefaults() {
	if c.MaxConns == 0 {
		c.MaxConns = 10
	}
	if c.MinConns == 0 {
		c.MinConns = 2
	}
	if c.HealthCheckInterval == 0 {
		c.HealthCheckInterval = 10 * time.Second
	}
	if c.ConnectTimeout == 0 {
		c.ConnectTimeout = 5 * time.Second
	}
	if c.AuthCacheTTL == 0 {
		c.AuthCacheTTL = 20 * time.Second
	}
	if c.AuthCacheSize == 0 {
		c.AuthCacheSize = 10000
	}
	if c.LogQueueSize == 0 {
		c.LogQueueSize = 10000
	}
	if c.LogBatchSize == 0 {
		c.LogBatchSize = 100
	}
	if c.LogFlushInterval == 0 {
		c.LogFlushInterval = 5 * time.Second
	}
	if c.LogRetryAttempts == 0 {
		c.LogRetryAttempts = 3
	}
	if c.LogRetryDelay == 0 {
		c.LogRetryDelay = time.Second
	}
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

// parseFunc is a function type that parses a string value into the desired type
type parseFunc[T any] func(string) (T, error)

// resolveEnvValue resolves environment variable and parses it using the provided parser
func resolveEnvValue[T any](value string, defaultValue T, parser parseFunc[T], typeName string) (T, error) {
	if value == "" {
		return defaultValue, nil
	}

	resolved := resolveEnvString(value)

	parsed, err := parser(resolved)
	if err != nil {
		return defaultValue, fmt.Errorf("failed to parse %s from '%s': %w", typeName, resolved, err)
	}
	return parsed, nil
}

// resolveEnvInt resolves environment variable and converts to int
func resolveEnvInt(value string, defaultValue int) (int, error) {
	return resolveEnvValue(value, defaultValue, strconv.Atoi, "int")
}

// resolveEnvBool resolves environment variable and converts to bool
func resolveEnvBool(value string, defaultValue bool) (bool, error) {
	return resolveEnvValue(value, defaultValue, strconv.ParseBool, "bool")
}

// resolveEnvDuration resolves environment variable and converts to duration
func resolveEnvDuration(value string, defaultValue time.Duration) (time.Duration, error) {
	return resolveEnvValue(value, defaultValue, time.ParseDuration, "duration")
}

// validateBaseURL validates that a URL is properly formed with http/https scheme
func validateBaseURL(credentialName, baseURL string) error {
	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("credential %s: invalid base_url: %w", credentialName, err)
	}
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return fmt.Errorf("credential %s: base_url must use http or https scheme, got: %s", credentialName, parsedURL.Scheme)
	}
	if parsedURL.Host == "" {
		return fmt.Errorf("credential %s: base_url must have a host", credentialName)
	}
	return nil
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
		c.Credentials[i].BaseURL = strings.TrimSuffix(c.Credentials[i].BaseURL, "/v1")
	}
}

func (c *Config) Validate() error {
	// Apply defaults for LiteLLMDB config before validation
	c.LiteLLMDB.ApplyDefaults()

	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Server.Port)
	}

	if c.Server.MaxBodySizeMB <= 0 {
		return fmt.Errorf("invalid max_body_size_mb: %d", c.Server.MaxBodySizeMB)
	}

	if c.Server.ResponseBodyMultiplier <= 0 {
		c.Server.ResponseBodyMultiplier = 10
	}

	// -1 means unlimited timeout
	if c.Server.RequestTimeout < 0 && c.Server.RequestTimeout != -1 {
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

	// Set default for read_timeout if not specified
	if c.Server.ReadTimeout == 0 {
		c.Server.ReadTimeout = 60 * time.Second
	}

	// Set default for write_timeout if not specified
	if c.Server.WriteTimeout == 0 {
		if c.Server.RequestTimeout > 0 {
			c.Server.WriteTimeout = time.Duration(float64(c.Server.RequestTimeout) * 1.5)
		} else {
			c.Server.WriteTimeout = 10 * time.Minute
		}
	}

	// Set default for idle_timeout if not specified
	if c.Server.IdleTimeout == 0 {
		c.Server.IdleTimeout = c.Server.WriteTimeout * 2
	}

	// Set default for health_check_path if not specified
	if c.Monitoring.HealthCheckPath == "" {
		c.Monitoring.HealthCheckPath = "/health"
	}

	if c.Fail2Ban.MaxAttempts <= 0 {
		return fmt.Errorf("invalid max_attempts: %d", c.Fail2Ban.MaxAttempts)
	}

	if len(c.Credentials) == 0 {
		return fmt.Errorf("no credentials configured")
	}

	// Validate Fail2Ban error code rules for duplicates
	seenErrorCodes := make(map[int]bool)
	for _, rule := range c.Fail2Ban.ErrorCodeRules {
		if seenErrorCodes[rule.Code] {
			return fmt.Errorf("fail2ban: duplicate error_code_rules for code %d", rule.Code)
		}
		seenErrorCodes[rule.Code] = true
	}

	for i, cred := range c.Credentials {
		if cred.Name == "" {
			return fmt.Errorf("credential %d: name is required", i)
		}

		// Validate provider type
		if !cred.Type.IsValid() {
			return fmt.Errorf("credential %s: invalid type: %s (must be 'openai', 'vertex-ai', 'anthropic', or 'proxy')", cred.Name, cred.Type)
		}

		// Validate by provider type
		switch cred.Type {
		case ProviderTypeProxy:
			// base_url is required for proxy
			if cred.BaseURL == "" {
				return fmt.Errorf("credential %s: base_url is required for proxy type", cred.Name)
			}
			// Validate base_url is a valid URL
			if err := validateBaseURL(cred.Name, cred.BaseURL); err != nil {
				return err
			}
			// api_key is optional for proxy

		case ProviderTypeVertexAI:
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
			// Validate credentials_file exists if provided
			if cred.CredentialsFile != "" {
				if _, err := os.Stat(cred.CredentialsFile); err != nil {
					return fmt.Errorf("credential %s: credentials_file does not exist or is not accessible: %w", cred.Name, err)
				}
			}
			// base_url is optional for Vertex AI (will be constructed dynamically)

		default:
			// For OpenAI and Anthropic, require APIKey and BaseURL
			if cred.APIKey == "" {
				return fmt.Errorf("credential %s: api_key is required", cred.Name)
			}
			if cred.BaseURL == "" {
				return fmt.Errorf("credential %s: base_url is required", cred.Name)
			}
			// Validate base_url is a valid URL
			if err := validateBaseURL(cred.Name, cred.BaseURL); err != nil {
				return err
			}
		}

		// -1 means unlimited RPM
		if cred.RPM <= 0 && !isUnlimited(cred.RPM) {
			return fmt.Errorf("credential %s: invalid rpm: %d (must be -1 for unlimited or positive number)", cred.Name, cred.RPM)
		}
		// TPM: 0 or -1 means unlimited, positive means limited
		if cred.TPM < -1 {
			return fmt.Errorf("credential %s: invalid tpm: %d (must be -1 or 0 for unlimited, or positive number)", cred.Name, cred.TPM)
		}
	}

	// Validate LiteLLM DB config
	if c.LiteLLMDB.Enabled {
		if c.LiteLLMDB.DatabaseURL == "" {
			return fmt.Errorf("litellm_db.database_url is required when enabled")
		}
	}

	return nil
}

// isUnlimited checks if a value represents unlimited (-1)
func isUnlimited(value int) bool {
	return value == -1
}
