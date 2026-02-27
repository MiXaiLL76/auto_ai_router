package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const DefaultMaxAttempts = 3
const DefaultBanDuration time.Duration = 0

var DefaultErrorCodes = []int{429}

// ProviderType represents the type of AI provider
type ProviderType string

const (
	ProviderTypeOpenAI    ProviderType = "openai"
	ProviderTypeVertexAI  ProviderType = "vertex-ai"
	ProviderTypeGemini    ProviderType = "gemini"
	ProviderTypeAnthropic ProviderType = "anthropic"
	ProviderTypeProxy     ProviderType = "proxy"
)

// IsValid checks if the provider type is valid
func (p ProviderType) IsValid() bool {
	switch p {
	case ProviderTypeOpenAI, ProviderTypeVertexAI, ProviderTypeGemini, ProviderTypeAnthropic, ProviderTypeProxy:
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
	Fail2Ban    Fail2BanConfig     `yaml:"fail2ban,omitempty"`
	Credentials []CredentialConfig `yaml:"credentials"`
	Monitoring  MonitoringConfig   `yaml:"monitoring"`
	Models      []ModelRPMConfig   `yaml:"models,omitempty"`
	ModelAlias  map[string]string  `yaml:"model_alias,omitempty"`
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
	ReadTimeout            time.Duration `yaml:"-"`                           // HTTP server read timeout (equals request_timeout, not configurable via YAML)
	WriteTimeout           time.Duration `yaml:"write_timeout"`               // HTTP server write timeout (default: 60s)
	IdleTimeout            time.Duration `yaml:"idle_timeout"`                // HTTP server idle timeout (default: 2*write_timeout)
	MaxProviderRetries     int           `yaml:"max_provider_retries"`        // Max same-type credential retries on provider errors (default: 2, meaning 3 total attempts)
	ModelPricesLink        string        `yaml:"model_prices_link,omitempty"` // URL or file path to model prices JSON - supports os.environ/VAR_NAME
}

// ErrorCodeRuleConfig defines per-error-code ban rules
type ErrorCodeRuleConfig struct {
	Code        int    `yaml:"code,omitempty"`
	MaxAttempts int    `yaml:"max_attempts,omitempty"`
	BanDuration string `yaml:"ban_duration,omitempty"`
}

type Fail2BanConfig struct {
	MaxAttempts    int                   `yaml:"max_attempts,omitempty"`
	BanDuration    time.Duration         `yaml:"ban_duration,omitempty"`
	ErrorCodes     []int                 `yaml:"error_codes,omitempty"`
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
		WriteTimeout           string `yaml:"write_timeout"`
		IdleTimeout            string `yaml:"idle_timeout"`
		MaxProviderRetries     string `yaml:"max_provider_retries"`
		ModelPricesLink        string `yaml:"model_prices_link,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Resolve and parse each field
	var err error

	// Integer fields
	if s.Port, err = parseField(temp.Port, 8080, strconv.Atoi, "port"); err != nil {
		return err
	}
	if s.MaxBodySizeMB, err = parseField(temp.MaxBodySizeMB, 100, strconv.Atoi, "max_body_size_mb"); err != nil {
		return err
	}
	if s.ResponseBodyMultiplier, err = parseField(temp.ResponseBodyMultiplier, 10, strconv.Atoi, "response_body_multiplier"); err != nil {
		return err
	}
	if s.DefaultModelsRPM, err = parseField(temp.DefaultModelsRPM, -1, strconv.Atoi, "default_models_rpm"); err != nil {
		return err
	}
	if s.MaxIdleConns, err = parseField(temp.MaxIdleConns, 200, strconv.Atoi, "max_idle_conns"); err != nil {
		return err
	}
	if s.MaxIdleConnsPerHost, err = parseField(temp.MaxIdleConnsPerHost, 20, strconv.Atoi, "max_idle_conns_per_host"); err != nil {
		return err
	}

	// Duration fields
	if s.RequestTimeout, err = parseField(temp.RequestTimeout, 60*time.Second, time.ParseDuration, "request_timeout"); err != nil {
		return err
	}
	if s.IdleConnTimeout, err = parseField(temp.IdleConnTimeout, 120*time.Second, time.ParseDuration, "idle_conn_timeout"); err != nil {
		return err
	}
	if s.WriteTimeout, err = parseField(temp.WriteTimeout, 60*time.Second, time.ParseDuration, "write_timeout"); err != nil {
		return err
	}
	if s.IdleTimeout, err = parseField(temp.IdleTimeout, 2*time.Minute, time.ParseDuration, "idle_timeout"); err != nil {
		return err
	}

	// Max provider retries (default: 2 = 3 total attempts)
	if s.MaxProviderRetries, err = parseField(temp.MaxProviderRetries, 2, strconv.Atoi, "max_provider_retries"); err != nil {
		return err
	}

	// String fields
	s.LoggingLevel = resolveEnvString(temp.LoggingLevel)
	s.MasterKey = resolveEnvString(temp.MasterKey)
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

	// Resolve and parse integer fields
	var err error
	if c.RPM, err = parseField(temp.RPM, -1, strconv.Atoi, "rpm for credential '"+c.Name+"'"); err != nil {
		return err
	}
	if c.TPM, err = parseField(temp.TPM, -1, strconv.Atoi, "tpm for credential '"+c.Name+"'"); err != nil {
		return err
	}

	// Resolve and parse boolean field
	if c.IsFallback, err = parseField(temp.IsFallback, false, strconv.ParseBool, "is_fallback for credential '"+c.Name+"'"); err != nil {
		return err
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
	HealthCheckPath   string `yaml:"-"` // Fixed to "/health", not configurable via YAML
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
	AuthCacheTTL  time.Duration `yaml:"auth_cache_ttl"`  // default: 5s
	AuthCacheSize int           `yaml:"auth_cache_size"` // default: 10000

	// Spend logging
	LogQueueSize     int           `yaml:"log_queue_size"`     // default: 10000
	LogBatchSize     int           `yaml:"log_batch_size"`     // default: 100
	LogFlushInterval time.Duration `yaml:"log_flush_interval"` // default: 5s
}

// UnmarshalYAML implements custom unmarshaling for MonitoringConfig with env variable support
func (m *MonitoringConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with all string fields
	type tempConfig struct {
		PrometheusEnabled string `yaml:"prometheus_enabled"`
		LogErrors         string `yaml:"log_errors,omitempty"`
		ErrorsLogPath     string `yaml:"errors_log_path,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Resolve and parse boolean fields
	var err error
	if m.PrometheusEnabled, err = parseField(temp.PrometheusEnabled, false, strconv.ParseBool, "prometheus_enabled"); err != nil {
		return err
	}
	if m.LogErrors, err = parseField(temp.LogErrors, false, strconv.ParseBool, "log_errors"); err != nil {
		return err
	}

	// Resolve string fields
	m.HealthCheckPath = "/health" // Fixed path, not configurable via YAML
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
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	var err error

	l.DatabaseURL = resolveEnvString(temp.DatabaseURL)

	// Boolean fields
	if l.Enabled, err = parseField(temp.Enabled, false, strconv.ParseBool, "litellm_db.enabled"); err != nil {
		return err
	}
	if l.IsRequired, err = parseField(temp.IsRequired, false, strconv.ParseBool, "litellm_db.is_required"); err != nil {
		return err
	}

	// Integer fields (defaults optimized for ~1000 requests/minute)
	if l.MaxConns, err = parseField(temp.MaxConns, 25, strconv.Atoi, "litellm_db.max_conns"); err != nil {
		return err
	}
	if l.MinConns, err = parseField(temp.MinConns, 5, strconv.Atoi, "litellm_db.min_conns"); err != nil {
		return err
	}
	if l.AuthCacheSize, err = parseField(temp.AuthCacheSize, 10000, strconv.Atoi, "litellm_db.auth_cache_size"); err != nil {
		return err
	}
	if l.LogQueueSize, err = parseField(temp.LogQueueSize, 5000, strconv.Atoi, "litellm_db.log_queue_size"); err != nil {
		return err
	}
	if l.LogBatchSize, err = parseField(temp.LogBatchSize, 100, strconv.Atoi, "litellm_db.log_batch_size"); err != nil {
		return err
	}
	// Duration fields
	if l.HealthCheckInterval, err = parseField(temp.HealthCheckInterval, 10*time.Second, time.ParseDuration, "litellm_db.health_check_interval"); err != nil {
		return err
	}
	if l.ConnectTimeout, err = parseField(temp.ConnectTimeout, 5*time.Second, time.ParseDuration, "litellm_db.connect_timeout"); err != nil {
		return err
	}
	if l.AuthCacheTTL, err = parseField(temp.AuthCacheTTL, 5*time.Second, time.ParseDuration, "litellm_db.auth_cache_ttl"); err != nil {
		return err
	}
	if l.LogFlushInterval, err = parseField(temp.LogFlushInterval, 5*time.Second, time.ParseDuration, "litellm_db.log_flush_interval"); err != nil {
		return err
	}

	return nil
}

// UnmarshalYAML implements custom unmarshaling for Fail2BanConfig
func (f *Fail2BanConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with string ban_duration
	type tempConfig struct {
		MaxAttempts    string                `yaml:"max_attempts,omitempty"`
		BanDuration    string                `yaml:"ban_duration,omitempty"`
		ErrorCodes     []int                 `yaml:"error_codes,omitempty"`
		ErrorCodeRules []ErrorCodeRuleConfig `yaml:"error_code_rules,omitempty"`
	}
	var err error
	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	if f.MaxAttempts, err = parseField(temp.MaxAttempts, DefaultMaxAttempts, strconv.Atoi, "fail2ban.max_attempts"); err != nil {
		return err
	}

	if temp.BanDuration == "permanent" {
		f.BanDuration = DefaultBanDuration
	} else {
		if f.BanDuration, err = parseField(temp.BanDuration, DefaultBanDuration, time.ParseDuration, "ban_duration"); err != nil {
			return err
		}
	}
	if len(temp.ErrorCodes) == 0 {
		f.ErrorCodes = DefaultErrorCodes
	} else {
		f.ErrorCodes = temp.ErrorCodes
	}

	f.ErrorCodeRules = temp.ErrorCodeRules

	return nil
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var root yaml.Node
	if err := yaml.Unmarshal(data, &root); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	if !hasMappingKey(&root, "fail2ban") {
		cfg.Fail2Ban = defaultFail2BanConfig()
	}

	// Resolve env variables in model_alias values
	if cfg.ModelAlias != nil {
		resolved := make(map[string]string, len(cfg.ModelAlias))
		for alias, target := range cfg.ModelAlias {
			resolved[resolveEnvString(alias)] = resolveEnvString(target)
		}
		cfg.ModelAlias = resolved
	}

	// Normalize credentials
	cfg.Normalize()

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &cfg, nil
}

func defaultFail2BanConfig() Fail2BanConfig {
	return Fail2BanConfig{
		MaxAttempts: DefaultMaxAttempts,
		BanDuration: DefaultBanDuration,
		ErrorCodes:  append([]int(nil), DefaultErrorCodes...),
	}
}

func hasMappingKey(node *yaml.Node, key string) bool {
	if node == nil {
		return false
	}
	if node.Kind == yaml.DocumentNode && len(node.Content) > 0 {
		node = node.Content[0]
	}
	if node.Kind != yaml.MappingNode {
		return false
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return true
		}
	}
	return false
}

// Normalize cleans up configuration values
func (c *Config) Normalize() {
	// Remove /v1 suffix from base_url to avoid duplication
	for i := range c.Credentials {
		c.Credentials[i].BaseURL = strings.TrimSuffix(c.Credentials[i].BaseURL, "/v1")
	}
}

func (c *Config) Validate() error {
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

	// Validate and normalize default_models_rpm
	// -1 means unlimited RPM, 0 is treated as unlimited
	if c.Server.DefaultModelsRPM == 0 {
		c.Server.DefaultModelsRPM = -1 // Convert 0 to unlimited (-1)
	} else if c.Server.DefaultModelsRPM < -1 {
		return fmt.Errorf("invalid default_models_rpm: %d (must be -1 for unlimited or positive number)", c.Server.DefaultModelsRPM)
	}

	// ReadTimeout equals RequestTimeout (not configurable via YAML)
	c.Server.ReadTimeout = c.Server.RequestTimeout

	// Validate MaxProviderRetries
	if c.Server.MaxProviderRetries < 0 {
		return fmt.Errorf("invalid max_provider_retries: %d (must be >= 0)", c.Server.MaxProviderRetries)
	}

	// Validate IdleTimeout against WriteTimeout
	if c.Server.IdleTimeout == 0 {
		c.Server.IdleTimeout = c.Server.WriteTimeout * 2
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
			return fmt.Errorf("credential %s: invalid type: %s (must be 'openai', 'vertex-ai', 'gemini', 'anthropic', or 'proxy')", cred.Name, cred.Type)
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

		case ProviderTypeGemini:
			// For Gemini (Google AI Studio), api_key and base_url are required
			if cred.APIKey == "" {
				return fmt.Errorf("credential %s: api_key is required for gemini type", cred.Name)
			}
			if cred.BaseURL == "" {
				return fmt.Errorf("credential %s: base_url is required for gemini type", cred.Name)
			}
			if err := validateBaseURL(cred.Name, cred.BaseURL); err != nil {
				return err
			}

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
		if !strings.HasPrefix(c.LiteLLMDB.DatabaseURL, "postgres://") && !strings.HasPrefix(c.LiteLLMDB.DatabaseURL, "postgresql://") {
			return fmt.Errorf("litellm_db.database_url must start with postgres:// or postgresql://, got: %s", c.LiteLLMDB.DatabaseURL)
		}
	}

	return nil
}
