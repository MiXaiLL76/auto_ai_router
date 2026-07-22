package config

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/scope"
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
	ProviderTypeCometAPI  ProviderType = "cometapi"
	ProviderTypeBedrock   ProviderType = "bedrock"
	ProviderTypeProxy     ProviderType = "proxy"
)

// LogValue implements slog.LogValuer so structured log backends (e.g. the
// OTEL bridge) serialize ProviderType as a plain string instead of an
// unhandled custom type.
func (p ProviderType) LogValue() slog.Value {
	return slog.StringValue(string(p))
}

// IsValid checks if the provider type is valid
func (p ProviderType) IsValid() bool {
	switch p {
	case ProviderTypeOpenAI, ProviderTypeVertexAI, ProviderTypeGemini, ProviderTypeAnthropic, ProviderTypeCometAPI, ProviderTypeBedrock, ProviderTypeProxy:
		return true
	}
	return false
}

func normalizeProviderType(raw string) ProviderType {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "comet-api", "comet_api":
		return ProviderTypeCometAPI
	default:
		return ProviderType(strings.ToLower(strings.TrimSpace(raw)))
	}
}

// ModelRPMConfig represents RPM and TPM limits for a specific model
type ModelRPMConfig struct {
	Name  string `yaml:"name"`
	Model string `yaml:"model,omitempty"` // Real model name sent to provider (alias for Name if different)
	// DeploymentID is the authoritative LiteLLM_ProxyModelTable.model_id.
	// It is populated only by the database loader and is never accepted from YAML.
	DeploymentID string `yaml:"-"`
	RPM          int    `yaml:"rpm"`
	TPM          int    `yaml:"tpm"`
	Weight       int    `yaml:"weight"`               // Weighted round-robin weight (0 = use credential default / 1)
	Credential   string `yaml:"credential,omitempty"` // If set, model is only available for this credential

	// PassthroughResponses controls whether Responses API requests for this model
	// are forwarded as-is to the provider's native /v1/responses endpoint instead
	// of being converted to Chat Completions format.
	// nil (omitted in config) = auto-detect: true for codex models, false otherwise.
	// Explicit true/false overrides the auto-detection.
	PassthroughResponses *bool `yaml:"passthrough_responses,omitempty"`
}

// UnmarshalYAML implements custom unmarshaling for ModelRPMConfig with env variable support.
func (m *ModelRPMConfig) UnmarshalYAML(value *yaml.Node) error {
	type tempConfig struct {
		Name                 string `yaml:"name"`
		Model                string `yaml:"model,omitempty"`
		RPM                  string `yaml:"rpm"`
		TPM                  string `yaml:"tpm"`
		Weight               string `yaml:"weight"`
		Credential           string `yaml:"credential,omitempty"`
		PassthroughResponses string `yaml:"passthrough_responses,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	m.Name = resolveEnvString(temp.Name)
	m.Model = resolveEnvString(temp.Model)
	m.Credential = resolveEnvString(temp.Credential)
	m.PassthroughResponses = nil

	var err error
	if m.RPM, err = parseField(temp.RPM, 0, strconv.Atoi, "rpm for model '"+m.Name+"'"); err != nil {
		return err
	}
	if m.TPM, err = parseField(temp.TPM, 0, strconv.Atoi, "tpm for model '"+m.Name+"'"); err != nil {
		return err
	}
	if m.Weight, err = parseField(temp.Weight, 0, strconv.Atoi, "weight for model '"+m.Name+"'"); err != nil {
		return err
	}

	if temp.PassthroughResponses != "" {
		resolved := resolveEnvString(temp.PassthroughResponses)
		if resolved != "" {
			passthroughResponses, err := strconv.ParseBool(resolved)
			if err != nil {
				return fmt.Errorf("invalid passthrough_responses for model '%s': %w", m.Name, err)
			}
			m.PassthroughResponses = &passthroughResponses
		}
	}

	return nil
}

type Config struct {
	Server             ServerConfig       `yaml:"server"`
	Fail2Ban           Fail2BanConfig     `yaml:"fail2ban,omitempty"`
	Credentials        []CredentialConfig `yaml:"credentials"`
	Monitoring         MonitoringConfig   `yaml:"monitoring"`
	Models             []ModelRPMConfig   `yaml:"models,omitempty"`
	ModelAlias         map[string]string  `yaml:"model_alias,omitempty"`
	ClientModelIDs     []string           `yaml:"client_model_ids,omitempty"`
	PublicModelAlias   map[string]string  `yaml:"public_model_alias,omitempty"`
	AcceptedModelAlias map[string]string  `yaml:"accepted_model_alias,omitempty"`
	LiteLLMDB          LiteLLMDBConfig    `yaml:"litellm_db,omitempty"`
	SpendLog           SpendLogConfig     `yaml:"spend_log,omitempty"`
	Redis              RedisConfig        `yaml:"redis,omitempty"`
	OTEL               OTELConfig         `yaml:"otel,omitempty"`
	Kafka              KafkaConfig        `yaml:"kafka,omitempty"`
	// ModelTemplates stores x-model-templates entries as raw interface{} so that
	// both single-model mappings and lists of models can be defined as YAML anchors
	// without type errors. The actual model data is extracted via anchor expansion.
	ModelTemplates map[string]interface{} `yaml:"x-model-templates,omitempty"`
}

// UnmarshalYAML implements custom unmarshaling for Config with YAML anchor/alias support.
// This allows using YAML anchors (&) and aliases (*) for x-model-templates.
func (c *Config) UnmarshalYAML(value *yaml.Node) error {
	// First, resolve all aliases in the YAML document
	resolvedData, err := resolveYAMLAliases(value)
	if err != nil {
		return fmt.Errorf("failed to resolve YAML aliases: %w", err)
	}

	// Then unmarshal the resolved data into Config
	type RawConfig struct {
		Server             ServerConfig           `yaml:"server"`
		Fail2Ban           Fail2BanConfig         `yaml:"fail2ban,omitempty"`
		Credentials        []CredentialConfig     `yaml:"credentials"`
		Monitoring         MonitoringConfig       `yaml:"monitoring"`
		Models             []ModelRPMConfig       `yaml:"models,omitempty"`
		ModelAlias         map[string]string      `yaml:"model_alias,omitempty"`
		ClientModelIDs     []string               `yaml:"client_model_ids,omitempty"`
		PublicModelAlias   map[string]string      `yaml:"public_model_alias,omitempty"`
		AcceptedModelAlias map[string]string      `yaml:"accepted_model_alias,omitempty"`
		LiteLLMDB          LiteLLMDBConfig        `yaml:"litellm_db,omitempty"`
		SpendLog           SpendLogConfig         `yaml:"spend_log,omitempty"`
		Redis              RedisConfig            `yaml:"redis,omitempty"`
		OTEL               OTELConfig             `yaml:"otel,omitempty"`
		Kafka              KafkaConfig            `yaml:"kafka,omitempty"`
		ModelTemplates     map[string]interface{} `yaml:"x-model-templates,omitempty"`
	}

	var raw RawConfig
	if err := yaml.Unmarshal(resolvedData, &raw); err != nil {
		return fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Copy values to actual config
	c.Server = raw.Server
	c.Fail2Ban = raw.Fail2Ban
	c.Credentials = raw.Credentials
	c.Monitoring = raw.Monitoring
	c.Models = raw.Models
	c.ModelAlias = raw.ModelAlias
	c.ClientModelIDs = raw.ClientModelIDs
	c.PublicModelAlias = raw.PublicModelAlias
	c.AcceptedModelAlias = raw.AcceptedModelAlias
	c.LiteLLMDB = raw.LiteLLMDB
	c.SpendLog = raw.SpendLog
	c.Redis = raw.Redis
	c.OTEL = raw.OTEL
	c.Kafka = raw.Kafka
	c.ModelTemplates = raw.ModelTemplates

	return nil
}

// resolveYAMLAliases takes raw YAML data, parses it, resolves aliases, and returns the resolved YAML.
func resolveYAMLAliases(node *yaml.Node) ([]byte, error) {
	// Collect all anchors from the document
	anchors := make(map[string]*yaml.Node)
	collectAnchors(node, anchors)

	// Replace aliases with their anchor values
	resolveAliasesInNode(node, anchors)

	// Flatten nested sequences that result from list-anchor expansion.
	// e.g. "- *list-anchor" in a sequence becomes a nested sequence after alias
	// resolution; flattenSequences collapses those into the parent sequence so
	// that model lists behave as expected.
	flattenSequences(node)

	// Marshal back to YAML
	return yaml.Marshal(node)
}

// collectAnchors collects all anchor definitions from the YAML node tree
func collectAnchors(node *yaml.Node, anchors map[string]*yaml.Node) {
	if node == nil {
		return
	}

	// If this node has an anchor, store it
	if node.Anchor != "" {
		anchors[node.Anchor] = node
	}

	for _, child := range node.Content {
		collectAnchors(child, anchors)
	}
}

// flattenSequences collapses nested sequences that arise when a list anchor is
// used as an item inside another sequence:
//
//	models:
//	  - *my-model-list   # after alias resolution this becomes a nested sequence
//	  - name: other      # ordinary mapping item
//
// After flattening the nested sequence items are promoted to the parent level,
// so the result is a flat list of model mappings.
func flattenSequences(root *yaml.Node) {
	if root == nil {
		return
	}
	queue := make([]*yaml.Node, 0, 16)
	queue = append(queue, root)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Kind == yaml.SequenceNode {
			flat := make([]*yaml.Node, 0, len(node.Content))
			for _, child := range node.Content {
				if child.Kind == yaml.SequenceNode {
					flat = append(flat, child.Content...)
				} else {
					flat = append(flat, child)
				}
			}
			node.Content = flat
		}
		queue = append(queue, node.Content...)
	}
}

// resolveAliasesInNode replaces alias nodes with their anchor values
func resolveAliasesInNode(node *yaml.Node, anchors map[string]*yaml.Node) {
	if node == nil {
		return
	}

	if node.Kind == yaml.AliasNode {
		// Replace the alias with the anchor's value
		if anchor, ok := anchors[node.Value]; ok {
			// Copy all fields from anchor to this node
			node.Kind = anchor.Kind
			node.Style = anchor.Style
			node.Tag = anchor.Tag
			node.Value = anchor.Value
			node.Anchor = anchor.Anchor
			node.Content = anchor.Content
			node.HeadComment = anchor.HeadComment
			node.LineComment = anchor.LineComment
		}
		return
	}

	for _, child := range node.Content {
		resolveAliasesInNode(child, anchors)
	}
}

// RedisConfig holds configuration for Redis/Valkey-backed distributed rate limiting.
// When Enabled=false the rate limiter falls back to in-process counters.
type RedisConfig struct {
	Enabled bool `yaml:"enabled"`

	// InitAddress is the list of Redis/Valkey node addresses (host:port).
	// If a single address is given and it is not a cluster node, the client
	// falls back to standalone mode automatically.
	InitAddresses []string `yaml:"addresses"`

	Username string `yaml:"username,omitempty"`
	Password string `yaml:"password,omitempty"`

	// SelectDB is the Redis database index (default: 0).
	SelectDB int `yaml:"select_db,omitempty"`

	// KeyPrefix is prepended to every rate-limit key (default: "rl:").
	KeyPrefix string `yaml:"key_prefix,omitempty"`

	TLSEnabled bool `yaml:"tls_enabled,omitempty"`

	ConnectTimeout   time.Duration `yaml:"connect_timeout,omitempty"`    // default: 5s
	ConnWriteTimeout time.Duration `yaml:"conn_write_timeout,omitempty"` // default: 10s

	// ForceSingleClient disables cluster detection (useful for single-node / Valkey).
	ForceSingleClient bool `yaml:"force_single_client,omitempty"`

	// Pool settings
	MinIdleConns    int           `yaml:"min_idle_conns,omitempty"`    // default: 10
	MaxIdleConns    int           `yaml:"max_idle_conns,omitempty"`    // default: 100
	MaxConnLifetime time.Duration `yaml:"max_conn_lifetime,omitempty"` // default: 30m

	// KeyTTL is the TTL for rate limit keys in seconds (default: 120).
	KeyTTL int `yaml:"key_ttl,omitempty"`

	// CommandTimeout is the timeout for individual Redis commands (default: 3s).
	CommandTimeout time.Duration `yaml:"command_timeout,omitempty"`

	// Hybrid enables the hybrid backend: rate-limit decisions are made locally
	// (zero added latency) and Redis is updated asynchronously in batches.
	// Recommended when Redis latency is high or for single-instance deployments.
	Hybrid bool `yaml:"hybrid,omitempty"`

	// SyncInterval controls how often the hybrid backend pulls aggregated stats
	// from Redis to account for traffic from other instances (default: 5s).
	SyncInterval time.Duration `yaml:"sync_interval,omitempty"`
}

// UnmarshalYAML implements custom unmarshaling for RedisConfig with env variable support.
func (r *RedisConfig) UnmarshalYAML(value *yaml.Node) error {
	type tempConfig struct {
		Enabled           string   `yaml:"enabled"`
		InitAddresses     []string `yaml:"addresses"`
		Username          string   `yaml:"username,omitempty"`
		Password          string   `yaml:"password,omitempty"`
		SelectDB          string   `yaml:"select_db,omitempty"`
		KeyPrefix         string   `yaml:"key_prefix,omitempty"`
		TLSEnabled        string   `yaml:"tls_enabled,omitempty"`
		ConnectTimeout    string   `yaml:"connect_timeout,omitempty"`
		ConnWriteTimeout  string   `yaml:"conn_write_timeout,omitempty"`
		ForceSingleClient string   `yaml:"force_single_client,omitempty"`
		MinIdleConns      string   `yaml:"min_idle_conns,omitempty"`
		MaxIdleConns      string   `yaml:"max_idle_conns,omitempty"`
		MaxConnLifetime   string   `yaml:"max_conn_lifetime,omitempty"`
		KeyTTL            string   `yaml:"key_ttl,omitempty"`
		CommandTimeout    string   `yaml:"command_timeout,omitempty"`
		Hybrid            string   `yaml:"hybrid,omitempty"`
		SyncInterval      string   `yaml:"sync_interval,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	var err error

	if r.Enabled, err = parseField(temp.Enabled, false, strconv.ParseBool, "redis.enabled"); err != nil {
		return err
	}

	// Resolve env variables in each address
	r.InitAddresses = make([]string, 0, len(temp.InitAddresses))
	for _, addr := range temp.InitAddresses {
		r.InitAddresses = append(r.InitAddresses, resolveEnvString(addr))
	}

	r.Username = resolveEnvString(temp.Username)
	r.Password = resolveEnvString(temp.Password)
	r.KeyPrefix = resolveEnvString(temp.KeyPrefix)

	if r.SelectDB, err = parseField(temp.SelectDB, 0, strconv.Atoi, "redis.select_db"); err != nil {
		return err
	}
	if r.TLSEnabled, err = parseField(temp.TLSEnabled, false, strconv.ParseBool, "redis.tls_enabled"); err != nil {
		return err
	}
	if r.ForceSingleClient, err = parseField(temp.ForceSingleClient, false, strconv.ParseBool, "redis.force_single_client"); err != nil {
		return err
	}
	if r.ConnectTimeout, err = parseField(temp.ConnectTimeout, 5*time.Second, time.ParseDuration, "redis.connect_timeout"); err != nil {
		return err
	}
	if r.ConnWriteTimeout, err = parseField(temp.ConnWriteTimeout, 10*time.Second, time.ParseDuration, "redis.conn_write_timeout"); err != nil {
		return err
	}

	// Pool settings
	if r.MinIdleConns, err = parseField(temp.MinIdleConns, 10, strconv.Atoi, "redis.min_idle_conns"); err != nil {
		return err
	}
	if r.MaxIdleConns, err = parseField(temp.MaxIdleConns, 100, strconv.Atoi, "redis.max_idle_conns"); err != nil {
		return err
	}
	if r.MaxConnLifetime, err = parseField(temp.MaxConnLifetime, 30*time.Minute, time.ParseDuration, "redis.max_conn_lifetime"); err != nil {
		return err
	}

	// Key TTL
	if r.KeyTTL, err = parseField(temp.KeyTTL, 120, strconv.Atoi, "redis.key_ttl"); err != nil {
		return err
	}

	// Command timeout
	if r.CommandTimeout, err = parseField(temp.CommandTimeout, 3*time.Second, time.ParseDuration, "redis.command_timeout"); err != nil {
		return err
	}

	if r.Hybrid, err = parseField(temp.Hybrid, false, strconv.ParseBool, "redis.hybrid"); err != nil {
		return err
	}
	if r.SyncInterval, err = parseField(temp.SyncInterval, 5*time.Second, time.ParseDuration, "redis.sync_interval"); err != nil {
		return err
	}

	// Apply default key prefix
	if r.KeyPrefix == "" {
		r.KeyPrefix = "rl:"
	}

	return nil
}

type ServerConfig struct {
	Port                       int           `yaml:"port"`
	MaxBodySizeMB              int           `yaml:"max_body_size_mb"`
	ResponseBodyMultiplier     int           `yaml:"response_body_multiplier"` // Multiplier for response body size limit relative to max_body_size_mb (default: 10)
	RequestTimeout             time.Duration `yaml:"request_timeout"`
	LoggingLevel               string        `yaml:"logging_level"`
	StdoutLogsEnabled          bool          `yaml:"stdout_logs_enabled"` // Write logs to stdout (default: true); disable to ship logs only via OTEL
	MasterKey                  string        `yaml:"master_key"`
	DefaultModelsRPM           int           `yaml:"default_models_rpm"`
	MaxIdleConns               int           `yaml:"max_idle_conns"`
	MaxIdleConnsPerHost        int           `yaml:"max_idle_conns_per_host"`
	IdleConnTimeout            time.Duration `yaml:"idle_conn_timeout"`
	ReadTimeout                time.Duration `yaml:"-"`                                 // HTTP server read timeout (equals request_timeout, not configurable via YAML)
	WriteTimeout               time.Duration `yaml:"write_timeout"`                     // HTTP server write timeout (default: 60s)
	IdleTimeout                time.Duration `yaml:"idle_timeout"`                      // HTTP server idle timeout (default: 2*write_timeout)
	MaxProviderRetries         int           `yaml:"max_provider_retries"`              // Max same-type credential retries on provider errors (default: 2, meaning 3 total attempts)
	MaxFallbackAttempts        int           `yaml:"max_fallback_attempts"`             // Max fallback proxy hops per request chain (default: 5)
	SessionStickyEnabled       bool          `yaml:"session_sticky_enabled"`            // Enable session-sticky credential routing (default: true)
	SessionStickyTTL           int           `yaml:"session_sticky_ttl_minutes"`        // Session binding TTL in minutes (0 = default 6)
	SessionStickyAutoCacheCtrl bool          `yaml:"session_sticky_auto_cache_control"` // Auto-inject Anthropic cache_control when session is active (default: true)
	ModelPricesLink            string        `yaml:"model_prices_link,omitempty"`       // URL or file path to model prices JSON - supports os.environ/VAR_NAME
	ShutdownDelay              time.Duration `yaml:"shutdown_delay"`                    // Delay between readiness=false and server.Shutdown (default: 5s)
	DrainUpstreamOnAbort       bool          `yaml:"drain_upstream_on_abort"`           // When true, keep reading upstream after client disconnect to capture real usage chunk (default: false — estimate from delta text)
	ProxyHealthTimeout         time.Duration `yaml:"proxy_health_timeout"`              // Timeout for fetching /health from remote proxy credentials (default: 15s)
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
		Port                       string `yaml:"port"`
		MaxBodySizeMB              string `yaml:"max_body_size_mb"`
		ResponseBodyMultiplier     string `yaml:"response_body_multiplier"`
		RequestTimeout             string `yaml:"request_timeout"`
		LoggingLevel               string `yaml:"logging_level"`
		StdoutLogsEnabled          string `yaml:"stdout_logs_enabled"`
		MasterKey                  string `yaml:"master_key"`
		DefaultModelsRPM           string `yaml:"default_models_rpm"`
		MaxIdleConns               string `yaml:"max_idle_conns"`
		MaxIdleConnsPerHost        string `yaml:"max_idle_conns_per_host"`
		IdleConnTimeout            string `yaml:"idle_conn_timeout"`
		WriteTimeout               string `yaml:"write_timeout"`
		IdleTimeout                string `yaml:"idle_timeout"`
		MaxProviderRetries         string `yaml:"max_provider_retries"`
		MaxFallbackAttempts        string `yaml:"max_fallback_attempts"`
		SessionStickyEnabled       string `yaml:"session_sticky_enabled"`
		SessionStickyTTL           string `yaml:"session_sticky_ttl_minutes"`
		SessionStickyAutoCacheCtrl string `yaml:"session_sticky_auto_cache_control"`
		ModelPricesLink            string `yaml:"model_prices_link,omitempty"`
		ShutdownDelay              string `yaml:"shutdown_delay"`
		DrainUpstreamOnAbort       string `yaml:"drain_upstream_on_abort"`
		ProxyHealthTimeout         string `yaml:"proxy_health_timeout"`
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
	// Max fallback proxy hops per request chain (default: 5)
	if s.MaxFallbackAttempts, err = parseField(temp.MaxFallbackAttempts, 5, strconv.Atoi, "max_fallback_attempts"); err != nil {
		return err
	}
	if s.SessionStickyEnabled, err = parseField(temp.SessionStickyEnabled, true, strconv.ParseBool, "session_sticky_enabled"); err != nil {
		return err
	}
	if s.SessionStickyTTL, err = parseField(temp.SessionStickyTTL, 0, strconv.Atoi, "session_sticky_ttl_minutes"); err != nil {
		return err
	}
	if s.SessionStickyAutoCacheCtrl, err = parseField(temp.SessionStickyAutoCacheCtrl, true, strconv.ParseBool, "session_sticky_auto_cache_control"); err != nil {
		return err
	}
	if s.StdoutLogsEnabled, err = parseField(temp.StdoutLogsEnabled, true, strconv.ParseBool, "stdout_logs_enabled"); err != nil {
		return err
	}

	if s.ShutdownDelay, err = parseField(temp.ShutdownDelay, 5*time.Second, time.ParseDuration, "shutdown_delay"); err != nil {
		return err
	}
	if s.DrainUpstreamOnAbort, err = parseField(temp.DrainUpstreamOnAbort, false, strconv.ParseBool, "drain_upstream_on_abort"); err != nil {
		return err
	}
	if s.ProxyHealthTimeout, err = parseField(temp.ProxyHealthTimeout, 15*time.Second, time.ParseDuration, "proxy_health_timeout"); err != nil {
		return err
	}

	// String fields
	s.LoggingLevel = resolveEnvString(temp.LoggingLevel)
	s.MasterKey = resolveEnvString(temp.MasterKey)
	s.ModelPricesLink = resolveEnvString(temp.ModelPricesLink)

	return nil
}

type CredentialConfig struct {
	Name                    string            `yaml:"name"`
	Type                    ProviderType      `yaml:"type"`
	APIKey                  string            `yaml:"api_key"`
	BaseURL                 string            `yaml:"base_url"`
	AuthType                string            `yaml:"auth_type,omitempty"`
	RPM                     int               `yaml:"rpm"`
	TPM                     int               `yaml:"tpm"`
	Weight                  int               `yaml:"weight"` // Default weighted round-robin weight for this credential (0 = 1)
	FallbackPriority        int               `yaml:"fallback_priority,omitempty"`
	Scopes                  []string          `yaml:"scopes,omitempty"`
	DeniedScopes            []string          `yaml:"denied_scopes,omitempty"`
	ProviderScopes          []string          `yaml:"-"`
	ProviderDeniedScopes    []string          `yaml:"-"`
	ProviderScopeExpression *scope.Expression `yaml:"-"`
	ProviderScopeKnown      bool              `yaml:"-"`

	// Models associated with this credential (used for x-model-templates)
	Models []ModelRPMConfig `yaml:"models,omitempty"`

	// Vertex AI specific fields
	ProjectID       string `yaml:"project_id,omitempty"`
	Location        string `yaml:"location,omitempty"`
	CredentialsFile string `yaml:"credentials_file,omitempty"`
	CredentialsJSON string `yaml:"credentials_json,omitempty"`

	// Proxy specific fields
	IsFallback bool `yaml:"is_fallback,omitempty"`
}

func (c CredentialConfig) VisibleTo(visibility scope.Context) bool {
	return visibility.AllowsExpression(c.ScopeExpression())
}

func (c CredentialConfig) ScopeExpression() *scope.Expression {
	providerExpression := c.ProviderScopeExpression
	if providerExpression == nil {
		providerExpression = scope.FromScopes(c.ProviderScopes, c.ProviderDeniedScopes)
	}
	return scope.And(
		scope.FromScopes(c.Scopes, c.DeniedScopes),
		providerExpression,
	)
}

// SameProviderIdentity reports whether learned provider metadata can be reused.
func (c CredentialConfig) SameProviderIdentity(other CredentialConfig) bool {
	return c.Name == other.Name &&
		c.Type == other.Type &&
		c.BaseURL == other.BaseURL &&
		c.APIKey == other.APIKey &&
		c.AuthType == other.AuthType &&
		c.IsFallback == other.IsFallback
}

// UnmarshalYAML implements custom unmarshaling for CredentialConfig with env variable support
func (c *CredentialConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with all string fields
	type tempConfig struct {
		Name             string           `yaml:"name"`
		Type             string           `yaml:"type"`
		APIKey           string           `yaml:"api_key"`
		BaseURL          string           `yaml:"base_url"`
		AuthType         string           `yaml:"auth_type,omitempty"`
		RPM              string           `yaml:"rpm"`
		TPM              string           `yaml:"tpm"`
		Weight           string           `yaml:"weight"`
		FallbackPriority string           `yaml:"fallback_priority,omitempty"`
		Scopes           []string         `yaml:"scopes,omitempty"`
		DeniedScopes     []string         `yaml:"denied_scopes,omitempty"`
		ForbiddenScopes  []string         `yaml:"forbidden_scopes,omitempty"`
		ProjectID        string           `yaml:"project_id,omitempty"`
		Location         string           `yaml:"location,omitempty"`
		CredentialsFile  string           `yaml:"credentials_file,omitempty"`
		CredentialsJSON  string           `yaml:"credentials_json,omitempty"`
		IsFallback       string           `yaml:"is_fallback,omitempty"`
		Models           []ModelRPMConfig `yaml:"models,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	// Resolve string fields
	c.Name = resolveEnvString(temp.Name)
	c.Type = normalizeProviderType(resolveEnvString(temp.Type))
	c.APIKey = resolveEnvString(temp.APIKey)
	c.BaseURL = resolveEnvString(temp.BaseURL)
	c.AuthType = strings.ToLower(resolveEnvString(temp.AuthType))
	c.Scopes = scope.NormalizeList(temp.Scopes)
	c.DeniedScopes = scope.NormalizeList(append(temp.DeniedScopes, temp.ForbiddenScopes...))

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
	if c.Weight, err = parseField(temp.Weight, 0, strconv.Atoi, "weight for credential '"+c.Name+"'"); err != nil {
		return err
	}
	if c.FallbackPriority, err = parseField(temp.FallbackPriority, 0, strconv.Atoi, "fallback_priority for credential '"+c.Name+"'"); err != nil {
		return err
	}

	// Resolve and parse boolean field
	if c.IsFallback, err = parseField(temp.IsFallback, false, strconv.ParseBool, "is_fallback for credential '"+c.Name+"'"); err != nil {
		return err
	}

	// Copy models decoded via YAML anchors / inline definitions
	c.Models = temp.Models

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
	// IsRequired specifies whether LiteLLM DB is mandatory (fail startup on error)
	// or optional (degrade to NoopManager with warning on error)
	LoadLitellmDBModels   bool          `yaml:"load_db_models"`         // default: false
	LitellmDBSyncInterval time.Duration `yaml:"db_model_sync_interval"` // default: 1m

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

	// DisableSpendLogsWrite disables writing SpendLogEntry/Daily* aggregates to
	// Postgres while leaving auth (ValidateToken) untouched. Intended for setups
	// where Kafka (see KafkaConfig) is the sole spend-analytics write-path.
	DisableSpendLogsWrite bool `yaml:"disable_spend_logs_write"` // default: false

	// EnforceBudgetReservation enables atomic Redis-backed budget pre-reservation
	// (prevents the overspend race described in todo_auth_billing.md P1.4).
	// Auto-disabled (safe no-op) if redis.enabled=false, regardless of this flag.
	EnforceBudgetReservation bool `yaml:"enforce_budget_reservation"` // default: false (opt-in)
	// BudgetReservationTTL bounds how long a Redis budget counter survives without
	// activity before it expires and is reseeded from the DB spend value on next use.
	// Also bounds how stale an out-of-band DB spend reset (e.g. admin resetting
	// budget_duration) can make the counter before it self-corrects.
	BudgetReservationTTL time.Duration `yaml:"budget_reservation_ttl"` // default: 15m
	// EnforceKeyRateLimits enables atomic Redis-backed RPM/TPM enforcement per
	// key/user/team/org/team-member/org-member (previously loaded but unused fields).
	EnforceKeyRateLimits bool `yaml:"enforce_key_rate_limits"` // default: false (opt-in)
	// DefaultEstimatedCompletionTokens is the completion-token estimate used for
	// budget pre-reservation when the request doesn't specify max_tokens.
	DefaultEstimatedCompletionTokens int `yaml:"default_estimated_completion_tokens"` // default: 1000
}

// KafkaConfig holds configuration for the Kafka spend-log analytics write-path
// (internal/kafkalog). When Enabled=false (default) no producer is started and
// spend events are only written to LiteLLM Postgres (unless disabled there too).
type KafkaConfig struct {
	Enabled bool `yaml:"enabled"`

	// Brokers is the list of Kafka bootstrap broker addresses (host:port).
	Brokers []string `yaml:"brokers"`

	// Topic is the Kafka topic spend events are published to.
	Topic string `yaml:"topic"` // default: "air.spend_logs"

	// ClientID identifies this producer to the Kafka cluster.
	ClientID string `yaml:"client_id"` // default: "auto_ai_router"

	// Async producer queue/batch settings (mirrors litellm_db.log_*).
	LogQueueSize     int           `yaml:"log_queue_size"`     // default: 5000
	LogBatchSize     int           `yaml:"log_batch_size"`     // default: 100
	LogFlushInterval time.Duration `yaml:"log_flush_interval"` // default: 5s

	// TLS/SASL — optional, for production clusters.
	TLSEnabled    bool   `yaml:"tls_enabled,omitempty"`
	SASLMechanism string `yaml:"sasl_mechanism,omitempty"` // "" | "PLAIN" | "SCRAM-SHA-256" | "SCRAM-SHA-512"
	SASLUsername  string `yaml:"sasl_username,omitempty"`
	SASLPassword  string `yaml:"sasl_password,omitempty"`
}

const (
	SpendAPIBase = "http://air-ru01/v1"
)

// SpendLogConfig owns a database connection that is independent from the
// LiteLLM control-plane/auth connection.
type SpendLogConfig struct {
	DatabaseURL          string        `yaml:"database_url"`
	ExpectedDatabaseName string        `yaml:"expected_database_name"`
	APIBase              string        `yaml:"api_base"`
	MaxConns             int           `yaml:"max_conns"`
	MinConns             int           `yaml:"min_conns"`
	HealthCheckInterval  time.Duration `yaml:"health_check_interval"`
	ConnectTimeout       time.Duration `yaml:"connect_timeout"`
	LogQueueSize         int           `yaml:"log_queue_size"`
	LogBatchSize         int           `yaml:"log_batch_size"`
	LogFlushInterval     time.Duration `yaml:"log_flush_interval"`
}

// IsEnabled reports whether an isolated spend destination is configured.
// Omitting database_url disables the writer; no separate mode flag is needed.
func (s SpendLogConfig) IsEnabled() bool {
	return strings.TrimSpace(s.DatabaseURL) != ""
}

// OTELConfig holds OpenTelemetry export configuration for logs, traces and metrics.
// When Enabled=false (default) no OTEL SDK components are initialized and the
// router behaves exactly as before (pretty stdout logs, no tracing).
type OTELConfig struct {
	Enabled bool `yaml:"enabled"`

	// Endpoint is the OTLP collector endpoint.
	// For grpc protocol: "host:port" (e.g. "localhost:4317").
	// For http protocol: "host:port" or full URL (e.g. "http://localhost:4318").
	Endpoint string `yaml:"endpoint"`

	// Protocol selects the OTLP transport: "grpc" (default) or "http" (http/protobuf).
	Protocol string `yaml:"protocol"`

	// Insecure disables TLS for the exporter connection (default: true,
	// matching the typical in-cluster collector setup).
	Insecure bool `yaml:"insecure"`

	// ServiceName is reported as service.name resource attribute (default: "auto-ai-router").
	ServiceName string `yaml:"service_name"`

	// Headers are added to every OTLP export request (e.g. auth tokens).
	// Values support os.environ/VAR_NAME resolution.
	Headers map[string]string `yaml:"headers,omitempty"`

	// LogsEnabled ships slog records via OTLP in addition to stdout (default: true).
	LogsEnabled bool `yaml:"logs_enabled"`

	// TracesEnabled creates server/client spans and propagates trace context
	// to upstream providers and chained routers (default: true).
	TracesEnabled bool `yaml:"traces_enabled"`

	// MetricExportInterval is the period between OTLP metric pushes (default: 60s).
	// OTLP metric export is driven by monitoring.prometheus_enabled (the metrics
	// master switch) — when OTEL is enabled and metrics are being collected, the
	// Prometheus registry is bridged to the collector and pushed on this interval.
	MetricExportInterval time.Duration `yaml:"metric_export_interval"`

	// TraceSampleRatio is the head sampling ratio in [0.0, 1.0] (default: 1.0).
	// The sampler is parent-based, so sampled upstream decisions are respected.
	TraceSampleRatio float64 `yaml:"trace_sample_ratio"`

	// TrustIncomingTraceparent controls whether the server adopts an incoming
	// W3C traceparent header and parents its span under it (default: true).
	// Enable when a trusted hop (e.g. a LiteLLM proxy with
	// forward_traceparent_to_llm_provider) sits in front, so the router's spans
	// nest inside that caller's trace. Disable for standalone or public-facing
	// deployments to ignore client-supplied trace context and start a fresh root
	// span per request. Outgoing traceparent propagation to upstreams is
	// unaffected either way.
	TrustIncomingTraceparent bool `yaml:"trust_incoming_traceparent"`
}

// UnmarshalYAML implements custom unmarshaling for OTELConfig with env variable support
func (o *OTELConfig) UnmarshalYAML(value *yaml.Node) error {
	type tempConfig struct {
		Enabled                  string            `yaml:"enabled"`
		Endpoint                 string            `yaml:"endpoint"`
		Protocol                 string            `yaml:"protocol"`
		Insecure                 string            `yaml:"insecure"`
		ServiceName              string            `yaml:"service_name"`
		Headers                  map[string]string `yaml:"headers,omitempty"`
		LogsEnabled              string            `yaml:"logs_enabled"`
		TracesEnabled            string            `yaml:"traces_enabled"`
		MetricExportInterval     string            `yaml:"metric_export_interval"`
		TraceSampleRatio         string            `yaml:"trace_sample_ratio"`
		TrustIncomingTraceparent string            `yaml:"trust_incoming_traceparent"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	var err error
	if o.Enabled, err = parseField(temp.Enabled, false, strconv.ParseBool, "otel.enabled"); err != nil {
		return err
	}
	if o.Insecure, err = parseField(temp.Insecure, true, strconv.ParseBool, "otel.insecure"); err != nil {
		return err
	}
	if o.LogsEnabled, err = parseField(temp.LogsEnabled, true, strconv.ParseBool, "otel.logs_enabled"); err != nil {
		return err
	}
	if o.TracesEnabled, err = parseField(temp.TracesEnabled, true, strconv.ParseBool, "otel.traces_enabled"); err != nil {
		return err
	}
	if o.MetricExportInterval, err = parseField(temp.MetricExportInterval, 60*time.Second, time.ParseDuration, "otel.metric_export_interval"); err != nil {
		return err
	}
	parseFloat := func(s string) (float64, error) { return strconv.ParseFloat(s, 64) }
	if o.TraceSampleRatio, err = parseField(temp.TraceSampleRatio, 1.0, parseFloat, "otel.trace_sample_ratio"); err != nil {
		return err
	}
	if o.TrustIncomingTraceparent, err = parseField(temp.TrustIncomingTraceparent, true, strconv.ParseBool, "otel.trust_incoming_traceparent"); err != nil {
		return err
	}

	o.Endpoint = resolveEnvString(temp.Endpoint)
	o.Protocol = strings.ToLower(resolveEnvString(temp.Protocol))
	o.ServiceName = resolveEnvString(temp.ServiceName)

	if len(temp.Headers) > 0 {
		o.Headers = make(map[string]string, len(temp.Headers))
		for k, v := range temp.Headers {
			o.Headers[k] = resolveEnvString(v)
		}
	}

	o.applyDefaults()
	return nil
}

// applyDefaults fills in default values for empty OTEL config fields.
func (o *OTELConfig) applyDefaults() {
	if o.Protocol == "" {
		o.Protocol = "grpc"
	}
	if o.ServiceName == "" {
		o.ServiceName = "auto-ai-router"
	}
	if o.Endpoint == "" {
		if o.Protocol == "grpc" {
			o.Endpoint = "localhost:4317"
		} else {
			o.Endpoint = "localhost:4318"
		}
	}
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
		Enabled               string `yaml:"enabled"`
		IsRequired            string `yaml:"is_required"`
		LoadLitellmDBModels   string `yaml:"load_db_models"`
		LitellmDBSyncInterval string `yaml:"db_model_sync_interval"`
		DatabaseURL           string `yaml:"database_url"`
		MaxConns              string `yaml:"max_conns"`
		MinConns              string `yaml:"min_conns"`
		HealthCheckInterval   string `yaml:"health_check_interval"`
		ConnectTimeout        string `yaml:"connect_timeout"`
		AuthCacheTTL          string `yaml:"auth_cache_ttl"`
		AuthCacheSize         string `yaml:"auth_cache_size"`
		LogQueueSize          string `yaml:"log_queue_size"`
		LogBatchSize          string `yaml:"log_batch_size"`
		LogFlushInterval      string `yaml:"log_flush_interval"`
		DisableSpendLogsWrite string `yaml:"disable_spend_logs_write"`

		EnforceBudgetReservation         string `yaml:"enforce_budget_reservation"`
		BudgetReservationTTL             string `yaml:"budget_reservation_ttl"`
		EnforceKeyRateLimits             string `yaml:"enforce_key_rate_limits"`
		DefaultEstimatedCompletionTokens string `yaml:"default_estimated_completion_tokens"`
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
	if l.LoadLitellmDBModels, err = parseField(temp.LoadLitellmDBModels, false, strconv.ParseBool, "litellm_db.load_db_models"); err != nil {
		return err
	}
	if l.DisableSpendLogsWrite, err = parseField(temp.DisableSpendLogsWrite, false, strconv.ParseBool, "litellm_db.disable_spend_logs_write"); err != nil {
		return err
	}
	if l.EnforceBudgetReservation, err = parseField(temp.EnforceBudgetReservation, false, strconv.ParseBool, "litellm_db.enforce_budget_reservation"); err != nil {
		return err
	}
	if l.EnforceKeyRateLimits, err = parseField(temp.EnforceKeyRateLimits, false, strconv.ParseBool, "litellm_db.enforce_key_rate_limits"); err != nil {
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
	if l.DefaultEstimatedCompletionTokens, err = parseField(temp.DefaultEstimatedCompletionTokens, 1000, strconv.Atoi, "litellm_db.default_estimated_completion_tokens"); err != nil {
		return err
	}
	// Duration fields
	if l.HealthCheckInterval, err = parseField(temp.HealthCheckInterval, 10*time.Second, time.ParseDuration, "litellm_db.health_check_interval"); err != nil {
		return err
	}
	if l.LitellmDBSyncInterval, err = parseField(temp.LitellmDBSyncInterval, 1*time.Minute, time.ParseDuration, "litellm_db.db_model_sync_interval"); err != nil {
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
	if l.BudgetReservationTTL, err = parseField(temp.BudgetReservationTTL, 15*time.Minute, time.ParseDuration, "litellm_db.budget_reservation_ttl"); err != nil {
		return err
	}

	return nil
}

// UnmarshalYAML resolves environment-backed spend-writer settings and applies
// safe defaults even when only part of spend_log is configured.
func (s *SpendLogConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawSpendLog struct {
		DatabaseURL          string `yaml:"database_url"`
		ExpectedDatabaseName string `yaml:"expected_database_name"`
		APIBase              string `yaml:"api_base"`
		MaxConns             string `yaml:"max_conns"`
		MinConns             string `yaml:"min_conns"`
		HealthCheckInterval  string `yaml:"health_check_interval"`
		ConnectTimeout       string `yaml:"connect_timeout"`
		LogQueueSize         string `yaml:"log_queue_size"`
		LogBatchSize         string `yaml:"log_batch_size"`
		LogFlushInterval     string `yaml:"log_flush_interval"`
	}

	var raw rawSpendLog
	if err := value.Decode(&raw); err != nil {
		return err
	}

	defaults := defaultSpendLogConfig()
	s.DatabaseURL = resolveEnvString(raw.DatabaseURL)
	s.ExpectedDatabaseName = resolveEnvString(raw.ExpectedDatabaseName)
	s.APIBase = resolveEnvString(raw.APIBase)
	if s.APIBase == "" {
		s.APIBase = defaults.APIBase
	}

	var err error
	if s.MaxConns, err = parseField(raw.MaxConns, defaults.MaxConns, strconv.Atoi, "spend_log.max_conns"); err != nil {
		return err
	}
	if s.MinConns, err = parseField(raw.MinConns, defaults.MinConns, strconv.Atoi, "spend_log.min_conns"); err != nil {
		return err
	}
	if s.LogQueueSize, err = parseField(raw.LogQueueSize, defaults.LogQueueSize, strconv.Atoi, "spend_log.log_queue_size"); err != nil {
		return err
	}
	if s.LogBatchSize, err = parseField(raw.LogBatchSize, defaults.LogBatchSize, strconv.Atoi, "spend_log.log_batch_size"); err != nil {
		return err
	}
	if s.HealthCheckInterval, err = parseField(raw.HealthCheckInterval, defaults.HealthCheckInterval, time.ParseDuration, "spend_log.health_check_interval"); err != nil {
		return err
	}
	if s.ConnectTimeout, err = parseField(raw.ConnectTimeout, defaults.ConnectTimeout, time.ParseDuration, "spend_log.connect_timeout"); err != nil {
		return err
	}
	if s.LogFlushInterval, err = parseField(raw.LogFlushInterval, defaults.LogFlushInterval, time.ParseDuration, "spend_log.log_flush_interval"); err != nil {
		return err
	}
	return nil
}

// UnmarshalYAML implements custom unmarshaling for KafkaConfig with env variable support.
func (k *KafkaConfig) UnmarshalYAML(value *yaml.Node) error {
	type tempConfig struct {
		Enabled          string   `yaml:"enabled"`
		Brokers          []string `yaml:"brokers"`
		Topic            string   `yaml:"topic"`
		ClientID         string   `yaml:"client_id"`
		LogQueueSize     string   `yaml:"log_queue_size"`
		LogBatchSize     string   `yaml:"log_batch_size"`
		LogFlushInterval string   `yaml:"log_flush_interval"`
		TLSEnabled       string   `yaml:"tls_enabled,omitempty"`
		SASLMechanism    string   `yaml:"sasl_mechanism,omitempty"`
		SASLUsername     string   `yaml:"sasl_username,omitempty"`
		SASLPassword     string   `yaml:"sasl_password,omitempty"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	var err error
	if k.Enabled, err = parseField(temp.Enabled, false, strconv.ParseBool, "kafka.enabled"); err != nil {
		return err
	}
	k.Brokers = make([]string, 0, len(temp.Brokers))
	for _, broker := range temp.Brokers {
		resolved := resolveEnvString(broker)
		for part := range strings.SplitSeq(resolved, ",") {
			part = strings.TrimSpace(part)
			if part != "" {
				k.Brokers = append(k.Brokers, part)
			}
		}
	}
	k.Topic = resolveEnvString(temp.Topic)
	k.ClientID = resolveEnvString(temp.ClientID)
	if k.LogQueueSize, err = parseField(temp.LogQueueSize, 5000, strconv.Atoi, "kafka.log_queue_size"); err != nil {
		return err
	}
	if k.LogBatchSize, err = parseField(temp.LogBatchSize, 100, strconv.Atoi, "kafka.log_batch_size"); err != nil {
		return err
	}
	if k.LogFlushInterval, err = parseField(temp.LogFlushInterval, 5*time.Second, time.ParseDuration, "kafka.log_flush_interval"); err != nil {
		return err
	}
	if k.TLSEnabled, err = parseField(temp.TLSEnabled, false, strconv.ParseBool, "kafka.tls_enabled"); err != nil {
		return err
	}
	k.SASLMechanism = resolveEnvString(temp.SASLMechanism)
	k.SASLUsername = resolveEnvString(temp.SASLUsername)
	k.SASLPassword = resolveEnvString(temp.SASLPassword)
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

	// Resolve YAML aliases first (for x-model-templates anchors)
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

	if !hasMappingKey(&root, "monitoring") {
		cfg.Monitoring = defaultMonitoringConfig()
	}

	if !hasMappingKey(&root, "redis") {
		cfg.Redis = defaultRedisConfig()
	}

	if !hasMappingKey(&root, "litellm_db") {
		cfg.LiteLLMDB = defaultLiteLLMDBConfig()
	}

	if !hasMappingKey(&root, "spend_log") {
		cfg.SpendLog = defaultSpendLogConfig()
	}

	if !hasMappingKey(&root, "otel") {
		cfg.OTEL = defaultOTELConfig()
	}

	if !hasMappingKey(&root, "kafka") {
		cfg.Kafka = defaultKafkaConfig()
	}

	// Ensure HealthCheckPath is always set regardless of whether monitoring section exists.
	// MonitoringConfig.UnmarshalYAML is only called when a "monitoring:" key is present in YAML.
	cfg.Monitoring.HealthCheckPath = "/health"

	// Resolve env variables in model_alias values
	if cfg.ModelAlias != nil {
		resolved := make(map[string]string, len(cfg.ModelAlias))
		for alias, target := range cfg.ModelAlias {
			resolved[resolveEnvString(alias)] = resolveEnvString(target)
		}
		cfg.ModelAlias = resolved
	}
	if cfg.ClientModelIDs != nil {
		resolved := make([]string, len(cfg.ClientModelIDs))
		for index, modelID := range cfg.ClientModelIDs {
			resolved[index] = resolveEnvString(modelID)
		}
		cfg.ClientModelIDs = resolved
	}
	if cfg.PublicModelAlias != nil {
		resolved := make(map[string]string, len(cfg.PublicModelAlias))
		for alias, target := range cfg.PublicModelAlias {
			resolved[resolveEnvString(alias)] = resolveEnvString(target)
		}
		cfg.PublicModelAlias = resolved
	}
	if cfg.AcceptedModelAlias != nil {
		resolved := make(map[string]string, len(cfg.AcceptedModelAlias))
		for alias, target := range cfg.AcceptedModelAlias {
			resolved[resolveEnvString(alias)] = resolveEnvString(target)
		}
		cfg.AcceptedModelAlias = resolved
	}

	if cfg.Credentials == nil {
		cfg.Credentials = []CredentialConfig{}
	}

	if cfg.Models == nil {
		cfg.Models = []ModelRPMConfig{}
	}

	if cfg.ModelAlias == nil {
		cfg.ModelAlias = map[string]string{}
	}
	if cfg.PublicModelAlias == nil {
		cfg.PublicModelAlias = map[string]string{}
	}
	if cfg.AcceptedModelAlias == nil {
		cfg.AcceptedModelAlias = map[string]string{}
	}

	// Extract models from credentials and add to main Models list
	// Models defined in credentials are "unpacked" to the main models list
	for _, cred := range cfg.Credentials {
		for _, model := range cred.Models {
			// Create a copy of the model with the credential reference set
			expandedModel := model
			if expandedModel.Credential == "" {
				expandedModel.Credential = cred.Name
			}
			if expandedModel.Name == "" {
				continue
			}
			cfg.Models = append(cfg.Models, expandedModel)
		}
	}

	// Clear models from credentials (they have been unpacked to main Models)
	for i := range cfg.Credentials {
		cfg.Credentials[i].Models = nil
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

func defaultMonitoringConfig() MonitoringConfig {
	return MonitoringConfig{
		PrometheusEnabled: true,
		HealthCheckPath:   "/health",
		LogErrors:         false,
		ErrorsLogPath:     "logs/logs.jsonl",
	}
}

func defaultRedisConfig() RedisConfig {
	return RedisConfig{
		Enabled:           false,
		InitAddresses:     nil,
		Username:          "",
		Password:          "",
		SelectDB:          0,
		KeyPrefix:         "rl:",
		TLSEnabled:        false,
		ConnectTimeout:    5 * time.Second,
		ConnWriteTimeout:  10 * time.Second,
		ForceSingleClient: false,
		MinIdleConns:      10,
		MaxIdleConns:      100,
		MaxConnLifetime:   30 * time.Minute,
		KeyTTL:            120,
		CommandTimeout:    3 * time.Second,
	}
}

func defaultKafkaConfig() KafkaConfig {
	return KafkaConfig{
		Enabled:          false,
		Brokers:          nil,
		Topic:            "air.spend_logs",
		ClientID:         "auto_ai_router",
		LogQueueSize:     5000,
		LogBatchSize:     100,
		LogFlushInterval: 5 * time.Second,
	}
}

func defaultLiteLLMDBConfig() LiteLLMDBConfig {
	return LiteLLMDBConfig{
		Enabled:                          false,
		IsRequired:                       false,
		LoadLitellmDBModels:              false,
		LitellmDBSyncInterval:            1 * time.Minute,
		DatabaseURL:                      "",
		MaxConns:                         25,
		MinConns:                         5,
		HealthCheckInterval:              10 * time.Second,
		ConnectTimeout:                   5 * time.Second,
		AuthCacheTTL:                     5 * time.Second,
		AuthCacheSize:                    10000,
		LogQueueSize:                     5000,
		LogBatchSize:                     100,
		LogFlushInterval:                 5 * time.Second,
		EnforceBudgetReservation:         false,
		BudgetReservationTTL:             15 * time.Minute,
		EnforceKeyRateLimits:             false,
		DefaultEstimatedCompletionTokens: 1000,
	}
}

func defaultSpendLogConfig() SpendLogConfig {
	return SpendLogConfig{
		APIBase:             SpendAPIBase,
		MaxConns:            10,
		MinConns:            2,
		HealthCheckInterval: 10 * time.Second,
		ConnectTimeout:      5 * time.Second,
		LogQueueSize:        5000,
		LogBatchSize:        100,
		LogFlushInterval:    5 * time.Second,
	}
}

// MetricsCollectionEnabled reports whether request/token metrics should be
// recorded into the Prometheus registry. Collection is decoupled from how the
// metrics leave the process: it is needed both for the pull-based /metrics
// endpoint (monitoring.prometheus_enabled) and for OTLP push (otel.enabled),
// which bridges the same registry. Either sink alone is enough to require it.
func (c *Config) MetricsCollectionEnabled() bool {
	return c.Monitoring.PrometheusEnabled || c.OTEL.Enabled
}

func defaultOTELConfig() OTELConfig {
	cfg := OTELConfig{
		Enabled:                  false,
		Insecure:                 true,
		LogsEnabled:              true,
		TracesEnabled:            true,
		MetricExportInterval:     60 * time.Second,
		TraceSampleRatio:         1.0,
		TrustIncomingTraceparent: true,
	}
	cfg.applyDefaults()
	return cfg
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

	// Validate MaxFallbackAttempts (0 = use default of 5)
	if c.Server.MaxFallbackAttempts < 0 {
		return fmt.Errorf("invalid max_fallback_attempts: %d (must be >= 0)", c.Server.MaxFallbackAttempts)
	}
	if c.Server.SessionStickyTTL < 0 {
		return fmt.Errorf("invalid session_sticky_ttl_minutes: %d (must be >= 0)", c.Server.SessionStickyTTL)
	}

	// Validate IdleTimeout against WriteTimeout
	if c.Server.IdleTimeout == 0 {
		c.Server.IdleTimeout = c.Server.WriteTimeout * 2
	}

	if c.Fail2Ban.MaxAttempts <= 0 {
		return fmt.Errorf("invalid max_attempts: %d", c.Fail2Ban.MaxAttempts)
	}

	if len(c.Credentials) == 0 && !c.LiteLLMDB.Enabled {
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
			return fmt.Errorf("credential %s: invalid type: %s (must be 'openai', 'vertex-ai', 'gemini', 'anthropic', 'cometapi', 'bedrock', or 'proxy')", cred.Name, cred.Type)
		}
		if cred.AuthType != "" && cred.AuthType != "bearer" && cred.AuthType != "x-api-key" {
			return fmt.Errorf("credential %s: invalid auth_type: %s (must be 'bearer' or 'x-api-key')", cred.Name, cred.AuthType)
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
		// Weight: 0 means default (1), positive means a higher share. Negative is invalid.
		if cred.Weight < 0 {
			return fmt.Errorf("credential %s: invalid weight: %d (must be 0 for default or positive number)", cred.Name, cred.Weight)
		}
		if cred.FallbackPriority < 0 {
			return fmt.Errorf("credential %s: invalid fallback_priority: %d (must be >= 0)", cred.Name, cred.FallbackPriority)
		}
		if cred.IsFallback && cred.FallbackPriority > 0 {
			return fmt.Errorf("credential %s: invalid fallback_priority: fallback credentials cannot set fallback_priority", cred.Name)
		}
	}

	for _, model := range c.Models {
		if model.Weight < 0 {
			return fmt.Errorf("model %s: invalid weight: %d (must be 0 for default or positive number)", model.Name, model.Weight)
		}
	}

	// client_model_ids is an explicit product boundary. When present (including
	// an empty list), AIR must not infer public request identifiers from provider
	// backend names. Keep it exact and require compatibility aliases to resolve
	// into that advertised canonical set.
	if c.ClientModelIDs != nil {
		clientIDs := make(map[string]struct{}, len(c.ClientModelIDs))
		for _, modelID := range c.ClientModelIDs {
			if modelID == "" {
				return fmt.Errorf("client_model_ids must not contain an empty model ID")
			}
			if _, duplicate := clientIDs[modelID]; duplicate {
				return fmt.Errorf("client_model_ids contains duplicate model ID %q", modelID)
			}
			clientIDs[modelID] = struct{}{}
		}
		for label, aliases := range map[string]map[string]string{
			"public_model_alias":   c.PublicModelAlias,
			"accepted_model_alias": c.AcceptedModelAlias,
		} {
			for alias, target := range aliases {
				if _, exists := clientIDs[target]; !exists {
					return fmt.Errorf("%s %q targets %q outside client_model_ids", label, alias, target)
				}
				if _, collision := clientIDs[alias]; collision {
					return fmt.Errorf("%s %q collides with client_model_ids", label, alias)
				}
			}
		}
	}

	// Validate Redis config
	if c.Redis.Enabled {
		if len(c.Redis.InitAddresses) == 0 {
			return fmt.Errorf("redis.addresses is required when redis is enabled")
		}
	}

	// Validate OTEL config
	if c.OTEL.Enabled {
		if c.OTEL.Protocol != "grpc" && c.OTEL.Protocol != "http" {
			return fmt.Errorf("otel.protocol must be 'grpc' or 'http', got: %s", c.OTEL.Protocol)
		}
		if c.OTEL.Endpoint == "" {
			return fmt.Errorf("otel.endpoint is required when otel is enabled")
		}
		if c.OTEL.TraceSampleRatio < 0 || c.OTEL.TraceSampleRatio > 1 {
			return fmt.Errorf("otel.trace_sample_ratio must be in [0.0, 1.0], got: %f", c.OTEL.TraceSampleRatio)
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
		if c.LiteLLMDB.EnforceBudgetReservation && c.LiteLLMDB.BudgetReservationTTL <= 0 {
			return fmt.Errorf("litellm_db.budget_reservation_ttl must be positive when budget reservation is enabled")
		}
		if c.LiteLLMDB.EnforceBudgetReservation && c.LiteLLMDB.DefaultEstimatedCompletionTokens <= 0 {
			return fmt.Errorf("litellm_db.default_estimated_completion_tokens must be positive when budget reservation is enabled")
		}
	}

	if !c.SpendLog.IsEnabled() {
		// expected_database_name indicates that the writer was intentionally
		// configured and its environment-backed database URL failed to resolve.
		if c.SpendLog.ExpectedDatabaseName != "" {
			return fmt.Errorf("spend_log.database_url is required when spend_log is configured")
		}
	} else {
		if !strings.HasPrefix(c.SpendLog.DatabaseURL, "postgres://") && !strings.HasPrefix(c.SpendLog.DatabaseURL, "postgresql://") {
			return fmt.Errorf("spend_log.database_url must start with postgres:// or postgresql://")
		}
		if c.SpendLog.ExpectedDatabaseName == "" {
			return fmt.Errorf("spend_log.expected_database_name is required when spend_log is configured")
		}
		if c.SpendLog.APIBase != SpendAPIBase {
			return fmt.Errorf("spend_log.api_base must be %s", SpendAPIBase)
		}
		if c.SpendLog.MaxConns <= 0 || c.SpendLog.MinConns < 0 || c.SpendLog.MinConns > c.SpendLog.MaxConns {
			return fmt.Errorf("spend_log connection limits must satisfy max_conns > 0 and 0 <= min_conns <= max_conns")
		}
		if c.SpendLog.HealthCheckInterval <= 0 || c.SpendLog.ConnectTimeout <= 0 {
			return fmt.Errorf("spend_log health_check_interval and connect_timeout must be positive")
		}
		if c.SpendLog.LogQueueSize <= 0 || c.SpendLog.LogBatchSize <= 0 || c.SpendLog.LogFlushInterval <= 0 {
			return fmt.Errorf("spend_log queue size, batch size, and flush interval must be positive")
		}
	}

	if c.Kafka.Enabled {
		if len(c.Kafka.Brokers) == 0 {
			return fmt.Errorf("kafka.brokers is required when kafka is enabled")
		}
		if c.Kafka.Topic == "" {
			return fmt.Errorf("kafka.topic is required when kafka is enabled")
		}
		if c.Kafka.LogQueueSize <= 0 {
			return fmt.Errorf("kafka.log_queue_size must be positive, got: %d", c.Kafka.LogQueueSize)
		}
		if c.Kafka.LogBatchSize <= 0 {
			return fmt.Errorf("kafka.log_batch_size must be positive, got: %d", c.Kafka.LogBatchSize)
		}
		if c.Kafka.LogFlushInterval <= 0 {
			return fmt.Errorf("kafka.log_flush_interval must be positive, got: %s", c.Kafka.LogFlushInterval)
		}
		switch c.Kafka.SASLMechanism {
		case "", "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512":
		default:
			return fmt.Errorf("kafka.sasl_mechanism unsupported, got: %s", c.Kafka.SASLMechanism)
		}
		if c.Kafka.SASLMechanism != "" && (c.Kafka.SASLUsername == "" || c.Kafka.SASLPassword == "") {
			return fmt.Errorf("kafka.sasl_username and kafka.sasl_password are required when kafka.sasl_mechanism is set")
		}
	}

	if !c.Kafka.Enabled && c.LiteLLMDB.DisableSpendLogsWrite {
		return fmt.Errorf("invalid config: litellm_db.disable_spend_logs_write=true requires kafka.enabled=true, otherwise spend logs are lost entirely")
	}

	return nil
}
