package config

import (
	"fmt"
	"os"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// Validate checks configuration and applies defaults in order, returning the first error.
func (c *Config) Validate() error {
	if err := c.validateFeatures(); err != nil {
		return err
	}
	if err := c.Server.validate(); err != nil {
		return err
	}
	if err := c.validateAccess(); err != nil {
		return err
	}
	if err := c.validateCredentials(); err != nil {
		return err
	}
	if err := c.validateModels(); err != nil {
		return err
	}
	return c.validateIntegrations()
}

func (c *Config) validateFeatures() error {
	if err := c.Video.Validate(); err != nil {
		return err
	}
	if err := c.YandexSearch.Validate(); err != nil {
		return err
	}
	if err := c.VisionFallback.Validate(); err != nil {
		return err
	}
	if err := c.validateLoopGate(); err != nil {
		return err
	}
	if c.VisionFallback.Mode == VisionFallbackDescribe {
		for _, model := range c.Models {
			if model.Name == c.VisionFallback.DescribeModel && model.SupportsVision != nil && !*model.SupportsVision {
				return fmt.Errorf("vision_fallback.describe_model %q is declared supports_vision: false", model.Name)
			}
		}
	}
	return nil
}

func (c *Config) validateLoopGate() error {
	if err := c.LoopGate.Validate(); err != nil {
		return err
	}
	for _, cred := range c.Credentials {
		if err := cred.LoopGate.Validate("credential " + cred.Name + " loop_gate"); err != nil {
			return err
		}
	}
	for _, model := range c.Models {
		if err := model.LoopGate.Validate("model " + model.Name + " loop_gate"); err != nil {
			return err
		}
	}

	return c.validateLoopGateCombinations()
}

// validateLoopGateCombinations resolves the loop gate settings of every
// credential and model pair, so a credential or model override that is valid
// alone but not together with the inherited values (cut_min_ratio below the
// effective min_ratio) fails at startup instead of disabling the gate at
// request time. A model's loop_gate is looked up as at request time
// (models.Manager.GetLoopGateOverride): a model entry without a credential
// serves every credential, except one with its own entry for that model and a
// loop_gate of its own, which wins there.
func (c *Config) validateLoopGateCombinations() error {
	// model name -> credential ("" = every credential) -> the entry, the last
	// one winning as in the models manager.
	gates := make(map[string]map[string]ModelRPMConfig)
	var names []string
	for _, model := range c.Models {
		if model.LoopGate == nil {
			continue
		}
		if gates[model.Name] == nil {
			gates[model.Name] = make(map[string]ModelRPMConfig)
			names = append(names, model.Name)
		}
		gates[model.Name][model.Credential] = model
	}
	for _, cred := range c.Credentials {
		if _, err := c.LoopGate.Resolve(cred.Type, cred.LoopGate); err != nil {
			return fmt.Errorf("credential %s: %w", cred.Name, err)
		}
		for _, name := range names {
			model, ok := gates[name][cred.Name]
			if !ok {
				if model, ok = gates[name][""]; !ok {
					continue
				}
			}
			if _, err := c.LoopGate.Resolve(cred.Type, cred.LoopGate, model.LoopGate); err != nil {
				return fmt.Errorf("model %s on credential %s: %w", model.Name, cred.Name, err)
			}
		}
	}
	return nil
}

func (s *ServerConfig) validate() error {
	if s.Port <= 0 || s.Port > 65535 {
		return fmt.Errorf("invalid port: %d", s.Port)
	}

	if s.MaxBodySizeMB <= 0 {
		return fmt.Errorf("invalid max_body_size_mb: %d", s.MaxBodySizeMB)
	}

	if s.ResponseBodyMultiplier <= 0 {
		s.ResponseBodyMultiplier = 10
	}
	switch s.ResponseCompatibility {
	case "", "native":
		s.ResponseCompatibility = "native"
	case "litellm":
	default:
		return fmt.Errorf("invalid response_compatibility: %s (must be native or litellm)", s.ResponseCompatibility)
	}

	if s.ResponseHeaders.Mode == "" {
		s.ResponseHeaders.Mode = ResponseHeaderModePassthrough
	}
	if s.ResponseHeaders.Mode != ResponseHeaderModePassthrough && s.ResponseHeaders.Mode != ResponseHeaderModeAllowlist {
		return fmt.Errorf("response_headers.mode must be passthrough or allowlist, got %q", s.ResponseHeaders.Mode)
	}

	// -1 means unlimited timeout
	if s.RequestTimeout < 0 && s.RequestTimeout != -1 {
		return fmt.Errorf("invalid request_timeout: %v", s.RequestTimeout)
	}

	// A non-positive interval would make the ticker panic; fall back to the default.
	if s.ModelPricesSyncInterval <= 0 {
		s.ModelPricesSyncInterval = DefaultModelPricesSyncInterval
	}

	// A non-positive timeout would cut every in-flight request on stop; fall back to the default.
	if s.ShutdownTimeout <= 0 {
		s.ShutdownTimeout = DefaultShutdownTimeout
	}

	switch s.LoggingLevel {
	case "":
		s.LoggingLevel = "info"
	case "info", "debug", "error":
	default:
		return fmt.Errorf("invalid logging_level: %s (must be info, debug, or error)", s.LoggingLevel)
	}

	// Validate master_key
	if s.MasterKey == "" {
		return fmt.Errorf("master_key is required")
	}

	// Validate and normalize default_models_rpm
	// -1 means unlimited RPM, 0 is treated as unlimited
	if s.DefaultModelsRPM == 0 {
		s.DefaultModelsRPM = -1 // Convert 0 to unlimited (-1)
	} else if s.DefaultModelsRPM < -1 {
		return fmt.Errorf("invalid default_models_rpm: %d (must be -1 for unlimited or positive number)", s.DefaultModelsRPM)
	}

	// ReadTimeout equals RequestTimeout (not configurable via YAML)
	s.ReadTimeout = s.RequestTimeout

	// Validate AnthropicDefaultMaxTokens (0 = use default of 32000)
	if s.AnthropicDefaultMaxTokens < 0 {
		return fmt.Errorf("invalid anthropic_default_max_tokens: %d (must be >= 0)", s.AnthropicDefaultMaxTokens)
	}

	// Validate MaxProviderRetries
	if s.MaxProviderRetries < 0 {
		return fmt.Errorf("invalid max_provider_retries: %d (must be >= 0)", s.MaxProviderRetries)
	}

	// Validate MaxFallbackAttempts (0 = use default of 5)
	if s.MaxFallbackAttempts < 0 {
		return fmt.Errorf("invalid max_fallback_attempts: %d (must be >= 0)", s.MaxFallbackAttempts)
	}
	if s.MaxConnectionLifetime < 0 {
		return fmt.Errorf("invalid max_connection_lifetime: %s (must be >= 0)", s.MaxConnectionLifetime)
	}
	if s.MaxInFlightRequests < 0 {
		return fmt.Errorf("invalid max_in_flight_requests: %d (must be >= 0, 0 = no cap)", s.MaxInFlightRequests)
	}
	if s.SessionStickyTTL < 0 {
		return fmt.Errorf("invalid session_sticky_ttl_minutes: %d (must be >= 0)", s.SessionStickyTTL)
	}

	// Validate IdleTimeout against WriteTimeout
	if s.IdleTimeout == 0 {
		s.IdleTimeout = s.WriteTimeout * 2
	}
	return nil
}

func (c *Config) validateAccess() error {
	if c.Fail2Ban.MaxAttempts <= 0 {
		return fmt.Errorf("invalid max_attempts: %d", c.Fail2Ban.MaxAttempts)
	}

	if len(c.Credentials) == 0 && !c.LiteLLMDB.Enabled {
		return fmt.Errorf("no credentials configured")
	}

	if err := ValidateOrganizationPolicies(c.OrganizationPolicies); err != nil {
		return err
	}
	return c.Fail2Ban.validateErrorCodeRules()
}

func (c *Fail2BanConfig) validateErrorCodeRules() error {
	if code, duplicate := duplicateErrorCode(c.ErrorCodeRules); duplicate {
		return fmt.Errorf("fail2ban: duplicate error_code_rules for code %d", code)
	}
	for name, override := range c.CredentialOverrides {
		if code, duplicate := duplicateErrorCode(override.ErrorCodeRules); duplicate {
			return fmt.Errorf("fail2ban: duplicate error_code_rules for code %d in credential_overrides[%s]", code, name)
		}
	}
	return nil
}

func duplicateErrorCode(rules []ErrorCodeRuleConfig) (int, bool) {
	seen := make(map[int]struct{}, len(rules))
	for _, rule := range rules {
		if _, duplicate := seen[rule.Code]; duplicate {
			return rule.Code, true
		}
		seen[rule.Code] = struct{}{}
	}
	return 0, false
}

func (c *Config) validateCredentials() error {
	routingIDs := make(map[string]string)
	for i := range c.Credentials {
		cred := &c.Credentials[i]
		if err := cred.validate(i); err != nil {
			return err
		}
		if cred.RoutingID != "" {
			if previous, ok := routingIDs[cred.RoutingID]; ok {
				return fmt.Errorf("credential %s: routing_id is already assigned to credential %s", cred.Name, previous)
			}
			routingIDs[cred.RoutingID] = cred.Name
		}
		// Share fallback priority normalization with DB credentials.
		cred.NormalizeFallbackPriority()
	}
	return nil
}

func (cred *CredentialConfig) validate(i int) error {
	if cred.Name == "" {
		return fmt.Errorf("credential %d: name is required", i)
	}

	if _, err := ParseProxyURL(cred.ProxyURL); err != nil {
		return fmt.Errorf("credential %s: %w", cred.Name, err)
	}

	// Validate provider type
	if !cred.Type.IsValid() {
		return fmt.Errorf("credential %s: invalid type: %s (must be 'openai', 'vertex-ai', 'gemini', 'anthropic', 'cometapi', 'proman', 'bedrock', 'proxy', 'air', or 'vllm')", cred.Name, cred.Type)
	}
	if cred.RoutingID != "" {
		if !scope.ValidCredentialRoutingID(cred.RoutingID) {
			return fmt.Errorf("credential %s: routing_id must be route- followed by 1 to 12 digits", cred.Name)
		}
		if cred.IsProxyLike() {
			return fmt.Errorf("credential %s: routing_id requires a direct credential", cred.Name)
		}
	}
	if cred.AuthType != "" && cred.AuthType != "bearer" && cred.AuthType != "x-api-key" {
		return fmt.Errorf("credential %s: invalid auth_type: %s (must be 'bearer' or 'x-api-key')", cred.Name, cred.AuthType)
	}
	if cred.OpenAIProtocol && cred.Type != ProviderTypeCometAPI {
		return fmt.Errorf("credential %s: openai_proto is only supported for 'cometapi' type, got: %s", cred.Name, cred.Type)
	}
	if cred.GoogleProtocol && cred.Type != ProviderTypeCometAPI {
		return fmt.Errorf("credential %s: google_proto is only supported for 'cometapi' type, got: %s", cred.Name, cred.Type)
	}
	if cred.OpenAIProtocol && cred.GoogleProtocol {
		return fmt.Errorf("credential %s: openai_proto and google_proto are mutually exclusive", cred.Name)
	}
	if cred.SystemOne && cred.EffectiveProviderType() != ProviderTypeOpenAI {
		return fmt.Errorf("credential %s: system_one needs an OpenAI-wire credential (openai, vllm), got: %s", cred.Name, cred.Type)
	}
	if err := validateCredentialRequestHeaders(cred.Name, cred.Type, cred.RequestHeaders); err != nil {
		return err
	}
	if err := cred.validateProviderSettings(); err != nil {
		return err
	}

	return cred.validateLimits()
}

func (cred *CredentialConfig) validateProviderSettings() error {
	switch cred.Type {
	case ProviderTypeVertexAI:
		return cred.validateVertexSettings()
	case ProviderTypeProxy, ProviderTypeAIR, ProviderTypeVLLM:
		// Remote routers and vLLM can run without an API key.
	case ProviderTypeGemini:
		if cred.APIKey == "" {
			return fmt.Errorf("credential %s: api_key is required for gemini type", cred.Name)
		}
	default:
		if cred.APIKey == "" {
			return fmt.Errorf("credential %s: api_key is required", cred.Name)
		}
	}

	if cred.BaseURL == "" {
		switch cred.Type {
		case ProviderTypeProxy, ProviderTypeAIR, ProviderTypeVLLM, ProviderTypeGemini:
			return fmt.Errorf("credential %s: base_url is required for %s type", cred.Name, cred.Type)
		default:
			return fmt.Errorf("credential %s: base_url is required", cred.Name)
		}
	}
	return validateBaseURL(cred.Name, cred.BaseURL)
}

func (cred *CredentialConfig) validateVertexSettings() error {
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
	return nil
}

func (cred *CredentialConfig) validateLimits() error {
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
	if cred.Priority < 0 {
		return fmt.Errorf("credential %s: invalid priority: %d (must be >= 0)", cred.Name, cred.Priority)
	}
	if cred.Priority > 0 && cred.FallbackPriority > 0 {
		return fmt.Errorf("credential %s: invalid priority: cannot set both priority and fallback_priority", cred.Name)
	}
	if cred.IsFallback && cred.Priority > 0 && cred.Priority != FallbackPriorityGroup {
		return fmt.Errorf("credential %s: invalid priority: fallback credentials cannot set priority (always tried last, pinned to group %d)", cred.Name, FallbackPriorityGroup)
	}
	return nil
}

func (c *Config) validateModels() error {
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
	return nil
}

func (c *Config) validateIntegrations() error {
	if c.Redis.Enabled && len(c.Redis.InitAddresses) == 0 {
		return fmt.Errorf("redis.addresses is required when redis is enabled")
	}
	if err := c.OTEL.validate(); err != nil {
		return err
	}
	if err := c.LiteLLMDB.validate(); err != nil {
		return err
	}
	if err := c.Kafka.validate(); err != nil {
		return err
	}
	if err := c.Kafka.validateAuxiliaryTopics(); err != nil {
		return err
	}

	if !c.Kafka.Enabled && c.LiteLLMDB.DisableSpendLogsWrite {
		return fmt.Errorf("invalid config: litellm_db.disable_spend_logs_write=true requires kafka.enabled=true, otherwise spend logs are lost entirely")
	}
	return nil
}

func (c *OTELConfig) validate() error {
	if !c.Enabled {
		return nil
	}
	if c.Protocol != "grpc" && c.Protocol != "http" {
		return fmt.Errorf("otel.protocol must be 'grpc' or 'http', got: %s", c.Protocol)
	}
	if c.Endpoint == "" {
		return fmt.Errorf("otel.endpoint is required when otel is enabled")
	}
	if c.TraceSampleRatio < 0 || c.TraceSampleRatio > 1 {
		return fmt.Errorf("otel.trace_sample_ratio must be in [0.0, 1.0], got: %f", c.TraceSampleRatio)
	}
	return nil
}

func (c *LiteLLMDBConfig) validate() error {
	if !c.Enabled {
		return nil
	}
	if c.DatabaseURL == "" {
		return fmt.Errorf("litellm_db.database_url is required when enabled")
	}
	if !strings.HasPrefix(c.DatabaseURL, "postgres://") && !strings.HasPrefix(c.DatabaseURL, "postgresql://") {
		return fmt.Errorf("litellm_db.database_url must start with postgres:// or postgresql://, got: %s", c.DatabaseURL)
	}
	if c.EnforceBudgetReservation && c.BudgetReservationTTL <= 0 {
		return fmt.Errorf("litellm_db.budget_reservation_ttl must be positive when budget reservation is enabled")
	}
	if c.EnforceBudgetReservation && c.DefaultEstimatedCompletionTokens <= 0 {
		return fmt.Errorf("litellm_db.default_estimated_completion_tokens must be positive when budget reservation is enabled")
	}
	if c.LogWorkers <= 0 {
		return fmt.Errorf("litellm_db.log_workers must be positive, got: %d", c.LogWorkers)
	}
	return nil
}

func (c *KafkaConfig) validate() error {
	if !c.Enabled {
		return nil
	}
	if len(c.Brokers) == 0 {
		return fmt.Errorf("kafka.brokers is required when kafka is enabled")
	}
	if c.Topic == "" {
		return fmt.Errorf("kafka.topic is required when kafka is enabled")
	}
	if c.LogQueueSize <= 0 {
		return fmt.Errorf("kafka.log_queue_size must be positive, got: %d", c.LogQueueSize)
	}
	if c.LogBatchSize <= 0 {
		return fmt.Errorf("kafka.log_batch_size must be positive, got: %d", c.LogBatchSize)
	}
	if c.LogFlushInterval <= 0 {
		return fmt.Errorf("kafka.log_flush_interval must be positive, got: %s", c.LogFlushInterval)
	}
	if c.LogWorkers <= 0 {
		return fmt.Errorf("kafka.log_workers must be positive, got: %d", c.LogWorkers)
	}
	switch c.SASLMechanism {
	case "", "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512":
	default:
		return fmt.Errorf("kafka.sasl_mechanism unsupported, got: %s", c.SASLMechanism)
	}
	if c.SASLMechanism != "" && (c.SASLUsername == "" || c.SASLPassword == "") {
		return fmt.Errorf("kafka.sasl_username and kafka.sasl_password are required when kafka.sasl_mechanism is set")
	}
	return nil
}

func (c *KafkaConfig) validateAuxiliaryTopics() error {
	if c.RawBodies.Enabled {
		if !c.Enabled {
			return fmt.Errorf("kafka.raw_bodies.enabled requires kafka.enabled=true")
		}
		if c.RawBodies.Topic == "" {
			return fmt.Errorf("kafka.raw_bodies.topic is required when kafka.raw_bodies is enabled")
		}
		if c.RawBodies.Topic == c.Topic {
			return fmt.Errorf("kafka.raw_bodies.topic must differ from kafka.topic")
		}
	}

	if c.RouterResponses.Enabled {
		if !c.Enabled {
			return fmt.Errorf("kafka.router_responses.enabled requires kafka.enabled=true")
		}
		if c.RouterResponses.Topic == "" {
			return fmt.Errorf("kafka.router_responses.topic is required when kafka.router_responses is enabled")
		}
		// Its own topic, distinct from the spend-log topic and from the
		// raw-bodies topic: the three carry different streams with different
		// retention, and a shared topic would mix a response body into the
		// spend log (which is billed on and kept for years).
		if c.RouterResponses.Topic == c.Topic {
			return fmt.Errorf("kafka.router_responses.topic must differ from kafka.topic")
		}
		if c.RouterResponses.Topic == c.RawBodies.Topic {
			return fmt.Errorf("kafka.router_responses.topic must differ from kafka.raw_bodies.topic")
		}
	}
	return nil
}
