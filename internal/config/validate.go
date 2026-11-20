package config

import (
	"fmt"
	"strings"
)

// Validate checks every section of the configuration and applies the
// defaults that depend on other fields (timeouts derived from one another,
// normalization of zero values). It returns the first error it finds, in
// the order the sections are listed below.
//
// The checks are a list rather than a hand-written sequence so that the
// order stays explicit and the driver stays trivial. Do not reorder entries:
// which error a broken config reports first is part of the contract.
func (c *Config) Validate() error {
	checks := []func() error{
		c.validateDelegated,
		c.validateServerEndpoint,
		c.validateServerResponseHeaders,
		c.validateServerTimeouts,
		c.validateServerLogging,
		c.validateServerAuth,
		c.validateServerLimits,
		c.validateFail2BanLimits,
		c.validateTopology,
		c.validateFail2BanRules,
		c.validateCredentials,
		c.validateModels,
		c.validateClientModelIDs,
		c.validateRedis,
		c.validateOTEL,
		c.validateLiteLLMDB,
		c.validateKafka,
		c.validateKafkaRawBodies,
		c.validateKafkaRouterResponses,
		c.validateSpendLogConsistency,
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func (c *Config) validateDelegated() error {
	if err := c.Video.Validate(); err != nil {
		return err
	}
	if err := c.YandexSearch.Validate(); err != nil {
		return err
	}
	if err := c.VisionFallback.Validate(); err != nil {
		return err
	}
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
	if err := c.validateLoopGateCombinations(); err != nil {
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

func (c *Config) validateServerEndpoint() error {
	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Server.Port)
	}

	if c.Server.MaxBodySizeMB <= 0 {
		return fmt.Errorf("invalid max_body_size_mb: %d", c.Server.MaxBodySizeMB)
	}

	if c.Server.ResponseBodyMultiplier <= 0 {
		c.Server.ResponseBodyMultiplier = 10
	}
	switch c.Server.ResponseCompatibility {
	case "", "native":
		c.Server.ResponseCompatibility = "native"
	case "litellm":
	default:
		return fmt.Errorf("invalid response_compatibility: %s (must be native or litellm)", c.Server.ResponseCompatibility)
	}

	return nil
}

func (c *Config) validateServerResponseHeaders() error {
	if c.Server.ResponseHeaders.Mode == "" {
		c.Server.ResponseHeaders.Mode = ResponseHeaderModePassthrough
	}
	if c.Server.ResponseHeaders.Mode != ResponseHeaderModePassthrough && c.Server.ResponseHeaders.Mode != ResponseHeaderModeAllowlist {
		return fmt.Errorf("response_headers.mode must be passthrough or allowlist, got %q", c.Server.ResponseHeaders.Mode)
	}

	return nil
}

func (c *Config) validateServerTimeouts() error {
	// -1 means unlimited timeout
	if c.Server.RequestTimeout < 0 && c.Server.RequestTimeout != -1 {
		return fmt.Errorf("invalid request_timeout: %v", c.Server.RequestTimeout)
	}

	// A non-positive interval would make the ticker panic; fall back to the default.
	if c.Server.ModelPricesSyncInterval <= 0 {
		c.Server.ModelPricesSyncInterval = DefaultModelPricesSyncInterval
	}

	// A non-positive timeout would cut every in-flight request on stop; fall back to the default.
	if c.Server.ShutdownTimeout <= 0 {
		c.Server.ShutdownTimeout = DefaultShutdownTimeout
	}

	return nil
}

func (c *Config) validateServerLogging() error {
	// Validate logging level
	if c.Server.LoggingLevel != "" {
		validLevels := map[string]bool{"info": true, "debug": true, "error": true}
		if !validLevels[c.Server.LoggingLevel] {
			return fmt.Errorf("invalid logging_level: %s (must be info, debug, or error)", c.Server.LoggingLevel)
		}
	} else {
		c.Server.LoggingLevel = "info" // Default to info
	}

	return nil
}

func (c *Config) validateServerAuth() error {
	// Validate master_key
	if c.Server.MasterKey == "" {
		return fmt.Errorf("master_key is required")
	}

	return nil
}

func (c *Config) validateServerLimits() error {
	// Validate and normalize default_models_rpm
	// -1 means unlimited RPM, 0 is treated as unlimited
	if c.Server.DefaultModelsRPM == 0 {
		c.Server.DefaultModelsRPM = -1 // Convert 0 to unlimited (-1)
	} else if c.Server.DefaultModelsRPM < -1 {
		return fmt.Errorf("invalid default_models_rpm: %d (must be -1 for unlimited or positive number)", c.Server.DefaultModelsRPM)
	}

	// ReadTimeout equals RequestTimeout (not configurable via YAML)
	c.Server.ReadTimeout = c.Server.RequestTimeout

	// Validate AnthropicDefaultMaxTokens (0 = use default of 32000)
	if c.Server.AnthropicDefaultMaxTokens < 0 {
		return fmt.Errorf("invalid anthropic_default_max_tokens: %d (must be >= 0)", c.Server.AnthropicDefaultMaxTokens)
	}

	// Validate MaxProviderRetries
	if c.Server.MaxProviderRetries < 0 {
		return fmt.Errorf("invalid max_provider_retries: %d (must be >= 0)", c.Server.MaxProviderRetries)
	}

	// Validate MaxFallbackAttempts (0 = use default of 5)
	if c.Server.MaxFallbackAttempts < 0 {
		return fmt.Errorf("invalid max_fallback_attempts: %d (must be >= 0)", c.Server.MaxFallbackAttempts)
	}
	if c.Server.MaxConnectionLifetime < 0 {
		return fmt.Errorf("invalid max_connection_lifetime: %s (must be >= 0)", c.Server.MaxConnectionLifetime)
	}
	if c.Server.SessionStickyTTL < 0 {
		return fmt.Errorf("invalid session_sticky_ttl_minutes: %d (must be >= 0)", c.Server.SessionStickyTTL)
	}

	// Validate IdleTimeout against WriteTimeout
	if c.Server.IdleTimeout == 0 {
		c.Server.IdleTimeout = c.Server.WriteTimeout * 2
	}

	return nil
}

func (c *Config) validateFail2BanLimits() error {
	if c.Fail2Ban.MaxAttempts <= 0 {
		return fmt.Errorf("invalid max_attempts: %d", c.Fail2Ban.MaxAttempts)
	}

	return nil
}

func (c *Config) validateTopology() error {
	if len(c.Credentials) == 0 && !c.LiteLLMDB.Enabled {
		return fmt.Errorf("no credentials configured")
	}

	if err := ValidateOrganizationPolicies(c.OrganizationPolicies); err != nil {
		return err
	}

	return nil
}

func (c *Config) validateFail2BanRules() error {
	// Validate Fail2Ban error code rules for duplicates
	seenErrorCodes := make(map[int]bool)
	for _, rule := range c.Fail2Ban.ErrorCodeRules {
		if seenErrorCodes[rule.Code] {
			return fmt.Errorf("fail2ban: duplicate error_code_rules for code %d", rule.Code)
		}
		seenErrorCodes[rule.Code] = true
	}

	// Validate each credential's error_code_rules override for duplicates
	for credName, override := range c.Fail2Ban.CredentialOverrides {
		seen := make(map[int]bool)
		for _, rule := range override.ErrorCodeRules {
			if seen[rule.Code] {
				return fmt.Errorf("fail2ban: duplicate error_code_rules for code %d in credential_overrides[%s]", rule.Code, credName)
			}
			seen[rule.Code] = true
		}
	}

	return nil
}

func (c *Config) validateModels() error {
	for _, model := range c.Models {
		if model.Weight < 0 {
			return fmt.Errorf("model %s: invalid weight: %d (must be 0 for default or positive number)", model.Name, model.Weight)
		}
	}

	return nil
}

func (c *Config) validateClientModelIDs() error {
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

func (c *Config) validateRedis() error {
	// Validate Redis config
	if c.Redis.Enabled {
		if len(c.Redis.InitAddresses) == 0 {
			return fmt.Errorf("redis.addresses is required when redis is enabled")
		}
	}

	return nil
}

func (c *Config) validateOTEL() error {
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

	return nil
}

func (c *Config) validateLiteLLMDB() error {
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
		if c.LiteLLMDB.LogWorkers <= 0 {
			return fmt.Errorf("litellm_db.log_workers must be positive, got: %d", c.LiteLLMDB.LogWorkers)
		}
	}

	return nil
}

func (c *Config) validateKafka() error {
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
		if c.Kafka.LogWorkers <= 0 {
			return fmt.Errorf("kafka.log_workers must be positive, got: %d", c.Kafka.LogWorkers)
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

	return nil
}

func (c *Config) validateKafkaRawBodies() error {
	if c.Kafka.RawBodies.Enabled {
		if !c.Kafka.Enabled {
			return fmt.Errorf("kafka.raw_bodies.enabled requires kafka.enabled=true")
		}
		if c.Kafka.RawBodies.Topic == "" {
			return fmt.Errorf("kafka.raw_bodies.topic is required when kafka.raw_bodies is enabled")
		}
		if c.Kafka.RawBodies.Topic == c.Kafka.Topic {
			return fmt.Errorf("kafka.raw_bodies.topic must differ from kafka.topic")
		}
	}

	return nil
}

func (c *Config) validateKafkaRouterResponses() error {
	if c.Kafka.RouterResponses.Enabled {
		if !c.Kafka.Enabled {
			return fmt.Errorf("kafka.router_responses.enabled requires kafka.enabled=true")
		}
		if c.Kafka.RouterResponses.Topic == "" {
			return fmt.Errorf("kafka.router_responses.topic is required when kafka.router_responses is enabled")
		}
		// Its own topic, distinct from the spend-log topic and from the
		// raw-bodies topic: the three carry different streams with different
		// retention, and a shared topic would mix a response body into the
		// spend log (which is billed on and kept for years).
		if c.Kafka.RouterResponses.Topic == c.Kafka.Topic {
			return fmt.Errorf("kafka.router_responses.topic must differ from kafka.topic")
		}
		if c.Kafka.RouterResponses.Topic == c.Kafka.RawBodies.Topic {
			return fmt.Errorf("kafka.router_responses.topic must differ from kafka.raw_bodies.topic")
		}
	}

	return nil
}

func (c *Config) validateSpendLogConsistency() error {
	if !c.Kafka.Enabled && c.LiteLLMDB.DisableSpendLogsWrite {
		return fmt.Errorf("invalid config: litellm_db.disable_spend_logs_write=true requires kafka.enabled=true, otherwise spend logs are lost entirely")
	}

	return nil
}

// validateCredentials validates every credential and normalizes the
// fallback priority of each one. Normalization happens only after that
// credential passed validation, exactly as it did inline before.
func (c *Config) validateCredentials() error {
	for i := range c.Credentials {
		cred := &c.Credentials[i]
		if err := cred.validate(i); err != nil {
			return err
		}
		cred.NormalizeFallbackPriority()
	}
	return nil
}
