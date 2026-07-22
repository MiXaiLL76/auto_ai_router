package config

import (
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/security"
)

// resolveEnvString resolves environment variable if value is in format "os.environ/VAR_NAME"
func resolveEnvString(value string) string {
	const prefix = "os.environ/"
	if strings.HasPrefix(value, prefix) {
		envVar := strings.TrimPrefix(value, prefix)
		if envValue := os.Getenv(envVar); envValue != "" {
			return envValue
		}
		slog.Warn("environment variable not set, returning empty string",
			"env_var", envVar,
			"pattern", value,
		)
		return ""
	}
	return value
}

// parseFunc is a function type that parses a string value into the desired type
type parseFunc[T any] func(string) (T, error)

// parseField resolves env variable and parses value with proper error context
func parseField[T any](tempValue string, defaultValue T, parser parseFunc[T], fieldPath string) (T, error) {
	if tempValue == "" {
		return defaultValue, nil
	}

	resolved := resolveEnvString(tempValue)
	if resolved == "" {
		return defaultValue, nil
	}

	parsed, err := parser(resolved)
	if err != nil {
		return defaultValue, fmt.Errorf("invalid %s: %w", fieldPath, err)
	}
	return parsed, nil
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

// isUnlimited checks if a value represents unlimited (-1)
func isUnlimited(value int) bool {
	return value == -1
}

// PrintConfig outputs the configuration in a structured, readable format to the logger
func PrintConfig(logger *slog.Logger, cfg *Config) {
	logger.Info("=== Configuration Loaded ===")

	// Server config
	logger.Info("server",
		"port", cfg.Server.Port,
		"max_body_size_mb", cfg.Server.MaxBodySizeMB,
		"response_body_multiplier", cfg.Server.ResponseBodyMultiplier,
		"request_timeout", cfg.Server.RequestTimeout.String(),
		"read_timeout", cfg.Server.ReadTimeout.String(),
		"write_timeout", cfg.Server.WriteTimeout.String(),
		"idle_timeout", cfg.Server.IdleTimeout.String(),
		"logging_level", cfg.Server.LoggingLevel,
		"stdout_logs_enabled", cfg.Server.StdoutLogsEnabled,
		"master_key", "***REDACTED***",
		"default_models_rpm", rpmToString(cfg.Server.DefaultModelsRPM),
		"max_idle_conns", cfg.Server.MaxIdleConns,
		"max_idle_conns_per_host", cfg.Server.MaxIdleConnsPerHost,
		"idle_conn_timeout", cfg.Server.IdleConnTimeout.String(),
		"model_prices_link", cfg.Server.ModelPricesLink,
		"max_provider_retries", cfg.Server.MaxProviderRetries,
		"max_fallback_attempts", cfg.Server.MaxFallbackAttempts,
		"session_sticky_enabled", cfg.Server.SessionStickyEnabled,
		"session_sticky_ttl_minutes", cfg.Server.SessionStickyTTL,
	)

	// Monitoring config
	logger.Info("monitoring",
		"prometheus_enabled", cfg.Monitoring.PrometheusEnabled,
		"health_check_path", cfg.Monitoring.HealthCheckPath,
		"log_errors", cfg.Monitoring.LogErrors,
		"errors_log_path", cfg.Monitoring.ErrorsLogPath,
	)

	// Fail2Ban config
	logger.Info("fail2ban",
		"max_attempts", cfg.Fail2Ban.MaxAttempts,
		"ban_duration", banDurationToString(cfg.Fail2Ban.BanDuration),
		"error_codes_count", len(cfg.Fail2Ban.ErrorCodes),
		"error_code_rules_count", len(cfg.Fail2Ban.ErrorCodeRules),
	)

	// Credentials
	logger.Info("credentials",
		"total_count", len(cfg.Credentials),
	)
	for i, cred := range cfg.Credentials {
		credLog := map[string]any{
			"name":              cred.Name,
			"type":              cred.Type,
			"base_url":          cred.BaseURL,
			"auth_type":         cred.AuthType,
			"rpm":               rpmToString(cred.RPM),
			"tpm":               tpmToString(cred.TPM),
			"is_fallback":       cred.IsFallback,
			"fallback_priority": cred.FallbackPriority,
		}

		// Add Vertex AI specific fields if present
		if cred.Type == ProviderTypeVertexAI {
			credLog["project_id"] = cred.ProjectID
			credLog["location"] = cred.Location
		}

		logger.Info(fmt.Sprintf("  [%d] credential", i), convertMapToArgs(credLog)...)
	}

	// Models
	logger.Info("models",
		"total_count", len(cfg.Models),
	)
	if len(cfg.Models) > 0 && len(cfg.Models) <= 10 {
		// Only show details if there are a few models
		for i, model := range cfg.Models {
			// Treat 0 as unlimited for display (same semantics as rate limiter)
			rpm := model.RPM
			if rpm == 0 {
				rpm = -1
			}
			tpm := model.TPM
			if tpm == 0 {
				tpm = -1
			}
			logger.Info(fmt.Sprintf("  [%d] model", i),
				"name", model.Name,
				"credential", model.Credential,
				"rpm", rpmToString(rpm),
				"tpm", tpmToString(tpm),
			)
		}
	}

	// Model aliases
	if cfg.ClientModelIDs != nil {
		logger.Info("client_model_ids", "total_count", len(cfg.ClientModelIDs), "enforced", true)
	}
	if len(cfg.ModelAlias) > 0 {
		logger.Info("model_alias", "total_count", len(cfg.ModelAlias))
		for alias, target := range cfg.ModelAlias {
			logger.Info("  alias", "from", alias, "to", target)
		}
	}
	if len(cfg.PublicModelAlias) > 0 {
		logger.Info("public_model_alias", "total_count", len(cfg.PublicModelAlias))
		for alias, target := range cfg.PublicModelAlias {
			logger.Info("  public alias", "from", alias, "to", target)
		}
	}
	if len(cfg.AcceptedModelAlias) > 0 {
		logger.Info("accepted_model_alias", "total_count", len(cfg.AcceptedModelAlias))
		for alias, target := range cfg.AcceptedModelAlias {
			logger.Info("  accepted alias", "from", alias, "to", target)
		}
	}

	// LiteLLM DB config
	if cfg.LiteLLMDB.Enabled {
		logger.Info("litellm_db (ENABLED)",
			"database_url", security.MaskDatabaseURL(cfg.LiteLLMDB.DatabaseURL),
			"is_required", cfg.LiteLLMDB.IsRequired,
			"max_conns", cfg.LiteLLMDB.MaxConns,
			"min_conns", cfg.LiteLLMDB.MinConns,
			"health_check_interval", cfg.LiteLLMDB.HealthCheckInterval.String(),
			"connect_timeout", cfg.LiteLLMDB.ConnectTimeout.String(),
			"auth_cache_ttl", cfg.LiteLLMDB.AuthCacheTTL.String(),
			"auth_cache_size", cfg.LiteLLMDB.AuthCacheSize,
			"log_queue_size", cfg.LiteLLMDB.LogQueueSize,
			"log_batch_size", cfg.LiteLLMDB.LogBatchSize,
			"log_flush_interval", cfg.LiteLLMDB.LogFlushInterval.String(),
			"disable_spend_logs_write", cfg.LiteLLMDB.DisableSpendLogsWrite,
			"enforce_budget_reservation", cfg.LiteLLMDB.EnforceBudgetReservation,
			"budget_reservation_ttl", cfg.LiteLLMDB.BudgetReservationTTL.String(),
			"enforce_key_rate_limits", cfg.LiteLLMDB.EnforceKeyRateLimits,
			"default_estimated_completion_tokens", cfg.LiteLLMDB.DefaultEstimatedCompletionTokens,
		)
	} else {
		logger.Info("litellm_db", "status", "DISABLED")
	}

	if cfg.SpendLog.IsEnabled() {
		logger.Info("spend_log (ENABLED)",
			"database_url", security.MaskDatabaseURL(cfg.SpendLog.DatabaseURL),
			"expected_database_name", cfg.SpendLog.ExpectedDatabaseName,
			"api_base", cfg.SpendLog.APIBase,
			"max_conns", cfg.SpendLog.MaxConns,
			"min_conns", cfg.SpendLog.MinConns,
			"log_queue_size", cfg.SpendLog.LogQueueSize,
			"log_batch_size", cfg.SpendLog.LogBatchSize,
			"log_flush_interval", cfg.SpendLog.LogFlushInterval.String(),
		)
	} else {
		logger.Info("spend_log", "status", "DISABLED")
	}

	// Kafka spend-log config
	if cfg.Kafka.Enabled {
		saslPassword := ""
		if cfg.Kafka.SASLPassword != "" {
			saslPassword = "***REDACTED***"
		}
		logger.Info("kafka (ENABLED)",
			"brokers", cfg.Kafka.Brokers,
			"topic", cfg.Kafka.Topic,
			"client_id", cfg.Kafka.ClientID,
			"log_queue_size", cfg.Kafka.LogQueueSize,
			"log_batch_size", cfg.Kafka.LogBatchSize,
			"log_flush_interval", cfg.Kafka.LogFlushInterval.String(),
			"tls_enabled", cfg.Kafka.TLSEnabled,
			"sasl_mechanism", cfg.Kafka.SASLMechanism,
			"sasl_username", cfg.Kafka.SASLUsername,
			"sasl_password", saslPassword,
		)
	} else {
		logger.Info("kafka", "status", "DISABLED")
	}

	// OTEL config
	if cfg.OTEL.Enabled {
		logger.Info("otel (ENABLED)",
			"endpoint", cfg.OTEL.Endpoint,
			"protocol", cfg.OTEL.Protocol,
			"insecure", cfg.OTEL.Insecure,
			"service_name", cfg.OTEL.ServiceName,
			"logs_enabled", cfg.OTEL.LogsEnabled,
			"traces_enabled", cfg.OTEL.TracesEnabled,
			"trace_sample_ratio", cfg.OTEL.TraceSampleRatio,
		)
	} else {
		logger.Info("otel", "status", "DISABLED")
	}

	logger.Info("=== Configuration Ready ===")
}

// rpmToString converts RPM value to string, showing "unlimited" for -1
func rpmToString(rpm int) string {
	if rpm == -1 {
		return "unlimited (-1)"
	}
	return fmt.Sprintf("%d", rpm)
}

// tpmToString converts TPM value to string, showing "unlimited" for -1
func tpmToString(tpm int) string {
	if tpm == -1 {
		return "unlimited (-1)"
	}
	return fmt.Sprintf("%d", tpm)
}

// banDurationToString converts ban duration to readable string
func banDurationToString(d time.Duration) string {
	if d == 0 {
		return "permanent"
	}
	return d.String()
}

// convertMapToArgs converts a map[string]any to []any for logger.Info
// Maintains consistent key ordering for deterministic output
func convertMapToArgs(m map[string]any) []any {
	// Define preferred order of keys
	keyOrder := []string{
		"name", "type", "base_url", "auth_type", "api_key", "project_id", "location",
		"credentials_file", "credentials_json", "rpm", "tpm", "is_fallback",
	}

	args := make([]any, 0, len(m)*2)

	// Add keys in preferred order
	for _, key := range keyOrder {
		if val, exists := m[key]; exists {
			args = append(args, key, val)
		}
	}

	// Add any remaining keys not in preferred order
	addedKeys := make(map[string]bool)
	for _, key := range keyOrder {
		addedKeys[key] = true
	}
	for key, val := range m {
		if !addedKeys[key] {
			args = append(args, key, val)
		}
	}

	return args
}
