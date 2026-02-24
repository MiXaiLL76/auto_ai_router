package config

import (
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strings"
	"time"
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
		"master_key", "***REDACTED***",
		"default_models_rpm", rpmToString(cfg.Server.DefaultModelsRPM),
		"max_idle_conns", cfg.Server.MaxIdleConns,
		"max_idle_conns_per_host", cfg.Server.MaxIdleConnsPerHost,
		"idle_conn_timeout", cfg.Server.IdleConnTimeout.String(),
		"model_prices_link", cfg.Server.ModelPricesLink,
		"max_provider_retries", cfg.Server.MaxProviderRetries,
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
		logger.Info(fmt.Sprintf("  [%d] credential", i),
			"name", cred.Name,
			"type", cred.Type,
			"rpm", rpmToString(cred.RPM),
			"tpm", tpmToString(cred.TPM),
			"is_fallback", cred.IsFallback,
		)
	}

	// Models
	logger.Info("models",
		"total_count", len(cfg.Models),
	)
	if len(cfg.Models) > 0 && len(cfg.Models) <= 10 {
		// Only show details if there are a few models
		for i, model := range cfg.Models {
			logger.Info(fmt.Sprintf("  [%d] model", i),
				"name", model.Name,
				"credential", model.Credential,
				"rpm", model.RPM,
				"tpm", model.TPM,
			)
		}
	}

	// LiteLLM DB config
	if cfg.LiteLLMDB.Enabled {
		logger.Info("litellm_db (ENABLED)",
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
		)
	} else {
		logger.Info("litellm_db", "status", "DISABLED")
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
