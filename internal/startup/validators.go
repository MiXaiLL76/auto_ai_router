package startup

import (
	"context"
	"log/slog"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

// ValidateProxyCredentialsAtStartup performs connectivity checks for proxy credentials at startup.
// For each proxy credential, it attempts an HTTP GET to the /health endpoint with a 5-second timeout.
// Results are logged as WARN if unreachable, but startup continues (non-blocking).
// This helps catch misconfigured proxies early without failing the startup sequence.
func ValidateProxyCredentialsAtStartup(cfg *config.Config, log *slog.Logger) {
	proxyCredentials := make([]config.CredentialConfig, 0)
	for _, cred := range cfg.Credentials {
		if cred.Type == config.ProviderTypeProxy {
			proxyCredentials = append(proxyCredentials, cred)
		}
	}

	if len(proxyCredentials) == 0 {
		return
	}

	log.Info("Checking proxy credential accessibility at startup", "total_proxies", len(proxyCredentials))

	reachableCount := 0
	unreachableCount := 0

	for _, cred := range proxyCredentials {
		// Create HTTP client with 5-second timeout for connectivity check
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err := proxy.FetchHealthFromRemoteProxy(ctx, &cred, log)
		cancel()

		if err != nil {
			unreachableCount++
			log.Warn("Proxy credential unreachable at startup",
				"name", cred.Name,
				"url", cred.BaseURL,
				"error", err.Error(),
				"recommendation", "Verify proxy service is running and network accessible. If this is expected, the proxy will be retried during runtime (every 30s)",
			)
		} else {
			reachableCount++
			log.Debug("Proxy credential accessible at startup",
				"name", cred.Name,
				"url", cred.BaseURL,
			)
		}
	}

	// Log summary
	log.Info("Proxy credential accessibility check completed at startup",
		"total_proxies", len(proxyCredentials),
		"reachable", reachableCount,
		"unreachable", unreachableCount,
	)

	// Alert if ALL proxies are unreachable (critical condition at startup)
	if unreachableCount == len(proxyCredentials) && len(proxyCredentials) > 0 {
		log.Error("WARNING: All proxy credentials are unreachable at startup",
			"healthy", 0,
			"total", len(proxyCredentials),
			"impact", "Proxy fallback routing will not be available until proxies become reachable",
			"action_recommended", "Check that all proxy services are running and network-accessible before production deployment",
		)
	}
}
