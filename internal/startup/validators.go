package startup

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

// ValidateProxyCredentialsAtStartup performs connectivity checks for proxy/AIR credentials at startup.
// For each remote-router credential, it attempts an HTTP GET to the /health endpoint with a 5-second timeout.
// Results are logged as WARN if unreachable, but startup continues (non-blocking).
// This helps catch misconfigured proxies early without failing the startup sequence.
func ValidateProxyCredentialsAtStartup(cfg *config.Config, log *slog.Logger) {
	proxyCredentials := make([]config.CredentialConfig, 0)
	for _, cred := range cfg.Credentials {
		if cred.IsProxyLike() {
			proxyCredentials = append(proxyCredentials, cred)
		}
	}

	if len(proxyCredentials) == 0 {
		return
	}

	log.Info("Checking proxy/AIR credential accessibility at startup", "total_remote_routers", len(proxyCredentials))

	reachableCount := 0
	unreachableCount := 0

	for _, cred := range proxyCredentials {
		// Create HTTP client with 5-second timeout for connectivity check
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, headers, err := proxy.FetchHealthResponseFromRemoteProxy(ctx, &cred, log)
		cancel()

		if err != nil {
			unreachableCount++
			log.Warn("Proxy/AIR credential unreachable at startup",
				"name", cred.Name,
				"url", cred.BaseURL,
				"error", err.Error(),
				"recommendation", "Verify proxy service is running and network accessible. If this is expected, the proxy will be retried during runtime (every 30s)",
			)
		} else {
			reachableCount++
			warnIfGenericProxyPointsToAIR(cred, headers, log)
			log.Debug("Proxy/AIR credential accessible at startup",
				"name", cred.Name,
				"url", cred.BaseURL,
			)
		}
	}

	// Log summary
	log.Info("Proxy/AIR credential accessibility check completed at startup",
		"total_remote_routers", len(proxyCredentials),
		"reachable", reachableCount,
		"unreachable", unreachableCount,
	)

	// Alert if ALL proxies are unreachable (critical condition at startup)
	if unreachableCount == len(proxyCredentials) && len(proxyCredentials) > 0 {
		log.Error("WARNING: All proxy/AIR credentials are unreachable at startup",
			"healthy", 0,
			"total", len(proxyCredentials),
			"impact", "Proxy/AIR fallback routing will not be available until remote routers become reachable",
			"action_recommended", "Check that all proxy/AIR services are running and network-accessible before production deployment",
		)
	}
}

func warnIfGenericProxyPointsToAIR(cred config.CredentialConfig, headers http.Header, log *slog.Logger) {
	if cred.Type != config.ProviderTypeProxy || headers == nil {
		return
	}
	routerVersion := headers.Get("X-Router-Version")
	routerCommit := headers.Get("X-Router-Commit")
	if routerVersion == "" && routerCommit == "" {
		return
	}
	log.Warn("Proxy credential appears to point to an Auto AI Router upstream",
		"name", cred.Name,
		"url", cred.BaseURL,
		"configured_type", cred.Type,
		"recommended_type", config.ProviderTypeAIR,
		"router_version", routerVersion,
		"router_commit", routerCommit,
		"impact", "cached audio usage may be billed incorrectly when AIR upstreams are configured as generic proxy credentials",
	)
}
