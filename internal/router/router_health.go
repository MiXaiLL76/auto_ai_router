package router

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

func (r *Router) handleHealth(w http.ResponseWriter, req *http.Request) {
	visibility := r.visibilityScopeOrPublic(req)
	_, status := r.proxy.HealthCheckScoped(visibility)
	serviceHealthy := r.proxy.HasAvailableCredentials()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Router-Version", r.proxy.GetVersion())
	w.Header().Set("X-Router-Commit", r.proxy.GetCommit())
	if !serviceHealthy {
		w.WriteHeader(http.StatusServiceUnavailable)
	} else {
		w.WriteHeader(http.StatusOK)
	}

	if err := json.NewEncoder(w).Encode(status); err != nil {
		if r.logger != nil {
			r.logger.ErrorContext(req.Context(), "Failed to encode health response",
				"endpoint", "/health",
				"error", err.Error(),
			)
		}
		// Headers already sent, cannot send http.Error
		return
	}
}

type Readiness struct {
	Status              string   `json:"status"`
	DB                  string   `json:"db"`
	Cache               string   `json:"cache"`
	LitellmVersion      string   `json:"litellm_version"`
	SuccessCallbacks    []string `json:"success_callbacks"`
	UseAioHTTPTransport bool     `json:"use_aiohttp_transport"`
	LastUpdated         string   `json:"last_updated"`
}

func (r *Router) handleReadiness(w http.ResponseWriter, req *http.Request) {
	ready := r.isReady.Load()

	status := "ready"
	httpStatus := http.StatusOK
	if !ready {
		status = "not_ready"
		httpStatus = http.StatusServiceUnavailable
	}

	var body = Readiness{
		Status:         status,
		LitellmVersion: r.proxy.GetVersion(),
		LastUpdated:    utils.NowUTC().Format(time.RFC3339),
	}

	if r.proxy.LiteLLMDB.IsHealthy() {
		body.DB = "connected"
	} else {
		body.DB = "Not connected"
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(httpStatus)

	if err := json.NewEncoder(w).Encode(body); err != nil {
		if r.logger != nil {
			r.logger.ErrorContext(req.Context(), "Failed to encode readiness response",
				"endpoint", "/health/readiness",
				"error", err.Error(),
			)
		}
		// Headers already sent, cannot send http.Error
		return
	}
}

func (r *Router) handleVisualHealth(w http.ResponseWriter, req *http.Request) {
	r.proxy.VisualHealthCheck(w, req)
}
