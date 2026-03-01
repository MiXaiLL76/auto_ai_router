package router

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

func (r *Router) handleHealth(w http.ResponseWriter, req *http.Request) {
	healthy, status := r.proxy.HealthCheck()

	w.Header().Set("Content-Type", "application/json")
	if !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
	} else {
		w.WriteHeader(http.StatusOK)
	}

	if err := json.NewEncoder(w).Encode(status); err != nil {
		if r.logger != nil {
			r.logger.Error("Failed to encode health response",
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
	UseAioHttpTransport bool     `json:"use_aiohttp_transport"`
	LastUpdated         string   `json:"last_updated"`
}

func (r *Router) handleReadiness(w http.ResponseWriter, req *http.Request) {
	var body = Readiness{
		Status:         "healthy",
		LitellmVersion: proxy.Version,
		LastUpdated:    utils.NowUTC().Format(time.RFC3339),
	}

	if r.proxy.LiteLLMDB.IsHealthy() {
		body.DB = "connected"
	} else {
		body.DB = "Not connected"
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(body); err != nil {
		if r.logger != nil {
			r.logger.Error("Failed to encode readiness response",
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
