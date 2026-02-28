package router

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

type Router struct {
	proxy            *proxy.Proxy
	modelManager     *models.Manager
	monitoringConfig *config.MonitoringConfig
	logger           *slog.Logger
}

func New(p *proxy.Proxy, modelManager *models.Manager, monitoringConfig *config.MonitoringConfig, logger *slog.Logger) *Router {
	return &Router{
		proxy:            p,
		modelManager:     modelManager,
		monitoringConfig: monitoringConfig,
		logger:           logger,
	}
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path == r.monitoringConfig.HealthCheckPath {
		r.handleHealth(w, req)
		return
	}

	// Visual health dashboard
	if req.URL.Path == "/vhealth" {
		r.handleVisualHealth(w, req)
		return
	}

	if req.URL.Path == "/health/readiness" {
		r.handleReadiness(w, req)
		return
	}

	if r.handleLitellm(w, req) {
		return
	}

	// Handle GET /v1/models
	if req.URL.Path == "/v1/models" && req.Method == "GET" {
		r.handleModels(w, req)
		return
	}

	allowedPaths := map[string]bool{
		"/v1/chat/completions":   true,
		"/v1/completions":        true,
		"/v1/embeddings":         true,
		"/v1/images/generations": true,
		"/v1/responses":          true,
	}
	if !allowedPaths[req.URL.Path] {
		http.Error(w, "Not Found", http.StatusNotFound)
		return
	}

	if r.monitoringConfig.LogErrors {
		// Capture request body for logging (detects streaming requests)
		reqBody, isStreaming, err := captureRequestBody(req)
		if err != nil {
			r.proxy.ProxyRequest(w, req)
			return
		}

		// Create response capture wrapper
		rc := newResponseCapture(w)

		// Proxy the request through captured response
		r.proxy.ProxyRequest(rc, req)

		// Log error responses if enabled and status is error (4xx or 5xx).
		// Skip logging for streaming requests to avoid memory overhead with large responses.
		if r.monitoringConfig.ErrorsLogPath != "" && isErrorStatus(rc.statusCode) && !isStreaming {
			_ = logErrorResponse(r.monitoringConfig.ErrorsLogPath, req, rc, reqBody)
			// Log error internally but don't fail the response
			// (error logging shouldn't break the API response)
		}
	} else {
		r.proxy.ProxyRequest(w, req)
	}
}

func (r *Router) handleModels(w http.ResponseWriter, req *http.Request) {
	var modelsResp models.ModelsResponse
	if r.modelManager != nil {
		modelsResp = r.modelManager.GetAllModels()
	} else {
		modelsResp = models.ModelsResponse{Object: "list", Data: []models.Model{}}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(modelsResp); err != nil {
		if r.logger != nil {
			r.logger.Error("Failed to encode models response",
				"endpoint", "/v1/models",
				"error", err.Error(),
			)
		}
		// Headers already sent, cannot send http.Error
		return
	}
}
