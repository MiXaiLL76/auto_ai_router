package router

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

type Router struct {
	proxy            *proxy.Proxy
	modelManager     *models.Manager
	monitoringConfig *config.MonitoringConfig
}

func New(p *proxy.Proxy, modelManager *models.Manager, monitoringConfig *config.MonitoringConfig) *Router {
	return &Router{
		proxy:            p,
		modelManager:     modelManager,
		monitoringConfig: monitoringConfig,
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

	// Handle GET /v1/models
	if req.URL.Path == "/v1/models" && req.Method == "GET" {
		r.handleModels(w, req)
		return
	}

	if strings.HasPrefix(req.URL.Path, "/v1/") {
		if r.monitoringConfig.LogErrors {
			// Capture request body for logging
			reqBody, err := captureRequestBody(req)
			if err != nil {
				r.proxy.ProxyRequest(w, req)
				return
			}

			// Create response capture wrapper
			rc := newResponseCapture(w)

			// Proxy the request through captured response
			r.proxy.ProxyRequest(rc, req)

			// Log error responses if logging is enabled and status is 4xx or 5xx
			if r.monitoringConfig.ErrorsLogPath != "" && isErrorStatus(rc.statusCode) {
				_ = logErrorResponse(r.monitoringConfig.ErrorsLogPath, req, rc, reqBody)
				// Log error internally but don't fail the response
				// (error logging shouldn't break the API response)
			}
		} else {
			r.proxy.ProxyRequest(w, req)
		}
		return
	}

	http.Error(w, "Not Found", http.StatusNotFound)
}

func (r *Router) handleHealth(w http.ResponseWriter, req *http.Request) {
	healthy, status := r.proxy.HealthCheck()

	w.Header().Set("Content-Type", "application/json")
	if !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
	} else {
		w.WriteHeader(http.StatusOK)
	}

	if err := json.NewEncoder(w).Encode(status); err != nil {
		return
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
		return
	}
}

func (r *Router) handleVisualHealth(w http.ResponseWriter, req *http.Request) {
	r.proxy.VisualHealthCheck(w, req)
}
