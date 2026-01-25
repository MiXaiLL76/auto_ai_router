package router

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

type Router struct {
	proxy           *proxy.Proxy
	healthCheckPath string
	modelManager    *models.Manager
}

func New(p *proxy.Proxy, healthCheckPath string, modelManager *models.Manager) *Router {
	return &Router{
		proxy:           p,
		healthCheckPath: healthCheckPath,
		modelManager:    modelManager,
	}
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path == r.healthCheckPath {
		r.handleHealth(w, req)
		return
	}

	// Visual health dashboard
	if req.URL.Path == "/vhealth" {
		r.handleVisualHealth(w, req)
		return
	}

	// Handle GET /v1/models if model manager is enabled
	if req.URL.Path == "/v1/models" && req.Method == "GET" && r.modelManager != nil && r.modelManager.IsEnabled() {
		r.handleModels(w, req)
		return
	}

	if strings.HasPrefix(req.URL.Path, "/v1/") {
		r.proxy.ProxyRequest(w, req)
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
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

func (r *Router) handleModels(w http.ResponseWriter, req *http.Request) {
	modelsResp := r.modelManager.GetAllModels()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(modelsResp); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

func (r *Router) handleVisualHealth(w http.ResponseWriter, req *http.Request) {
	r.proxy.VisualHealthCheck(w, req)
}
