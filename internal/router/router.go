package router

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

type Router struct {
	proxy           *proxy.Proxy
	healthCheckPath string
}

func New(p *proxy.Proxy, healthCheckPath string) *Router {
	return &Router{
		proxy:           p,
		healthCheckPath: healthCheckPath,
	}
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path == r.healthCheckPath {
		r.handleHealth(w, req)
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

	json.NewEncoder(w).Encode(status)
}
