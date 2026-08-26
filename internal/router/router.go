package router

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strings"
	"sync/atomic"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/proxy/webui"
)

type Router struct {
	proxy            *proxy.Proxy
	modelManager     *models.Manager
	monitoringConfig *config.MonitoringConfig
	appConfig        *config.Config
	logger           *slog.Logger
	isReady          atomic.Bool
}

var proxiedPublicPaths = map[string]struct{}{
	"/v1/chat/completions":   {},
	"/v1/completions":        {},
	"/v1/embeddings":         {},
	"/v1/images/generations": {},
	"/v1/images/edits":       {},
	"/v1/messages":           {},
	"/v1/responses":          {},
}

var legacyPublicPathAliases = map[string]string{
	"/chat/completions":   "/v1/chat/completions",
	"/completions":        "/v1/completions",
	"/embeddings":         "/v1/embeddings",
	"/image/generations":  "/v1/images/generations",
	"/images/edits":       "/v1/images/edits",
	"/images/generations": "/v1/images/generations",
	"/messages":           "/v1/messages",
	"/models":             "/v1/models",
	"/responses":          "/v1/responses",
}

func canonicalPublicPath(path string) string {
	canonicalPath := strings.TrimSuffix(path, "/")
	if target, ok := legacyPublicPathAliases[canonicalPath]; ok {
		return target
	}
	if strings.HasPrefix(path, "/responses/") {
		return "/v1" + path
	}
	return path
}

func normalizePublicPath(req *http.Request) {
	path := canonicalPublicPath(req.URL.Path)
	if path == req.URL.Path {
		return
	}
	req.URL.Path = path
	req.URL.RawPath = ""
}

func publicPathAllowedMethod(req *http.Request) (string, bool) {
	path := req.URL.Path
	if path == "/v1/models" {
		return http.MethodGet, true
	}
	if path == "/v1/responses" && req.Method == http.MethodGet && strings.EqualFold(req.Header.Get("Upgrade"), "websocket") {
		return http.MethodGet, true
	}
	if path == "/v1/responses/compact" {
		return http.MethodPost, true
	}
	if _, ok := proxiedPublicPaths[path]; ok {
		return http.MethodPost, true
	}
	if strings.HasPrefix(path, "/v1/responses/") {
		return http.MethodGet, true
	}
	return "", false
}

func writeMethodNotAllowed(w http.ResponseWriter, allowedMethod string) {
	w.Header().Set("Allow", allowedMethod)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusMethodNotAllowed)
	_, _ = w.Write([]byte(`{"detail":"Method Not Allowed"}`))
}

// SetReady marks the router as ready (true) or not ready (false).
// Called by main: true after the TCP listener is bound, false at shutdown start.
func (r *Router) SetReady(v bool) {
	r.isReady.Store(v)
}

func New(p *proxy.Proxy, modelManager *models.Manager, monitoringConfig *config.MonitoringConfig, logger *slog.Logger, appConfig *config.Config) *Router {
	return &Router{
		proxy:            p,
		modelManager:     modelManager,
		monitoringConfig: monitoringConfig,
		appConfig:        appConfig,
		logger:           logger,
	}
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Keep the public framing policy consistent for proxy successes and for
	// locally generated auth/validation errors.
	w.Header().Set("Content-Security-Policy", "frame-ancestors 'none'")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	normalizePublicPath(req)

	// Recover from handler panics so they land in our logging system at ERROR
	// (net/http's built-in recovery only prints to stderr) and the client gets
	// a proper JSON 500 instead of a dropped connection.
	defer func() {
		if rec := recover(); rec != nil {
			if r.logger != nil {
				r.logger.ErrorContext(req.Context(), "Panic in HTTP handler",
					"error_code", http.StatusInternalServerError,
					"panic", rec,
					"method", req.Method,
					"path", req.URL.Path,
					"stack", string(debug.Stack()),
				)
			}
			// If the handler already sent headers this write is a no-op.
			proxy.WriteErrorInternal(w, "Internal Server Error")
		}
	}()

	if allowedMethod, public := publicPathAllowedMethod(req); public && req.Method != allowedMethod {
		writeMethodNotAllowed(w, allowedMethod)
		return
	}

	if req.URL.Path == r.monitoringConfig.HealthCheckPath {
		r.handleHealth(w, req)
		return
	}

	// WebUI static assets
	if req.URL.Path == "/webui/style.css" {
		webui.ServeCSS(w, req)
		return
	}

	// Visual config dashboard
	if req.URL.Path == "/vconfig" {
		r.handleVisualConfig(w, req)
		return
	}

	// Visual health dashboard
	if req.URL.Path == "/vhealth" {
		r.handleVisualHealth(w, req)
		return
	}

	// Router chain trace (JSON)
	if req.URL.Path == "/trace" {
		r.handleTrace(w, req)
		return
	}

	// Router chain trace (visual HTML)
	if req.URL.Path == "/vtrace" {
		r.handleVisualTrace(w, req)
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
		if r.proxy == nil {
			r.handleModels(w, req)
		} else {
			r.proxy.WithResponseCompatibility(w, req, r.handleModels)
		}
		return
	}

	// Handle GET /v1/responses/{response_id} — retrieve a stored response
	if req.Method == "GET" && strings.HasPrefix(req.URL.Path, "/v1/responses/") {
		r.proxy.HandleGetResponse(w, req)
		return
	}

	// Handle POST /v1/responses/compact — compact a conversation
	if req.Method == "POST" && req.URL.Path == "/v1/responses/compact" {
		r.proxy.HandleCompactResponse(w, req)
		return
	}

	// Handle WebSocket upgrade on /v1/responses
	if req.URL.Path == "/v1/responses" && strings.EqualFold(req.Header.Get("Upgrade"), "websocket") {
		r.proxy.HandleWebSocketResponses(w, req)
		return
	}

	if _, allowed := proxiedPublicPaths[req.URL.Path]; !allowed {
		proxy.WriteErrorNotFound(w, "Not Found")
		return
	}

	if r.monitoringConfig.LogErrors {
		// Capture request body for logging (detects streaming requests)
		reqBody, isStreaming, err := captureRequestBody(req)
		if err != nil {
			r.proxyPublicRequest(w, req)
			return
		}

		// Create response capture wrapper
		rc := newResponseCapture(w)

		// Proxy the request through captured response
		r.proxyPublicRequest(rc, req)

		// Log error responses if enabled and status is error (4xx or 5xx).
		// Skip logging for streaming requests to avoid memory overhead with large responses.
		if r.monitoringConfig.ErrorsLogPath != "" && isErrorStatus(rc.statusCode) && !isStreaming {
			// Don't fail the response on a logging error, but surface it —
			// a silently broken errors-log file hides every subsequent error.
			if err := logErrorResponse(r.monitoringConfig.ErrorsLogPath, req, rc, reqBody); err != nil && r.logger != nil {
				r.logger.WarnContext(req.Context(), "Failed to write error log file",
					"path", r.monitoringConfig.ErrorsLogPath, "error", err)
			}
		}
	} else {
		r.proxyPublicRequest(w, req)
	}
}

func (r *Router) handleModels(w http.ResponseWriter, req *http.Request) {
	tokenInfo, visibility, ok := r.proxy.AuthenticateClientRequestScoped(w, req)
	if !ok {
		return
	}

	var modelsResp models.ModelsResponse
	var organizationPolicy *models.OrganizationPolicy
	if r.proxy != nil {
		var dangling bool
		organizationPolicy, dangling = r.proxy.OrganizationPolicyForTokenInfo(tokenInfo)
		if dangling {
			proxy.WriteErrorForbidden(w, "Forbidden")
			return
		}
	}
	if r.modelManager != nil {
		includeGroups := strings.EqualFold(req.URL.Query().Get("include_model_access_groups"), "true")
		if organizationPolicy != nil {
			if includeGroups {
				modelsResp = r.modelManager.GetAllModelsWithAccessGroupsScopedForOrganization(visibility, organizationPolicy)
			} else {
				modelsResp = r.modelManager.GetAllModelsScopedForOrganization(visibility, organizationPolicy)
			}
		} else if includeGroups {
			modelsResp = r.modelManager.GetAllModelsWithAccessGroupsScoped(visibility)
		} else {
			modelsResp = r.modelManager.GetAllModelsScoped(visibility)
		}
	} else {
		modelsResp = models.ModelsResponse{Object: "list", Data: []models.Model{}}
	}
	if tokenInfo != nil {
		filtered := make([]models.Model, 0, len(modelsResp.Data))
		for _, model := range modelsResp.Data {
			if r.proxy.IsModelAllowedForToken(tokenInfo, model.ID) {
				filtered = append(filtered, model)
			}
		}
		modelsResp.Data = filtered
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(modelsResp); err != nil {
		if r.logger != nil {
			r.logger.ErrorContext(req.Context(), "Failed to encode models response",
				"endpoint", "/v1/models",
				"error", err.Error(),
			)
		}
		// Headers already sent, cannot send http.Error
		return
	}
}
