package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/proxy/webui"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

const defaultTraceDepth = 25

// TraceCheck builds a recursive trace of this router and all reachable downstream proxy routers.
// depth controls how many hops to follow (0 = local only).
func (p *Proxy) TraceCheck(ctx context.Context, depth int) *httputil.ProxyTraceResponse {
	return p.TraceCheckScoped(ctx, depth, scope.AdminContext())
}

func (p *Proxy) TraceCheckScoped(ctx context.Context, depth int, visibility scope.Context) *httputil.ProxyTraceResponse {
	_, health := p.HealthCheckScoped(visibility)

	status := "healthy"
	if health.CredentialsAvailable == 0 {
		status = "unhealthy"
	} else if health.CredentialsBanned > 0 {
		status = "degraded"
	}

	trace := &httputil.ProxyTraceResponse{
		RouterID:    p.routerID,
		Status:      status,
		Credentials: health.Credentials,
		Models:      health.Models,
	}

	if depth <= 0 {
		return trace
	}

	var proxyCreds []config.CredentialConfig
	for _, cred := range visibleCredentials(p.balancer.GetCredentialsSnapshot(), visibility) {
		if cred.IsProxyLike() {
			proxyCreds = append(proxyCreds, cred)
		}
	}

	if len(proxyCreds) > 0 {
		type result struct {
			name     string
			upstream *httputil.ProxyTraceResponse
			err      error
		}
		results := make(chan result, len(proxyCreds))

		var wg sync.WaitGroup
		for _, cred := range proxyCreds {
			wg.Add(1)
			go func(c config.CredentialConfig) {
				defer wg.Done()
				upstream, err := p.fetchUpstreamTrace(ctx, &c, depth-1)
				results <- result{name: c.Name, upstream: upstream, err: err}
			}(cred)
		}
		wg.Wait()
		close(results)

		trace.Upstreams = make(map[string]*httputil.ProxyTraceResponse, len(proxyCreds))
		for r := range results {
			if r.err != nil {
				trace.Upstreams[r.name] = &httputil.ProxyTraceResponse{FetchError: r.err.Error()}
			} else {
				trace.Upstreams[r.name] = r.upstream
			}
		}
	}

	return trace
}

func (p *Proxy) fetchUpstreamTrace(ctx context.Context, cred *config.CredentialConfig, depth int) (*httputil.ProxyTraceResponse, error) {
	var result httputil.ProxyTraceResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, fmt.Sprintf("/trace?depth=%d", depth), p.logger, &result); err != nil {
		p.logger.DebugContext(ctx, "upstream /trace unavailable, falling back to /health", "credential", cred.Name, "error", err)
		return p.fetchUpstreamHealth(ctx, cred)
	}
	return &result, nil
}

// fetchUpstreamHealth is a fallback for upstream routers that don't have /trace yet.
// It fetches /health and converts the result to a ProxyTraceResponse (no recursive upstreams).
func (p *Proxy) fetchUpstreamHealth(ctx context.Context, cred *config.CredentialConfig) (*httputil.ProxyTraceResponse, error) {
	var health httputil.ProxyHealthResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, "/health", p.logger, &health); err != nil {
		return nil, err
	}
	return &httputil.ProxyTraceResponse{
		RouterID:    cred.Name,
		Status:      health.Status,
		Credentials: health.Credentials,
		Models:      health.Models,
	}, nil
}

// HandleTrace returns the full trace tree as JSON.
func (p *Proxy) HandleTrace(w http.ResponseWriter, r *http.Request) {
	p.HandleTraceScoped(w, r, scope.AdminContext())
}

func (p *Proxy) HandleTraceScoped(w http.ResponseWriter, r *http.Request, visibility scope.Context) {
	depth := defaultTraceDepth
	if d := r.URL.Query().Get("depth"); d != "" {
		if n, err := strconv.Atoi(d); err == nil && n >= 0 && n <= 10 {
			depth = n
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	trace := p.TraceCheckScoped(ctx, depth, visibility)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(trace); err != nil {
		p.logger.ErrorContext(r.Context(), "Failed to encode trace response", "error", err)
	}
}

// HandleVisualTrace serves the static trace dashboard HTML.
func (p *Proxy) HandleVisualTrace(w http.ResponseWriter, r *http.Request) {
	webui.ServeTrace(w, r)
}
