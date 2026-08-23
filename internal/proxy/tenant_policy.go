package proxy

import (
	"net"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

func normalizeRequestHostname(hostport string) string {
	hostport = strings.TrimSpace(hostport)
	if host, _, err := net.SplitHostPort(hostport); err == nil {
		hostport = host
	}
	hostport = strings.TrimPrefix(strings.TrimSuffix(hostport, "]"), "[")
	return strings.ToLower(strings.TrimSuffix(hostport, "."))
}

func containsString(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func (p *Proxy) hasExplicitProtectedModelACL(modelIDs []string, visibility scope.Context) bool {
	if len(modelIDs) == 0 {
		return false
	}
	if p == nil || p.modelManager == nil {
		return false
	}
	for _, modelID := range modelIDs {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" || modelID == "*" || modelID == "all-proxy-models" ||
			modelID == "all-team-models" || strings.ContainsRune(modelID, '*') {
			return false
		}
		// Exact configured public IDs and tenant-scoped aliases are allowed.
		// Regex-looking entries are rejected implicitly because they are not a
		// routable product identifier in the model manager.
		if !p.modelManager.IsClientModelIDRoutableScoped(modelID, visibility) {
			return false
		}
	}
	return true
}

func (p *Proxy) protectedTenantForOrganization(organizationID string) *config.ProtectedTenantConfig {
	if p == nil || organizationID == "" {
		return nil
	}
	for index := range p.protectedTenants {
		if containsString(p.protectedTenants[index].OrganizationIDs, organizationID) {
			return &p.protectedTenants[index]
		}
	}
	return nil
}

func (p *Proxy) protectedTenantForHostname(hostname string) *config.ProtectedTenantConfig {
	if p == nil || hostname == "" {
		return nil
	}
	for index := range p.protectedTenants {
		if containsString(p.protectedTenants[index].Hostnames, hostname) {
			return &p.protectedTenants[index]
		}
	}
	return nil
}

func (p *Proxy) rejectProtectedTenant(w http.ResponseWriter, logCtx *RequestLogContext, message string) bool {
	if logCtx != nil {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusForbidden
		logCtx.ErrorMsg = message
		// Admission failures are not billable and happen before a provider exists.
		logCtx.Logged = true
	}
	WriteErrorForbidden(w, message)
	return false
}

func (p *Proxy) enforceProtectedTenantAdmission(w http.ResponseWriter, r *http.Request, logCtx *RequestLogContext) bool {
	if p == nil || logCtx == nil || logCtx.TokenInfo == nil || logCtx.TokenInfo.IsMasterKey {
		return true
	}

	hostname := normalizeRequestHostname(r.Host)
	organizationPolicy := p.protectedTenantForOrganization(logCtx.TokenInfo.OrganizationID)
	hostPolicy := p.protectedTenantForHostname(hostname)
	if hostPolicy != nil && organizationPolicy != hostPolicy {
		return p.rejectProtectedTenant(w, logCtx, "Organization is not allowed for this hostname")
	}
	// Protected scopes are reserved to the organizations which own their policy.
	// A copied/misconfigured air_scopes value must not unlock aliases or routes
	// for another organization on an ordinary hostname.
	for index := range p.protectedTenants {
		policy := &p.protectedTenants[index]
		if len(policy.RequiredScopes) == 0 || organizationPolicy == policy {
			continue
		}
		if logCtx.Scope.AllowsExpression(scope.FromScopes(policy.RequiredScopes, nil)) {
			return p.rejectProtectedTenant(w, logCtx, "Protected tenant scope does not belong to this organization")
		}
	}
	if organizationPolicy == nil {
		return true
	}

	if len(organizationPolicy.RequiredScopes) > 0 &&
		!logCtx.Scope.AllowsExpression(scope.FromScopes(organizationPolicy.RequiredScopes, nil)) {
		return p.rejectProtectedTenant(w, logCtx, "Required tenant scope is missing")
	}
	if organizationPolicy.RequireModelACL && !p.hasExplicitProtectedModelACL(logCtx.TokenInfo.Models, logCtx.Scope) {
		return p.rejectProtectedTenant(w, logCtx, "Explicit model access is required")
	}
	if organizationPolicy.RequireRouteACL {
		if hasTrustedRouteAuthorization(r) {
			return true
		}
		switch evaluateRouteACL(logCtx.TokenInfo.AllowedRoutes, r.Method, r.URL.Path) {
		case routeACLAllowed:
			return true
		case routeACLUnconfigured:
			return p.rejectProtectedTenant(w, logCtx, "Explicit route access is required")
		default:
			return p.rejectProtectedTenant(w, logCtx, "Route is not allowed")
		}
	}
	return true
}
