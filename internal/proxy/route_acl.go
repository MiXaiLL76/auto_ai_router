package proxy

import (
	"net/http"
	"strings"
)

// routeACLDecision keeps an absent ACL distinct from a configured ACL which
// did not match. LiteLLM verification-token rows use an empty allowed_routes
// array for the legacy unrestricted case; the integration point must decide
// whether that legacy default is acceptable for its tenant policy.
type routeACLDecision uint8

const (
	routeACLUnconfigured routeACLDecision = iota
	routeACLDenied
	routeACLAllowed
)

type publicRouteGroup uint8

const (
	publicRouteOpenAI publicRouteGroup = 1 << iota
	publicRouteLLM
	publicRouteInfo
)

var routeACLLegacyAliases = map[string]string{
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

var routeACLStaticPublicRoutes = map[string]map[string]publicRouteGroup{
	http.MethodGet: {
		"/v1/models":    publicRouteOpenAI | publicRouteLLM | publicRouteInfo,
		"/v1/responses": publicRouteOpenAI | publicRouteLLM,
	},
	http.MethodPost: {
		"/v1/chat/completions":   publicRouteOpenAI | publicRouteLLM,
		"/v1/completions":        publicRouteOpenAI | publicRouteLLM,
		"/v1/embeddings":         publicRouteOpenAI | publicRouteLLM,
		"/v1/images/generations": publicRouteOpenAI | publicRouteLLM,
		"/v1/images/edits":       publicRouteOpenAI | publicRouteLLM,
		"/v1/messages":           publicRouteLLM,
		"/v1/responses":          publicRouteOpenAI | publicRouteLLM,
		"/v1/responses/compact":  publicRouteOpenAI | publicRouteLLM,
	},
}

// evaluateRouteACL evaluates LiteLLM VerificationToken.allowed_routes against
// AIR's public API surface. An empty slice is deliberately not interpreted as
// allow or deny: routeACLUnconfigured lets the caller preserve legacy
// unrestricted keys or enforce a tenant-specific fail-closed policy.
//
// Plain LiteLLM route entries are path-only permissions. They allow every HTTP
// method which AIR supports for that public path. AIR additionally understands
// "METHOD /path" (and "METHOD:/path") entries so method-scoped data already
// stored by an operator remains enforceable.
//
// Exact entries, the LiteLLM presets llm_api_routes/openai_routes/info_routes,
// the response-id template, LiteLLM segment-boundary prefixes, and terminal /*
// wildcards are supported. No other glob matching is performed.
func evaluateRouteACL(allowedRoutes []string, method, path string) routeACLDecision {
	if len(allowedRoutes) == 0 {
		return routeACLUnconfigured
	}

	method = strings.ToUpper(strings.TrimSpace(method))
	path = canonicalRouteACLPath(path)
	groups, public := classifyRouteACLPublicRoute(method, path)
	if !public {
		return routeACLDenied
	}

	for _, rawEntry := range allowedRoutes {
		entryMethod, entryPath, ok := parseRouteACLEntry(rawEntry)
		if !ok || (entryMethod != "" && entryMethod != method) {
			continue
		}

		switch entryPath {
		case "llm_api_routes":
			if groups&publicRouteLLM != 0 {
				return routeACLAllowed
			}
		case "openai_routes":
			if groups&publicRouteOpenAI != 0 {
				return routeACLAllowed
			}
		case "info_routes":
			if groups&publicRouteInfo != 0 {
				return routeACLAllowed
			}
		default:
			entryPath = canonicalRouteACLPath(entryPath)
			if routeACLPathMatches(entryPath, path) {
				return routeACLAllowed
			}
		}
	}

	return routeACLDenied
}

func canonicalRouteACLPath(path string) string {
	trimmedTrailingSlash := strings.TrimSuffix(path, "/")
	if canonical, ok := routeACLLegacyAliases[trimmedTrailingSlash]; ok {
		return canonical
	}
	if strings.HasPrefix(path, "/responses/") {
		return "/v1" + path
	}
	return path
}

func classifyRouteACLPublicRoute(method, path string) (publicRouteGroup, bool) {
	if methodRoutes, ok := routeACLStaticPublicRoutes[method]; ok {
		if groups, ok := methodRoutes[path]; ok {
			return groups, true
		}
	}

	if method == http.MethodGet && isRouteACLResponseIDPath(path) {
		return publicRouteOpenAI | publicRouteLLM, true
	}
	return 0, false
}

func isRouteACLResponseIDPath(path string) bool {
	const prefix = "/v1/responses/"
	if !strings.HasPrefix(path, prefix) {
		return false
	}
	remainder := strings.TrimPrefix(path, prefix)
	return remainder != "" && remainder != "compact" && !strings.Contains(remainder, "/")
}

func parseRouteACLEntry(raw string) (method, path string, ok bool) {
	entry := strings.TrimSpace(raw)
	if entry == "" {
		return "", "", false
	}

	if fields := strings.Fields(entry); len(fields) == 2 && isRouteACLHTTPMethod(fields[0]) {
		return strings.ToUpper(fields[0]), fields[1], true
	} else if len(fields) != 1 {
		return "", "", false
	}

	if separator := strings.IndexByte(entry, ':'); separator > 0 {
		candidateMethod := entry[:separator]
		if isRouteACLHTTPMethod(candidateMethod) {
			candidatePath := strings.TrimSpace(entry[separator+1:])
			if candidatePath == "" {
				return "", "", false
			}
			return strings.ToUpper(candidateMethod), candidatePath, true
		}
	}

	return "", entry, true
}

func isRouteACLHTTPMethod(method string) bool {
	switch strings.ToUpper(method) {
	case http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut,
		http.MethodPatch, http.MethodDelete, http.MethodConnect, http.MethodOptions,
		http.MethodTrace:
		return true
	default:
		return false
	}
}

func routeACLPathMatches(allowedPath, requestPath string) bool {
	if allowedPath == requestPath {
		return true
	}
	if allowedPath == "/v1/responses/{response_id}" || allowedPath == "/v1/responses/{id}" {
		return isRouteACLResponseIDPath(requestPath)
	}
	if !strings.HasSuffix(allowedPath, "/*") || strings.Count(allowedPath, "*") != 1 {
		// LiteLLM treats a configured route as a segment-boundary prefix. Keep
		// that compatibility while classifyRouteACLPublicRoute limits the result
		// to AIR's actual public surface.
		return strings.HasPrefix(requestPath, allowedPath+"/")
	}
	prefix := strings.TrimSuffix(allowedPath, "*")
	return strings.HasPrefix(requestPath, prefix)
}
