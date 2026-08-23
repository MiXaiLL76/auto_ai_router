package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEvaluateRouteACLEmptyIsExposedAsUnconfigured(t *testing.T) {
	assert.Equal(t, routeACLUnconfigured, evaluateRouteACL(nil, http.MethodPost, "/v1/chat/completions"))
	assert.Equal(t, routeACLUnconfigured, evaluateRouteACL([]string{}, http.MethodPost, "/v1/chat/completions"))

	// A non-empty but unusable ACL is configured and therefore fails closed.
	assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{""}, http.MethodPost, "/v1/chat/completions"))
}

func TestEvaluateRouteACLExactCanonicalPaths(t *testing.T) {
	tests := []struct {
		name    string
		allowed []string
		method  string
		path    string
		want    routeACLDecision
	}{
		{name: "chat", allowed: []string{"/v1/chat/completions"}, method: http.MethodPost, path: "/v1/chat/completions", want: routeACLAllowed},
		{name: "models", allowed: []string{"/v1/models"}, method: http.MethodGet, path: "/v1/models", want: routeACLAllowed},
		{name: "response create", allowed: []string{"/v1/responses"}, method: http.MethodPost, path: "/v1/responses", want: routeACLAllowed},
		{name: "response websocket path", allowed: []string{"/v1/responses"}, method: http.MethodGet, path: "/v1/responses", want: routeACLAllowed},
		{name: "compact", allowed: []string{"/v1/responses/compact"}, method: http.MethodPost, path: "/v1/responses/compact", want: routeACLAllowed},
		{name: "unknown route", allowed: []string{"/*"}, method: http.MethodGet, path: "/health", want: routeACLDenied},
		{name: "wrong method", allowed: []string{"/v1/models"}, method: http.MethodPost, path: "/v1/models", want: routeACLDenied},
		{name: "canonical trailing slash is not a route", allowed: []string{"/v1/chat/completions/*"}, method: http.MethodPost, path: "/v1/chat/completions/", want: routeACLDenied},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, evaluateRouteACL(tt.allowed, tt.method, tt.path))
		})
	}
}

func TestEvaluateRouteACLNormalizesLegacyAliases(t *testing.T) {
	tests := []struct {
		name        string
		allowed     string
		requestPath string
		method      string
	}{
		{name: "legacy ACL canonical request", allowed: "/chat/completions", requestPath: "/v1/chat/completions", method: http.MethodPost},
		{name: "canonical ACL legacy request", allowed: "/v1/embeddings", requestPath: "/embeddings", method: http.MethodPost},
		{name: "singular image alias", allowed: "/image/generations", requestPath: "/v1/images/generations", method: http.MethodPost},
		{name: "legacy models trailing slash", allowed: "/v1/models", requestPath: "/models/", method: http.MethodGet},
		{name: "legacy response id", allowed: "/responses/{response_id}", requestPath: "/v1/responses/resp_123", method: http.MethodGet},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, routeACLAllowed, evaluateRouteACL([]string{tt.allowed}, tt.method, tt.requestPath))
		})
	}
}

func TestEvaluateRouteACLLiteLLMPresetsOnlyCoverAIRPublicRoutes(t *testing.T) {
	tests := []struct {
		name   string
		preset string
		method string
		path   string
		want   routeACLDecision
	}{
		{name: "llm covers openai", preset: "llm_api_routes", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLAllowed},
		{name: "llm covers anthropic messages", preset: "llm_api_routes", method: http.MethodPost, path: "/v1/messages", want: routeACLAllowed},
		{name: "llm covers response retrieval", preset: "llm_api_routes", method: http.MethodGet, path: "/v1/responses/resp_123", want: routeACLAllowed},
		{name: "openai covers discovery", preset: "openai_routes", method: http.MethodGet, path: "/v1/models", want: routeACLAllowed},
		{name: "openai excludes anthropic", preset: "openai_routes", method: http.MethodPost, path: "/v1/messages", want: routeACLDenied},
		{name: "info covers discovery", preset: "info_routes", method: http.MethodGet, path: "/models", want: routeACLAllowed},
		{name: "info excludes inference", preset: "info_routes", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLDenied},
		{name: "preset cannot expose unsupported litellm route", preset: "llm_api_routes", method: http.MethodPost, path: "/v1/audio/speech", want: routeACLDenied},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, evaluateRouteACL([]string{tt.preset}, tt.method, tt.path))
		})
	}
}

func TestEvaluateRouteACLResponseIDIsOneSafeDynamicSegment(t *testing.T) {
	for _, allowed := range []string{"/v1/responses/{response_id}", "/v1/responses/{id}"} {
		assert.Equal(t, routeACLAllowed, evaluateRouteACL([]string{allowed}, http.MethodGet, "/v1/responses/resp_123"))
		assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{allowed}, http.MethodGet, "/v1/responses/compact"))
		assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{allowed}, http.MethodGet, "/v1/responses/resp_123/input_items"))
		assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{allowed}, http.MethodPost, "/v1/responses/resp_123"))
	}

	assert.Equal(t, routeACLAllowed, evaluateRouteACL([]string{"/v1/responses/resp_123"}, http.MethodGet, "/v1/responses/resp_123"))
}

func TestEvaluateRouteACLTerminalWildcardOnly(t *testing.T) {
	tests := []struct {
		name    string
		allowed string
		method  string
		path    string
		want    routeACLDecision
	}{
		{name: "response descendants", allowed: "/v1/responses/*", method: http.MethodGet, path: "/v1/responses/resp_123", want: routeACLAllowed},
		{name: "wildcard does not imply base", allowed: "/v1/responses/*", method: http.MethodPost, path: "/v1/responses", want: routeACLDenied},
		{name: "global public wildcard", allowed: "/*", method: http.MethodPost, path: "/v1/images/edits", want: routeACLAllowed},
		{name: "nonterminal wildcard", allowed: "/v1/*/completions", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLDenied},
		{name: "bare star", allowed: "*", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLDenied},
		{name: "litellm segment prefix", allowed: "/v1", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLAllowed},
		{name: "root is not global prefix", allowed: "/", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLDenied},
		{name: "trailing slash does not widen", allowed: "/v1/", method: http.MethodPost, path: "/v1/chat/completions", want: routeACLDenied},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, evaluateRouteACL([]string{tt.allowed}, tt.method, tt.path))
		})
	}
}

func TestEvaluateRouteACLMethodQualifiedEntries(t *testing.T) {
	assert.Equal(t, routeACLAllowed, evaluateRouteACL([]string{"GET /v1/responses"}, http.MethodGet, "/v1/responses"))
	assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{"GET /v1/responses"}, http.MethodPost, "/v1/responses"))
	assert.Equal(t, routeACLAllowed, evaluateRouteACL([]string{"post:/responses"}, http.MethodPost, "/v1/responses"))
	assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{"post:/responses"}, http.MethodGet, "/v1/responses"))
	assert.Equal(t, routeACLAllowed, evaluateRouteACL([]string{"GET info_routes"}, http.MethodGet, "/v1/models"))
	assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{"POST info_routes"}, http.MethodGet, "/v1/models"))
	assert.Equal(t, routeACLDenied, evaluateRouteACL([]string{"GET /v1/models extra"}, http.MethodGet, "/v1/models"))
}
