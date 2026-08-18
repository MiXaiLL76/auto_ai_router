package proxy

import (
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

func shouldMaskProxyResponseErrors(cred *config.CredentialConfig, resp *ProxyResponse) bool {
	if shouldMaskUpstreamErrors(cred) {
		return true
	}
	if resp == nil {
		return false
	}
	return containsSosanaMarker(resp.ActualCredentialName)
}

func maskProxyErrorResponse(resp *ProxyResponse) {
	if resp == nil {
		return
	}
	resp.Body = maskedUpstreamErrorBody(resp.StatusCode)
	if resp.Headers == nil {
		resp.Headers = http.Header{}
	}
	resp.Headers.Set("Content-Type", "application/json")
	resp.Headers.Del("Content-Encoding")
	resp.Headers.Del("Content-Length")
}

func containsSosanaMarker(value string) bool {
	value = strings.ToLower(value)
	return strings.Contains(value, "sosana") || strings.Contains(value, "sasana")
}
