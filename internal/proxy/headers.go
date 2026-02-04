package proxy

import (
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// hopByHopHeaders are headers that should not be proxied
// See RFC 7230 Section 6.1
var hopByHopHeaders = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"TE":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
}

// isHopByHopHeader checks if a header should not be proxied
func isHopByHopHeader(key string) bool {
	return hopByHopHeaders[key]
}

// copyRequestHeaders copies headers from source request to destination request,
// skipping hop-by-hop headers and optionally handling the Authorization header
func copyRequestHeaders(dst *http.Request, src *http.Request, apiKey string) {
	for key, values := range src.Header {
		if isHopByHopHeader(key) {
			continue
		}
		if key == "Authorization" {
			// Handle Authorization header: use credential API key if available, otherwise copy original
			if apiKey != "" {
				dst.Header.Set("Authorization", "Bearer "+apiKey)
			} else {
				// Copy original Authorization header if no API key configured
				for _, value := range values {
					dst.Header.Add(key, value)
				}
			}
		} else {
			for _, value := range values {
				dst.Header.Add(key, value)
			}
		}
	}
}

// copyHeadersSkipAuth copies headers from source request to destination request,
// skipping hop-by-hop headers and Authorization header (Authorization will be set separately)
func copyHeadersSkipAuth(dst *http.Request, src *http.Request) {
	for key, values := range src.Header {
		if isHopByHopHeader(key) || key == "Authorization" {
			continue
		}
		for _, value := range values {
			dst.Header.Add(key, value)
		}
	}
}

// copyResponseHeaders copies response headers to the response writer,
// skipping hop-by-hop headers and optionally transformation-related headers
func copyResponseHeaders(w http.ResponseWriter, src http.Header, credType config.ProviderType) {
	for key, values := range src {
		if isHopByHopHeader(key) {
			continue
		}
		// Skip Content-Length and Content-Encoding as we may have transformed the response
		if credType == config.ProviderTypeVertexAI || credType == config.ProviderTypeAnthropic {
			if key == "Content-Length" || key == "Content-Encoding" {
				continue
			}
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
}
