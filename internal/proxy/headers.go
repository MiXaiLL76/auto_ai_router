package proxy

import (
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// hopByHopHeaders are headers that should not be proxied.
// These are hop-by-hop headers as defined in RFC 7230 Section 6.1.
// They are meant for single HTTP connection and must not be forwarded to the next hop.
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

// privacyHeaders are headers that reveal client IP or routing information.
// These are added by reverse proxies/load balancers and must not be forwarded
// to upstream AI providers to protect user privacy.
var privacyHeaders = map[string]bool{
	"X-Forwarded-For":     true,
	"X-Forwarded-Host":    true,
	"X-Forwarded-Port":    true,
	"X-Forwarded-Proto":   true,
	"X-Forwarded-Server":  true,
	"X-Real-Ip":           true,
	"X-Original-For":      true,
	"X-Client-Ip":         true,
	"X-Cluster-Client-Ip": true,
	"Forwarded":           true,
	"Via":                 true,
}

// isPrivacyHeader checks if a header reveals client IP or routing information.
func isPrivacyHeader(key string) bool {
	return privacyHeaders[key]
}

// isHopByHopHeader checks if a header should not be proxied.
// Returns true for hop-by-hop headers that must not be forwarded to upstream.
// RFC 7230: https://tools.ietf.org/html/rfc7230#section-6.1
func isHopByHopHeader(key string) bool {
	return hopByHopHeaders[key]
}

func isInternalAARRequestHeader(key string) bool {
	return strings.EqualFold(key, "X-Aar-Proxy-Client") ||
		strings.EqualFold(key, HeaderAARUsageAudioTokens)
}

// GetHopByHopHeaders returns a copy of the hop-by-hop headers map for reference.
// Use isHopByHopHeader() to check if a specific header should be filtered.
func GetHopByHopHeaders() map[string]bool {
	// Return a copy to prevent external modifications
	headers := make(map[string]bool)
	for k, v := range hopByHopHeaders {
		headers[k] = v
	}
	return headers
}

// copyRequestHeaders copies headers from source request to destination request,
// skipping hop-by-hop headers and optionally handling the Authorization header.
// Accept-Encoding is also skipped (see copyHeadersSkipAuth for rationale).
func copyRequestHeaders(dst *http.Request, src *http.Request, apiKey string) {
	for key, values := range src.Header {
		if isHopByHopHeader(key) || isPrivacyHeader(key) {
			continue
		}
		// Don't forward Accept-Encoding to upstream (proxy handles per-segment).
		if key == "Accept-Encoding" {
			continue
		}
		// Strip internal AIR markers — they must not be forwarded to the actual provider.
		if isInternalAARRequestHeader(key) {
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
// skipping hop-by-hop headers and Authorization header (Authorization will be set separately).
// Accept-Encoding is also skipped: the proxy handles encoding negotiation independently
// for each connection segment (client↔proxy and proxy↔upstream). Forwarding Accept-Encoding
// to upstream prevents Go's Transport from auto-decompressing responses, causing raw gzip
// bytes to flow through instead of decoded content.
func copyHeadersSkipAuth(dst *http.Request, src *http.Request) {
	for key, values := range src.Header {
		if isHopByHopHeader(key) || isPrivacyHeader(key) || key == "Authorization" {
			continue
		}
		// Don't forward Accept-Encoding: proxy manages compression per connection segment.
		// If forwarded, Go's Transport won't auto-decompress upstream gzip responses,
		// leading to raw gzip bytes being passed to converters (gzip parse errors).
		if key == "Accept-Encoding" {
			continue
		}
		// Strip internal AIR markers — they must not be forwarded to the actual provider.
		if isInternalAARRequestHeader(key) {
			continue
		}
		for _, value := range values {
			dst.Header.Add(key, value)
		}
	}
}

// copyResponseHeaders copies response headers to the response writer,
// skipping hop-by-hop headers and transformation-related headers.
// Note: Content-Encoding is always skipped because Go's http.Client automatically
// decompresses gzip/deflate responses, so the body is already decompressed.
// The caller should compress the body if needed and set Content-Encoding appropriately.
func copyResponseHeaders(w http.ResponseWriter, src http.Header, credType config.ProviderType) {
	for key, values := range src {
		if isHopByHopHeader(key) {
			continue
		}
		if strings.EqualFold(key, HeaderAARUsageAudioTokens) && credType != config.ProviderTypeProxy {
			continue
		}
		// Skip Content-Length and Content-Encoding for all response types
		// - Content-Length will be set based on actual body size
		// - Content-Encoding: Go's http.Client already decompressed the body,
		//   so we skip upstream's Content-Encoding header and let caller set it if recompressing
		// Skip X-Credential-Name — internal header for proxy-to-proxy routing, not exposed to end clients
		if key == "Content-Length" || key == "Content-Encoding" || key == "X-Credential-Name" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
}
