package proxy

import (
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
)

const accelBufferingHeader = "X-Accel-Buffering"

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

var allowedResponseHeaders = map[string]bool{
	"Cache-Control":                  true,
	"Content-Disposition":            true,
	"Content-Range":                  true,
	"Content-Type":                   true,
	"Last-Modified":                  true,
	"Location":                       true,
	"Retry-After":                    true,
	"X-Ratelimit-Limit-Requests":     true,
	"X-Ratelimit-Limit-Tokens":       true,
	"X-Ratelimit-Remaining-Requests": true,
	"X-Ratelimit-Remaining-Tokens":   true,
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
	return privacyHeaders[http.CanonicalHeaderKey(key)]
}

// isHopByHopHeader checks if a header should not be proxied.
// Returns true for hop-by-hop headers that must not be forwarded to upstream.
// RFC 7230: https://tools.ietf.org/html/rfc7230#section-6.1
func isHopByHopHeader(key string) bool {
	return hopByHopHeaders[http.CanonicalHeaderKey(key)]
}

func isInternalAIRRequestHeader(key string) bool {
	return strings.EqualFold(key, HeaderAIRProxyClient) ||
		strings.EqualFold(key, HeaderLegacyAIRProxyClient) ||
		strings.EqualFold(key, HeaderAIRUsageAudioTokens)
}

func isProxyOwnedResponseHeader(key string) bool {
	return strings.EqualFold(key, accelBufferingHeader)
}

func isClientAuthHeader(key string) bool {
	return strings.EqualFold(key, "Authorization") || strings.EqualFold(key, "X-Api-Key")
}

func setSuccessfulSSEHeaders(headers http.Header, statusCode int) {
	if headers == nil || statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return
	}
	headers.Set(accelBufferingHeader, "no")
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
		if isHopByHopHeader(key) || isPrivacyHeader(key) || isClientAuthHeader(key) {
			continue
		}
		// Don't forward Accept-Encoding to upstream (proxy handles per-segment).
		if key == "Accept-Encoding" {
			continue
		}
		// Strip internal AIR markers — they must not be forwarded to the actual provider.
		if isInternalAIRRequestHeader(key) {
			continue
		}
		for _, value := range values {
			dst.Header.Add(key, value)
		}
	}
	if apiKey != "" {
		dst.Header.Set("Authorization", "Bearer "+apiKey)
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
		if isHopByHopHeader(key) || isPrivacyHeader(key) || isClientAuthHeader(key) {
			continue
		}
		// Don't forward Accept-Encoding: proxy manages compression per connection segment.
		// If forwarded, Go's Transport won't auto-decompress upstream gzip responses,
		// leading to raw gzip bytes being passed to converters (gzip parse errors).
		if key == "Accept-Encoding" {
			continue
		}
		// Strip internal AIR markers — they must not be forwarded to the actual provider.
		if isInternalAIRRequestHeader(key) {
			continue
		}
		for _, value := range values {
			dst.Header.Add(key, value)
		}
	}
}

func mergeAnthropicBetaHeader(header http.Header, betas []string) {
	if len(betas) == 0 {
		return
	}
	values := make([]string, 0, len(header.Values("anthropic-beta"))+len(betas))
	values = append(values, header.Values("anthropic-beta")...)
	values = append(values, betas...)
	seen := make(map[string]struct{}, len(values))
	merged := make([]string, 0, len(values))
	for _, value := range values {
		for beta := range strings.SplitSeq(value, ",") {
			beta = strings.TrimSpace(beta)
			if beta == "" {
				continue
			}
			if _, ok := seen[beta]; ok {
				continue
			}
			seen[beta] = struct{}{}
			merged = append(merged, beta)
		}
	}
	if len(merged) > 0 {
		header.Set("anthropic-beta", strings.Join(merged, ","))
	}
}

// copyResponseHeaders copies response headers to the response writer,
// skipping hop-by-hop headers and transformation-related headers.
// Note: Content-Encoding is always skipped because Go's http.Client automatically
// decompresses gzip/deflate responses, so the body is already decompressed.
// The caller should compress the body if needed and set Content-Encoding appropriately.
func (p *Proxy) copyResponseHeaders(w http.ResponseWriter, src http.Header, cred *config.CredentialConfig, stripIntegrityHeaders bool) {
	for key, values := range src {
		canonicalKey := http.CanonicalHeaderKey(key)
		if p.shouldSkipResponseHeaderForClient(canonicalKey, cred) {
			continue
		}
		if stripIntegrityHeaders && isRepresentationIntegrityHeader(canonicalKey) {
			continue
		}
		if strings.EqualFold(canonicalKey, HeaderAIRUsageAudioTokens) && w.Header().Get(HeaderAIRUsageAudioTokens) != "" {
			continue
		}
		for _, value := range values {
			w.Header().Add(canonicalKey, value)
		}
	}
}

func (p *Proxy) setCredentialResponseHeader(w http.ResponseWriter, logCtx *RequestLogContext, credentialName string) {
	if credentialName == "" && logCtx != nil {
		credentialName = logCtx.ActualCredentialName
	}
	if p.responseHeaderMode == config.ResponseHeaderModeAllowlist || logCtx == nil || !logCtx.IsProxyRequest || credentialName == "" {
		w.Header().Del("X-Credential-Name")
		return
	}
	w.Header().Set("X-Credential-Name", credentialName)
}

func (p *Proxy) shouldSkipResponseHeaderForClient(key string, cred *config.CredentialConfig) bool {
	canonical := http.CanonicalHeaderKey(key)
	if p.responseHeaderMode == config.ResponseHeaderModeAllowlist && !allowedResponseHeaders[canonical] {
		return true
	}
	return shouldSkipResponseHeaderForClient(canonical, cred)
}

func shouldSkipResponseHeaderForClient(key string, cred *config.CredentialConfig) bool {
	canonical := http.CanonicalHeaderKey(key)
	if isHopByHopHeader(canonical) || isProxyOwnedResponseHeader(canonical) {
		return true
	}
	// Skip Content-Length and Content-Encoding for all response types:
	// Content-Length is recalculated after conversion/compression, and Go's client
	// has already decompressed gzip/deflate bodies when Content-Encoding is present.
	switch canonical {
	case "Content-Length", "Content-Encoding", "Transfer-Encoding", "X-Credential-Name":
		return true
	}
	if strings.EqualFold(canonical, HeaderAIRUsageAudioTokens) && (cred == nil || !cred.IsProxyLike()) {
		return true
	}
	if promanutils.ShouldSanitizeUpstreamSurface(cred) && promanutils.IsProviderInternalResponseHeader(canonical) {
		return true
	}
	return false
}

func dropRepresentationIntegrityHeaders(headers http.Header) {
	for key := range headers {
		if isRepresentationIntegrityHeader(key) {
			headers.Del(key)
		}
	}
}
