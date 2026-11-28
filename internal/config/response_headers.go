package config

import (
	"fmt"
	"maps"
	"net/textproto"
	"slices"
	"strings"
)

func validateCredentialResponseHeaders(name string, provider ProviderType, headers map[string]string) error {
	if len(headers) == 0 {
		return nil
	}
	if provider.IsProxyLike() {
		return fmt.Errorf("credential %s: response_headers are only supported on direct providers", name)
	}
	seen := make(map[string]bool, len(headers))
	for _, key := range slices.Sorted(maps.Keys(headers)) {
		canonical := textproto.CanonicalMIMEHeaderKey(key)
		if !validHeaderFieldName(key) || !validHeaderFieldValue(headers[key]) {
			return fmt.Errorf("credential %s: invalid response_headers entry %q", name, key)
		}
		if seen[canonical] {
			return fmt.Errorf("credential %s: duplicate response_headers name %q", name, key)
		}
		seen[canonical] = true
		if reservedCredentialResponseHeader(canonical) {
			return fmt.Errorf("credential %s: response_headers cannot set managed header %q", name, key)
		}
	}
	return nil
}

func reservedCredentialResponseHeader(key string) bool {
	switch key {
	case "Connection", "Keep-Alive", "Proxy-Connection", "Proxy-Authenticate", "Proxy-Authorization",
		"Te", "Trailer", "Transfer-Encoding", "Upgrade", "X-Frame-Options", "X-Content-Type-Options",
		"Set-Cookie", "Www-Authenticate", "Location", "Digest", "Repr-Digest", "Etag", "Retry-After",
		"X-Credential-Name", "X-Request-Id", "X-Total-Cost", "X-Cost-Breakdown", "X-Accel-Buffering",
		"X-Aar-Proxy-Client":
		return true
	}
	return strings.HasPrefix(key, "Content-") || strings.HasPrefix(key, "Air-") || strings.HasPrefix(key, "Sec-Websocket-")
}
