package config

import (
	"fmt"
	"maps"
	"net/textproto"
	"os"
	"slices"
	"strings"
)

// reservedCredentialRequestHeaders are headers the router derives itself on every
// upstream request. Letting request_headers override them would break
// authentication, body framing, response decoding or AIR-to-AIR signalling, so
// such configs are rejected at load time instead of misbehaving at request time.
var reservedCredentialRequestHeaders = map[string]string{
	"Authorization":     "provider authentication comes from api_key/auth_type",
	"X-Api-Key":         "provider authentication comes from api_key/auth_type",
	"X-Goog-Api-Key":    "provider authentication comes from api_key",
	"Host":              "the host comes from base_url",
	"Transfer-Encoding": "it is derived from the request body",
	"Expect":            "the HTTP client manages request body transmission",
	"Accept-Encoding":   "the router negotiates compression per connection",
	"Anthropic-Beta":    "the router merges it from the client header, the request body and converter needs",
	"Anthropic-Version": "the router pins the Anthropic API version",
	// The router strips the client's Origin for Anthropic: it switches the API to
	// browser CORS authentication.
	"Origin":              "the router controls it per provider",
	"Connection":          "it is a hop-by-hop header",
	"Proxy-Connection":    "it is a hop-by-hop header",
	"Keep-Alive":          "it is a hop-by-hop header",
	"Proxy-Authenticate":  "it is a hop-by-hop header",
	"Proxy-Authorization": "it is a hop-by-hop header; put proxy credentials into proxy_url",
	"Te":                  "it is a hop-by-hop header",
	"Trailer":             "it is a hop-by-hop header",
	"Upgrade":             "it is a hop-by-hop header",
	// Legacy AIR marker; current internal AIR headers are covered by the Air- prefix.
	"X-Aar-Proxy-Client": "it is an internal AIR header",
}

var reservedCredentialRequestHeaderPrefixes = []struct {
	prefix string
	reason string
}{
	// Content-Type, Content-Length, Content-Encoding, ... describe the body the router builds.
	{prefix: "Content-", reason: "it is derived from the request body"},
	{prefix: "Air-", reason: "it is an internal AIR header"},
	{prefix: "Sec-Websocket-", reason: "it is a WebSocket handshake header"},
}

// parseCredentialRequestHeaders resolves os.environ/ values, validates the
// credential's request_headers and returns them keyed by canonical header name.
// Values are trimmed of surrounding spaces and tabs (net/http would trim them on
// the wire anyway), so a blank value means "do not send this header". An env
// reference that resolves to a blank value is therefore an error instead.
func parseCredentialRequestHeaders(credName string, credType ProviderType, raw map[string]string) (map[string]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	const envPrefix = "os.environ/"
	resolved := make(map[string]string, len(raw))
	for _, name := range slices.Sorted(maps.Keys(raw)) {
		value := raw[name]
		if envVar, ok := strings.CutPrefix(value, envPrefix); ok {
			// os.Getenv rather than resolveEnvString: an unset variable is reported
			// once, as the error below, instead of also as a warning.
			value = trimHeaderFieldValue(os.Getenv(envVar))
			if value == "" {
				return nil, fmt.Errorf("credential %s: request_headers[%s]: environment variable %s is not set or empty",
					credName, name, envVar)
			}
		}
		resolved[name] = trimHeaderFieldValue(value)
	}
	if err := validateCredentialRequestHeaders(credName, credType, resolved); err != nil {
		return nil, err
	}
	canonical := make(map[string]string, len(resolved))
	for name, value := range resolved {
		canonical[textproto.CanonicalMIMEHeaderKey(name)] = value
	}
	return canonical, nil
}

// validateCredentialRequestHeaders checks header names and values of a
// credential's request_headers. Values are never echoed in errors: they may hold
// secrets resolved from the environment.
func validateCredentialRequestHeaders(credName string, credType ProviderType, headers map[string]string) error {
	if len(headers) == 0 {
		return nil
	}
	if credType.IsProxyLike() {
		// A remote router treats them as client headers: it would forward them to
		// every provider behind it and log them unmasked.
		return fmt.Errorf("credential %s: request_headers are not supported for %s credentials; set them on the provider credential of the router that calls the provider", credName, credType)
	}
	seen := make(map[string]string, len(headers))
	for _, name := range slices.Sorted(maps.Keys(headers)) {
		if !validHeaderFieldName(name) {
			return fmt.Errorf("credential %s: request_headers: invalid header name %q", credName, name)
		}
		canonical := textproto.CanonicalMIMEHeaderKey(name)
		if previous, ok := seen[canonical]; ok {
			return fmt.Errorf("credential %s: request_headers: %q and %q name the same header", credName, previous, name)
		}
		seen[canonical] = name
		if reason, ok := reservedCredentialRequestHeaders[canonical]; ok {
			return fmt.Errorf("credential %s: request_headers cannot set %s: %s", credName, canonical, reason)
		}
		for _, reserved := range reservedCredentialRequestHeaderPrefixes {
			if strings.HasPrefix(canonical, reserved.prefix) {
				return fmt.Errorf("credential %s: request_headers cannot set %s: %s", credName, canonical, reserved.reason)
			}
		}
		if !validHeaderFieldValue(headers[name]) {
			return fmt.Errorf("credential %s: request_headers[%s]: value contains control characters", credName, canonical)
		}
	}
	return nil
}

// validHeaderFieldName reports whether name is an RFC 9110 token.
func validHeaderFieldName(name string) bool {
	if name == "" {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case 'a' <= c && c <= 'z', 'A' <= c && c <= 'Z', '0' <= c && c <= '9':
		case strings.IndexByte("!#$%&'*+-.^_`|~", c) >= 0:
		default:
			return false
		}
	}
	return true
}

// validHeaderFieldValue mirrors net/http's check: no control characters other
// than horizontal tab, so a value can never split into extra header lines.
func validHeaderFieldValue(value string) bool {
	for i := 0; i < len(value); i++ {
		c := value[i]
		if (c < ' ' && c != '\t') || c == 0x7f {
			return false
		}
	}
	return true
}

// trimHeaderFieldValue drops optional whitespace around a field value (RFC 9110
// section 5.5): spaces and horizontal tabs only.
func trimHeaderFieldValue(value string) string {
	return strings.Trim(value, " \t")
}
