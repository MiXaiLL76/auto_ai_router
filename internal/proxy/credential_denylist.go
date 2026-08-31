package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

const (
	HeaderAIRCredentialDenylist = "Air-Credential-Denylist" //nolint:gosec // G101 header name

	maxCredentialDenylistEntries = 1024
	maxCredentialNameBytes       = 256
	maxCredentialDenylistBytes   = 64 * 1024
)

var errInvalidCredentialDenylist = errors.New("invalid AIR credential denylist")

type credentialDenylistContextKey struct{}

type credentialDenylistContext struct {
	markerPresent bool
	inboundValues []string
	effective     []string
}

func captureCredentialDenylist(r *http.Request, markerPresent bool) *http.Request {
	values := append([]string(nil), r.Header.Values(HeaderAIRCredentialDenylist)...)
	r.Header.Del(HeaderAIRCredentialDenylist)
	state := credentialDenylistContext{
		markerPresent: markerPresent,
		inboundValues: values,
	}
	return r.WithContext(context.WithValue(r.Context(), credentialDenylistContextKey{}, state))
}

func credentialDenylistState(ctx context.Context) credentialDenylistContext {
	state, _ := ctx.Value(credentialDenylistContextKey{}).(credentialDenylistContext)
	return state
}

func withEffectiveCredentialDenylist(r *http.Request, values []string) *http.Request {
	state := credentialDenylistState(r.Context())
	state.effective = append([]string(nil), values...)
	return r.WithContext(context.WithValue(r.Context(), credentialDenylistContextKey{}, state))
}

func effectiveCredentialDenylist(ctx context.Context) []string {
	state := credentialDenylistState(ctx)
	return append([]string(nil), state.effective...)
}

func trustedInboundCredentialDenylist(ctx context.Context, masterKeyAuthenticated bool) ([]string, error) {
	state := credentialDenylistState(ctx)
	if !state.markerPresent || !masterKeyAuthenticated || len(state.inboundValues) == 0 {
		return nil, nil
	}
	if len(state.inboundValues) != 1 {
		return nil, errInvalidCredentialDenylist
	}
	return parseCredentialDenylist(state.inboundValues[0])
}

func parseCredentialDenylist(value string) ([]string, error) {
	if value == "" || len(value) > maxCredentialDenylistBytes {
		return nil, errInvalidCredentialDenylist
	}
	decoder := json.NewDecoder(strings.NewReader(value))
	var names []string
	if err := decoder.Decode(&names); err != nil {
		return nil, errInvalidCredentialDenylist
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errInvalidCredentialDenylist
	}
	if len(names) > maxCredentialDenylistEntries {
		return nil, errInvalidCredentialDenylist
	}
	seen := make(map[string]struct{}, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		if !validCredentialDenylistName(name) {
			return nil, errInvalidCredentialDenylist
		}
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}
		result = append(result, name)
	}
	sort.Strings(result)
	return result, nil
}

func validCredentialDenylistName(name string) bool {
	if name == "" || strings.TrimSpace(name) != name || !utf8.ValidString(name) || len(name) > maxCredentialNameBytes {
		return false
	}
	for _, character := range name {
		if unicode.IsControl(character) {
			return false
		}
	}
	return true
}

func mergeCredentialDenylists(lists ...[]string) []string {
	seen := make(map[string]struct{})
	for _, list := range lists {
		for _, name := range list {
			seen[name] = struct{}{}
		}
	}
	result := make([]string, 0, len(seen))
	for name := range seen {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

func setCredentialDenylistHeader(header http.Header, denylist []string) error {
	if len(denylist) == 0 {
		return nil
	}
	if len(denylist) > maxCredentialDenylistEntries {
		return errInvalidCredentialDenylist
	}
	for _, name := range denylist {
		if !validCredentialDenylistName(name) {
			return errInvalidCredentialDenylist
		}
	}
	value, err := json.Marshal(denylist)
	if err != nil || len(value) > maxCredentialDenylistBytes || bytes.ContainsAny(value, "\r\n") {
		return errors.New("serialize AIR credential denylist")
	}
	header.Set(HeaderAIRCredentialDenylist, string(value))
	return nil
}

func carriesCredentialDenylist(credential *config.CredentialConfig) bool {
	if credential == nil {
		return false
	}
	if credential.Type == config.ProviderTypeAIR {
		return true
	}
	if credential.Type != config.ProviderTypeProxy {
		return false
	}
	parsed, err := url.Parse(credential.BaseURL)
	if err != nil || parsed.Scheme != "http" {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	if !strings.HasPrefix(host, "air-") {
		return false
	}
	labels := strings.Split(host, ".")
	return len(labels) == 1 ||
		len(labels) == 2 && labels[1] == "production" ||
		strings.HasSuffix(host, ".svc") ||
		strings.HasSuffix(host, ".svc.cluster.local")
}
