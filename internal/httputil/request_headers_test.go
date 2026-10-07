package httputil

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApplyCredentialRequestHeaders(t *testing.T) {
	header := http.Header{}
	header.Set("User-Agent", "Python-urllib/3.11")
	header.Set("X-Client-Hint", "hint")
	header.Set("X-Keep", "keep")
	header.Set("Authorization", "Bearer provider-key")

	ApplyCredentialRequestHeaders(header, &config.CredentialConfig{RequestHeaders: map[string]string{
		"User-Agent":    "auto-ai-router/1.0",
		"X-Client-Hint": "",
		"X-Added":       "added",
	}})

	assert.Equal(t, []string{"auto-ai-router/1.0"}, header.Values("User-Agent"))
	_, hasHint := header["X-Client-Hint"]
	assert.False(t, hasHint)
	assert.Equal(t, "added", header.Get("X-Added"))
	assert.Equal(t, "keep", header.Get("X-Keep"))
	assert.Equal(t, "Bearer provider-key", header.Get("Authorization"))
}

// The client may send a header several times; the configured value replaces all of them.
func TestApplyCredentialRequestHeadersReplacesEveryClientValue(t *testing.T) {
	header := http.Header{"X-Extra": {"one", "two"}, "X-Drop": {"one", "two"}}

	ApplyCredentialRequestHeaders(header, &config.CredentialConfig{RequestHeaders: map[string]string{
		"X-Extra": "configured",
		"X-Drop":  "",
	}})

	assert.Equal(t, []string{"configured"}, header.Values("X-Extra"))
	_, hasDrop := header["X-Drop"]
	assert.False(t, hasDrop)
}

// Credentials built in code skip the YAML trimming; a blank value still removes
// the header instead of sending an empty one.
func TestApplyCredentialRequestHeadersBlankValueRemoves(t *testing.T) {
	header := http.Header{"X-Client-Hint": {"hint"}, "User-Agent": {"Python-urllib/3.11"}}

	ApplyCredentialRequestHeaders(header, &config.CredentialConfig{RequestHeaders: map[string]string{
		"X-Client-Hint": " \t",
		"user-agent":    " ",
	}})

	_, hasHint := header["X-Client-Hint"]
	assert.False(t, hasHint)
	assert.Equal(t, []string{""}, header["User-Agent"])
}

func TestApplyCredentialRequestHeadersNoop(t *testing.T) {
	header := http.Header{"User-Agent": {"Python-urllib/3.11"}}
	ApplyCredentialRequestHeaders(header, nil)
	ApplyCredentialRequestHeaders(header, &config.CredentialConfig{})
	assert.Equal(t, http.Header{"User-Agent": {"Python-urllib/3.11"}}, header)
}

// The User-Agent is what a provider WAF actually sees, so check it on the wire for
// both protocols: an override replaces the client's value, and an empty value sends
// no User-Agent at all instead of net/http's Go-http-client default.
func TestApplyCredentialRequestHeadersOnTheWire(t *testing.T) {
	for _, proto := range []string{"HTTP/1.1", "HTTP/2.0"} {
		t.Run(proto, func(t *testing.T) {
			type seen struct {
				proto string
				ua    []string
				has   bool
			}
			got := make(chan seen, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				ua, has := r.Header["User-Agent"]
				got <- seen{proto: r.Proto, ua: ua, has: has}
			}))
			server.EnableHTTP2 = proto == "HTTP/2.0"
			server.StartTLS()
			defer server.Close()

			for _, tc := range []struct {
				name    string
				value   string
				wantUA  []string
				wantHas bool
			}{
				{name: "override", value: "auto-ai-router/1.0", wantUA: []string{"auto-ai-router/1.0"}, wantHas: true},
				{name: "remove", value: "", wantHas: false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					req, err := http.NewRequest(http.MethodPost, server.URL, nil)
					require.NoError(t, err)
					req.Header.Set("User-Agent", "Python-urllib/3.11")
					ApplyCredentialRequestHeaders(req.Header, &config.CredentialConfig{
						RequestHeaders: map[string]string{"User-Agent": tc.value},
					})
					resp, err := server.Client().Do(req)
					require.NoError(t, err)
					require.NoError(t, resp.Body.Close())

					s := <-got
					assert.Equal(t, proto, s.proto)
					assert.Equal(t, tc.wantHas, s.has, "User-Agent present: %q", s.ua)
					if tc.wantHas {
						assert.Equal(t, tc.wantUA, s.ua)
					}
				})
			}
		})
	}
}
