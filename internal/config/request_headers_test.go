package config

import (
	"bytes"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestCredentialRequestHeadersParse(t *testing.T) {
	t.Setenv("TEST_PROVIDER_HEADER_SECRET", "s3cr3t")
	var cred CredentialConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
name: novita
type: openai
base_url: https://api.novita.ai/openai
request_headers:
  user-agent: auto-ai-router/1.0
  x-provider-token: os.environ/TEST_PROVIDER_HEADER_SECRET
  X-Client-Hint: ""
  X-Null:
`), &cred))

	assert.Equal(t, map[string]string{
		"User-Agent":       "auto-ai-router/1.0",
		"X-Provider-Token": "s3cr3t",
		"X-Client-Hint":    "",
		"X-Null":           "",
	}, cred.RequestHeaders)

	data, err := yaml.Marshal(cred)
	require.NoError(t, err)
	var restored CredentialConfig
	require.NoError(t, yaml.Unmarshal(data, &restored))
	assert.Equal(t, cred.RequestHeaders, restored.RequestHeaders)
}

// Surrounding whitespace never reaches the wire (net/http trims it), so it is
// dropped at load time and a blank value means "remove", like an empty one.
func TestCredentialRequestHeadersTrimValues(t *testing.T) {
	t.Setenv("TEST_PROVIDER_HEADER_PADDED", "  s3cr3t\t")
	var cred CredentialConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
name: novita
type: openai
request_headers:
  User-Agent: "  auto-ai-router/1.0  "
  X-Inner: "a \t b"
  X-Blank: "  \t "
  X-Token: os.environ/TEST_PROVIDER_HEADER_PADDED
`), &cred))
	assert.Equal(t, map[string]string{
		"User-Agent": "auto-ai-router/1.0",
		"X-Inner":    "a \t b",
		"X-Blank":    "",
		"X-Token":    "s3cr3t",
	}, cred.RequestHeaders)
}

// Ordinary request headers a provider WAF or gateway may look at stay allowed.
func TestCredentialRequestHeadersAllowed(t *testing.T) {
	var cred CredentialConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
name: novita
type: vllm
request_headers:
  Accept: application/json
  Referer: https://router.example
  X-Title: router
  Cookie: a=b
  X-Request-Source: "\u00e9t\u00e9"
`), &cred))
	assert.Len(t, cred.RequestHeaders, 5)
}

func TestCredentialRequestHeadersAbsent(t *testing.T) {
	var cred CredentialConfig
	require.NoError(t, yaml.Unmarshal([]byte("name: plain\ntype: openai\nbase_url: https://api.openai.com/v1"), &cred))
	assert.Nil(t, cred.RequestHeaders)
}

func TestCredentialRequestHeadersRejected(t *testing.T) {
	tests := []struct {
		name    string
		headers string
		wantErr string
	}{
		{name: "authorization", headers: "Authorization: Bearer x", wantErr: "cannot set Authorization"},
		{name: "api key lowercase", headers: "x-api-key: x", wantErr: "cannot set X-Api-Key"},
		{name: "google api key", headers: "X-Goog-Api-Key: x", wantErr: "cannot set X-Goog-Api-Key"},
		{name: "host", headers: "Host: example.com", wantErr: "cannot set Host"},
		{name: "content type", headers: "Content-Type: text/plain", wantErr: "cannot set Content-Type"},
		{name: "content length", headers: "Content-Length: 1", wantErr: "cannot set Content-Length"},
		{name: "content encoding", headers: "Content-Encoding: gzip", wantErr: "cannot set Content-Encoding"},
		{name: "any content header", headers: "content-language: en", wantErr: "cannot set Content-Language"},
		{name: "transfer encoding", headers: "Transfer-Encoding: chunked", wantErr: "cannot set Transfer-Encoding"},
		{name: "expect", headers: "Expect: 100-continue", wantErr: "cannot set Expect"},
		{name: "origin", headers: "Origin: https://example.com", wantErr: "cannot set Origin"},
		{name: "proxy connection", headers: "Proxy-Connection: keep-alive", wantErr: "cannot set Proxy-Connection"},
		{name: "upgrade", headers: "Upgrade: websocket", wantErr: "cannot set Upgrade"},
		{name: "accept encoding", headers: "Accept-Encoding: gzip", wantErr: "cannot set Accept-Encoding"},
		{name: "anthropic beta", headers: "anthropic-beta: context-1m-2025-08-07", wantErr: "cannot set Anthropic-Beta"},
		{name: "anthropic version", headers: "Anthropic-Version: 2023-01-01", wantErr: "cannot set Anthropic-Version"},
		{name: "hop by hop", headers: "Connection: close", wantErr: "cannot set Connection"},
		{name: "te", headers: "TE: trailers", wantErr: "cannot set Te"},
		{name: "proxy authorization", headers: "Proxy-Authorization: Basic x", wantErr: "put proxy credentials into proxy_url"},
		{name: "internal air header", headers: "Air-Proxy-Client: 1", wantErr: "internal AIR header"},
		{name: "future internal air header", headers: "air-something-new: 1", wantErr: "internal AIR header"},
		{name: "legacy air marker", headers: "X-Aar-Proxy-Client: 1", wantErr: "internal AIR header"},
		{name: "websocket handshake", headers: "Sec-WebSocket-Key: x", wantErr: "WebSocket handshake header"},
		{name: "name with space", headers: "\"Bad Header\": x", wantErr: "invalid header name"},
		{name: "name with colon", headers: "\"X-A:B\": x", wantErr: "invalid header name"},
		{name: "empty name", headers: "\"\": x", wantErr: "invalid header name"},
		{name: "value with newline", headers: "X-Injected: \"a\\r\\nX-Evil: 1\"", wantErr: "value contains control characters"},
		{name: "value with bare lf", headers: "X-Injected: \"a\\nb\"", wantErr: "value contains control characters"},
		{name: "value with nul", headers: "X-Injected: \"a\\0b\"", wantErr: "value contains control characters"},
		{name: "value with del", headers: "X-Injected: \"a\\x7Fb\"", wantErr: "value contains control characters"},
		{name: "non-ascii name", headers: "X-Заголовок: x", wantErr: "invalid header name"},
		{name: "same header twice", headers: "User-Agent: a\n  user-agent: b", wantErr: "name the same header"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cred CredentialConfig
			err := yaml.Unmarshal([]byte("name: test\ntype: openai\nrequest_headers:\n  "+tt.headers+"\n"), &cred)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "credential test:")
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// A remote router would forward the headers to every provider behind it as
// client headers and log them, so they belong on the leaf provider credential.
func TestCredentialRequestHeadersRejectedOnProxyLikeCredentials(t *testing.T) {
	for _, credType := range []string{"air", "proxy"} {
		t.Run(credType, func(t *testing.T) {
			var cred CredentialConfig
			err := yaml.Unmarshal([]byte("name: ger01\ntype: "+credType+"\nbase_url: http://air-ger01/v1\nrequest_headers:\n  User-Agent: auto-ai-router/1.0\n"), &cred)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "credential ger01: request_headers are not supported for "+credType+" credentials")

			require.NoError(t, yaml.Unmarshal([]byte("name: ger01\ntype: "+credType+"\nbase_url: http://air-ger01/v1\nrequest_headers: {}\n"), &cred))
		})
	}
}

func TestCredentialRequestHeadersUnsetEnvIsAnError(t *testing.T) {
	// An unresolved secret must not silently turn into "remove this header".
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte("name: test\ntype: openai\nrequest_headers:\n  X-Provider-Token: os.environ/TEST_REQUEST_HEADER_NOT_SET\n"), &cred)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "environment variable TEST_REQUEST_HEADER_NOT_SET is not set or empty")
}

// A blank variable would otherwise turn into "remove this header" after trimming.
func TestCredentialRequestHeadersBlankEnvIsAnError(t *testing.T) {
	t.Setenv("TEST_REQUEST_HEADER_BLANK", " \t ")
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte("name: test\ntype: openai\nrequest_headers:\n  X-Provider-Token: os.environ/TEST_REQUEST_HEADER_BLANK\n"), &cred)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "environment variable TEST_REQUEST_HEADER_BLANK is not set or empty")
}

// The unset variable is reported once, by the returned error, not also by
// resolveEnvString's generic warning.
func TestCredentialRequestHeadersUnsetEnvDoesNotWarn(t *testing.T) {
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })

	var cred CredentialConfig
	require.Error(t, yaml.Unmarshal([]byte("name: test\ntype: openai\nrequest_headers:\n  X-Provider-Token: os.environ/TEST_REQUEST_HEADER_NOT_SET\n"), &cred))
	assert.NotContains(t, buf.String(), "TEST_REQUEST_HEADER_NOT_SET")
}

func TestCredentialRequestHeadersErrorsDoNotEchoValues(t *testing.T) {
	t.Setenv("TEST_PROVIDER_HEADER_SECRET", "s3cr3t\nX-Evil: 1")
	var cred CredentialConfig
	err := yaml.Unmarshal([]byte("name: test\ntype: openai\nrequest_headers:\n  X-Provider-Token: os.environ/TEST_PROVIDER_HEADER_SECRET\n"), &cred)
	require.Error(t, err)
	assert.NotContains(t, err.Error(), "s3cr3t")
}

func TestConfigValidateRejectsBadCredentialRequestHeaders(t *testing.T) {
	newConfig := func(headers map[string]string) *Config {
		return &Config{
			Server: ServerConfig{Port: 8080, MaxBodySizeMB: 10, MasterKey: "test-key", RequestTimeout: 30 * time.Second},
			Credentials: []CredentialConfig{{
				Name: "novita", Type: ProviderTypeOpenAI, APIKey: "k", BaseURL: "https://api.novita.ai/openai",
				RPM: -1, TPM: -1, RequestHeaders: headers,
			}},
			Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
		}
	}

	require.NoError(t, newConfig(map[string]string{"User-Agent": "auto-ai-router/1.0"}).Validate())
	require.NoError(t, newConfig(nil).Validate())

	err := newConfig(map[string]string{"authorization": "Bearer x"}).Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "credential novita: request_headers cannot set Authorization")

	err = newConfig(map[string]string{"User-Agent": "a", "user-agent": "b"}).Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "name the same header")

	air := newConfig(map[string]string{"User-Agent": "auto-ai-router/1.0"})
	air.Credentials[0].Type = ProviderTypeAIR
	err = air.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "request_headers are not supported for air credentials")
}

// Changing a direct credential's request_headers on hot reload must not drop
// learned provider metadata: they do not change which provider it talks to.
func TestCredentialRequestHeadersAreNotPartOfProviderIdentity(t *testing.T) {
	cred := CredentialConfig{Name: "novita", Type: ProviderTypeOpenAI, RequestHeaders: map[string]string{"User-Agent": "a"}}

	changed := cred
	changed.RequestHeaders = map[string]string{"User-Agent": "b"}
	assert.True(t, cred.SameProviderIdentity(changed))

	dropped := cred
	dropped.RequestHeaders = nil
	assert.True(t, cred.SameProviderIdentity(dropped))
}

func TestPrintConfigLogsRequestHeaderNamesOnly(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))
	PrintConfig(logger, &Config{
		Server: ServerConfig{Port: 8080, MaxBodySizeMB: 10, MasterKey: "test-key", RequestTimeout: 30 * time.Second},
		Credentials: []CredentialConfig{{
			Name: "novita", Type: ProviderTypeOpenAI, BaseURL: "https://api.novita.ai/openai", RPM: -1, TPM: -1,
			RequestHeaders: map[string]string{"User-Agent": "auto-ai-router/1.0", "X-Provider-Token": "s3cr3t"},
		}},
	})

	out := buf.String()
	assert.Contains(t, out, `"request_headers":["User-Agent","X-Provider-Token"]`)
	assert.NotContains(t, out, "s3cr3t")
	assert.NotContains(t, out, "auto-ai-router/1.0")
}
