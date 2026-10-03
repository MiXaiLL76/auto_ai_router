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
	assert.Contains(t, err.Error(), "environment variable TEST_REQUEST_HEADER_NOT_SET is not set")
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

func TestCredentialRequestHeadersArePartOfProviderIdentity(t *testing.T) {
	cred := CredentialConfig{Name: "novita", Type: ProviderTypeOpenAI, RequestHeaders: map[string]string{"User-Agent": "a"}}
	same := cred
	same.RequestHeaders = map[string]string{"User-Agent": "a"}
	assert.True(t, cred.SameProviderIdentity(same))

	changed := cred
	changed.RequestHeaders = map[string]string{"User-Agent": "b"}
	assert.False(t, cred.SameProviderIdentity(changed))

	dropped := cred
	dropped.RequestHeaders = nil
	assert.False(t, cred.SameProviderIdentity(dropped))
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
