package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// Novita's Cloudflare answers any request whose User-Agent starts with
// "Python-urllib" with 403 "error code: 1010" before authentication, while the
// same body from the OpenAI SDK passes. The router used to forward the client's
// User-Agent verbatim; request_headers lets such a credential send its own.
const blockedClientUserAgent = "Python-urllib/3.11"

var novitaLikeRequestHeaders = map[string]string{
	"User-Agent":    "auto-ai-router/1.0",
	"X-Extra":       "extra",
	"X-Client-Hint": "",
}

type requestHeadersCase struct {
	provider     config.ProviderType
	path         string
	body         string
	response     string
	wantAuthName string
	wantAuth     string
}

var requestHeadersCases = []requestHeadersCase{
	{
		provider: config.ProviderTypeOpenAI, path: "/v1/chat/completions",
		body:         `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
		response:     `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		wantAuthName: "Authorization", wantAuth: "Bearer provider-key",
	},
	{
		provider: config.ProviderTypeProxy, path: "/v1/chat/completions",
		body:         `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
		response:     `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		wantAuthName: "Authorization", wantAuth: "Bearer provider-key",
	},
	{
		provider: config.ProviderTypeAIR, path: "/v1/chat/completions",
		body:         `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
		response:     `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		wantAuthName: "Authorization", wantAuth: "Bearer provider-key",
	},
	{
		provider: config.ProviderTypeAnthropic, path: "/v1/messages",
		body:         `{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`,
		response:     `{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		wantAuthName: "X-Api-Key", wantAuth: "provider-key",
	},
	{
		provider: config.ProviderTypeGemini, path: "/v1/chat/completions",
		body:         `{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}]}`,
		response:     `{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
		wantAuthName: "X-Goog-Api-Key", wantAuth: "provider-key",
	},
}

func newRequestHeadersClientRequest(path, body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", blockedClientUserAgent)
	req.Header.Set("X-Client-Hint", "from-client")
	return req
}

func TestProxyRequestAppliesCredentialRequestHeaders(t *testing.T) {
	for _, tc := range requestHeadersCases {
		if tc.provider.IsProxyLike() {
			continue // config rejects request_headers on air/proxy credentials
		}
		t.Run(string(tc.provider), func(t *testing.T) {
			captured := make(chan http.Header, 1)
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					_, _ = io.WriteString(w, `{}`) // remote discovery of air/proxy credentials
					return
				}
				captured <- r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
				Name: "novita", Type: tc.provider, BaseURL: upstream.URL, APIKey: "provider-key",
				RPM: -1, TPM: -1, RequestHeaders: novitaLikeRequestHeaders,
			}).Build()
			req := newRequestHeadersClientRequest(tc.path, tc.body)
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			outbound := <-captured
			assert.Equal(t, []string{"auto-ai-router/1.0"}, outbound.Values("User-Agent"))
			assert.Equal(t, "extra", outbound.Get("X-Extra"))
			assert.Empty(t, outbound.Values("X-Client-Hint"))
			assert.Equal(t, tc.wantAuth, outbound.Get(tc.wantAuthName))
			// The incoming request is shared across retries and must stay untouched.
			assert.Equal(t, blockedClientUserAgent, req.Header.Get("User-Agent"))
			assert.Equal(t, "from-client", req.Header.Get("X-Client-Hint"))
		})
	}
}

// Credentials without request_headers keep today's behavior: the client's
// User-Agent and custom headers reach the provider unchanged.
func TestProxyRequestWithoutCredentialRequestHeadersForwardsClientHeaders(t *testing.T) {
	for _, tc := range requestHeadersCases {
		t.Run(string(tc.provider), func(t *testing.T) {
			captured := make(chan http.Header, 1)
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					_, _ = io.WriteString(w, `{}`) // remote discovery of air/proxy credentials
					return
				}
				captured <- r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
				Name: "plain", Type: tc.provider, BaseURL: upstream.URL, APIKey: "provider-key", RPM: -1, TPM: -1,
			}).Build()
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, newRequestHeadersClientRequest(tc.path, tc.body))

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			outbound := <-captured
			assert.Equal(t, blockedClientUserAgent, outbound.Get("User-Agent"))
			assert.Equal(t, "from-client", outbound.Get("X-Client-Hint"))
			assert.Empty(t, outbound.Get("X-Extra"))
		})
	}
}

func TestProxyRequestStreamingAppliesCredentialRequestHeaders(t *testing.T) {
	captured := make(chan http.Header, 1)
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- r.Header.Clone()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"+
			"data: [DONE]\n\n")
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "novita", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "provider-key",
		RPM: -1, TPM: -1, RequestHeaders: map[string]string{"User-Agent": "auto-ai-router/1.0"},
	}).Build()
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions",
		`{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Contains(t, w.Body.String(), `"content":"OK"`)
	assert.Contains(t, w.Body.String(), "[DONE]")
	assert.Equal(t, "auto-ai-router/1.0", (<-captured).Get("User-Agent"))
}

// An empty User-Agent value sends none at all: not the client's, and not
// net/http's Go-http-client default either.
func TestProxyRequestCredentialRequestHeadersCanDropUserAgent(t *testing.T) {
	type seen struct {
		ua  []string
		has bool
	}
	captured := make(chan seen, 1)
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ua, has := r.Header["User-Agent"]
		captured <- seen{ua: ua, has: has}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, requestHeadersCases[0].response)
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "novita", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "provider-key",
		RPM: -1, TPM: -1, RequestHeaders: map[string]string{"User-Agent": ""},
	}).Build()
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	got := <-captured
	assert.False(t, got.has, "unexpected User-Agent %q", got.ua)
}

// Headers of one credential must not leak into a retry on another credential.
func TestProxyRequestRetryDoesNotCarryCredentialRequestHeaders(t *testing.T) {
	primaryHeaders := make(chan http.Header, 1)
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHeaders <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"rate limit exceeded","type":"rate_limit_error","code":"rate_limited"}}`)
	}))
	defer primary.Close()
	secondaryHeaders := make(chan http.Header, 1)
	secondary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondaryHeaders <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, requestHeadersCases[0].response)
	}))
	defer secondary.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "novita", Type: config.ProviderTypeOpenAI, BaseURL: primary.URL, APIKey: "primary-key",
			RPM: 100, TPM: 10000, RequestHeaders: novitaLikeRequestHeaders,
		},
		config.CredentialConfig{
			Name: "secondary", Type: config.ProviderTypeOpenAI, BaseURL: secondary.URL, APIKey: "secondary-key",
			Priority: 1, RPM: 100, TPM: 10000,
		},
	).WithMaxProviderRetries(3).Build()
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	first := <-primaryHeaders
	assert.Equal(t, "auto-ai-router/1.0", first.Get("User-Agent"))
	assert.Equal(t, "extra", first.Get("X-Extra"))
	second := <-secondaryHeaders
	assert.Equal(t, blockedClientUserAgent, second.Get("User-Agent"))
	assert.Equal(t, "from-client", second.Get("X-Client-Hint"))
	assert.Empty(t, second.Get("X-Extra"))
	assert.Equal(t, "Bearer secondary-key", second.Get("Authorization"))
}

func TestNativeWebSocketHandshakeAppliesCredentialRequestHeaders(t *testing.T) {
	captured := make(chan http.Header, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- r.Header.Clone()
		http.Error(w, "stop after handshake", http.StatusBadGateway)
	}))
	defer upstream.Close()
	cred := config.CredentialConfig{
		Name: "upstream", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "upstream-key",
		RPM: 100, TPM: 10000, RequestHeaders: map[string]string{"User-Agent": "auto-ai-router/1.0", "X-Extra": "extra"},
	}
	prx := NewTestProxyBuilder().WithCredentials(cred).Build()
	prx.modelManager = models.New(prx.logger, 100, []config.ModelRPMConfig{{Name: "gpt-6-astra", Credential: "upstream", WebSocketResponses: true}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{cred})
	prx.LiteLLMDB = &nativeWSTestDB{clientAuthTestDB: &clientAuthTestDB{}, entries: make(chan *dbmodels.SpendLogEntry, 1)}
	setTestModelPrice(prx, "gpt-6-astra", &models.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	defer server.Close()
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/v1/responses", http.Header{
		"Authorization": {"Bearer master-key"}, "User-Agent": {blockedClientUserAgent},
	})
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	wsWrite(t, conn, `{"type":"response.create","model":"gpt-6-astra","input":"hi"}`)

	select {
	case handshake := <-captured:
		assert.Equal(t, "auto-ai-router/1.0", handshake.Get("User-Agent"))
		assert.Equal(t, "extra", handshake.Get("X-Extra"))
		assert.Equal(t, "Bearer upstream-key", handshake.Get("Authorization"))
	case <-time.After(5 * time.Second):
		t.Fatal("no upstream handshake")
	}
}

// config rejects request_headers that would override internal AIR markers. The
// marker names live in this package, so keep the two lists in sync here.
func TestCredentialRequestHeadersCannotOverrideInternalAIRHeaders(t *testing.T) {
	for _, name := range []string{HeaderAIRProxyClient, HeaderLegacyAIRProxyClient, HeaderAIRCredentialDenylist, HeaderAIRUsageAudioTokens} {
		t.Run(name, func(t *testing.T) {
			var cred config.CredentialConfig
			err := yaml.Unmarshal([]byte("name: test\ntype: openai\nbase_url: https://api.example\nrequest_headers:\n  "+name+": x\n"), &cred)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "internal AIR header")
		})
	}
}
