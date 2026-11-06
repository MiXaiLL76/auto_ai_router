package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	// The native Vertex Responses converter, registered as cmd/server registers it:
	// without it /v1/responses would go through Chat Completions, not the production path.
	_ "github.com/mixaill76/auto_ai_router/internal/converter/vertex/responses"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// argonTestPrice is a gemini-4-argon price entry.
func argonTestPrice() *pricing.ModelPrice {
	perQuery := 0.0182
	return &pricing.ModelPrice{
		InputCostPerToken:       0.0000026,
		OutputCostPerToken:      0.000013,
		CacheReadInputTokenCost: 0.00000013,
		SearchContextCostPerQuery: map[string]float64{
			"search_context_size_low": perQuery, "search_context_size_medium": perQuery, "search_context_size_high": perQuery,
		},
		WebSearchBillingUnit: "per_query",
	}
}

// longDecodeUpstream serves Vertex AI generateContent for gemini-4-argon: answer
// gets the request number (0 for the first) and the decoded request body.
type longDecodeUpstream struct {
	t        *testing.T
	server   *httptest.Server
	mu       sync.Mutex
	requests []map[string]any
	paths    []string
	auth     []string
}

func newLongDecodeUpstream(t *testing.T, answer func(n int, req map[string]any, w http.ResponseWriter)) *longDecodeUpstream {
	t.Helper()
	u := &longDecodeUpstream{t: t}
	u.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"access_token":"vertex-token","token_type":"Bearer","expires_in":3600}`))
			return
		}
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		var req map[string]any
		require.NoError(t, json.Unmarshal(raw, &req))
		u.mu.Lock()
		n := len(u.requests)
		u.requests = append(u.requests, req)
		u.paths = append(u.paths, r.URL.Path)
		u.auth = append(u.auth, r.Header.Get("Authorization"))
		u.mu.Unlock()
		answer(n, req, w)
	}))
	t.Cleanup(u.server.Close)
	return u
}

func (u *longDecodeUpstream) proxy() (*Proxy, *stubLiteLLMManager) {
	target, err := url.Parse(u.server.URL)
	require.NoError(u.t, err)
	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:            "grant-global",
			Type:            config.ProviderTypeVertexAI,
			ProjectID:       "grant",
			Location:        "global",
			CredentialsJSON: fakeServiceAccountJSON(u.t, u.server.URL+"/token"),
			RPM:             -1,
			TPM:             -1,
		}).
		WithMasterKey("master-key").
		Build()
	prx.client = &http.Client{Transport: rewriteHostTransport{target: target}}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gemini-4-argon", argonTestPrice())
	return prx, dbStub
}

func postJSON(prx *Proxy, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// The two requests of one long decode: the first thinks through its whole output
// cap and stops with CONTINUATION, the second answers. Both search once.
const (
	argonFirstSegment = `{"candidates":[{"content":{"role":"model","parts":[{"text":"long plan","thought":true}]},` +
		`"finishReason":"CONTINUATION","continuationToken":"dG9rLTE=","groundingMetadata":{"webSearchQueries":["argon"]}}],` +
		`"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":32768,"totalTokenCount":33768},"modelVersion":"gemini-4-argon"}`
	argonSecondSegment = `{"candidates":[{"content":{"role":"model","parts":[{"text":"The answer is 42."}]},` +
		`"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["argon"]}}],` +
		`"usageMetadata":{"promptTokenCount":33768,"cachedContentTokenCount":33000,"candidatesTokenCount":500,"totalTokenCount":34268},"modelVersion":"gemini-4-argon"}`
	// 1768 uncached + 33000 cached input, 500 answer + 32768 thinking, 2 queries.
	argonTwoSegmentSpend = 1768*0.0000026 + 33000*0.00000013 + 500*0.000013 + 32768*0.000013 + 2*0.0182
)

const argonChatBody = `{"model": "gemini-4-argon", "max_tokens": 200000, "reasoning_effort": "high",
	"tools": [{"type": "web_search"}],
	"messages": [{"role": "system", "content": "Be exact."}, {"role": "user", "content": "Prove it."}]}`

func TestProxyRequest_GeminiLongDecodeContinuesNonStreamed(t *testing.T) {
	upstream := newLongDecodeUpstream(t, func(n int, _ map[string]any, w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		if n == 0 {
			_, _ = w.Write([]byte(argonFirstSegment))
			return
		}
		_, _ = w.Write([]byte(argonSecondSegment))
	})
	prx, dbStub := upstream.proxy()

	w := postJSON(prx, "/v1/chat/completions", argonChatBody)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	require.Len(t, upstream.requests, 2)
	for _, path := range upstream.paths {
		assert.Equal(t, "/v1beta1/projects/grant/locations/global/publishers/google/models/gemini-4-argon:generateContent", path)
	}
	assert.Equal(t, []string{"Bearer vertex-token", "Bearer vertex-token"}, upstream.auth)
	first, second := upstream.requests[0], upstream.requests[1]
	assert.NotContains(t, first, "continuationToken")
	assert.Equal(t, "dG9rLTE=", second["continuationToken"])
	delete(second, "continuationToken")
	assert.Equal(t, first, second, "the same contents, system instruction, tools and maxOutputTokens, no partial answer")
	assert.Equal(t, float64(200000), first["generationConfig"].(map[string]any)["maxOutputTokens"])

	var resp struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens        int `json:"cached_tokens"`
				InitialPromptTokens int `json:"initial_prompt_tokens"`
			} `json:"prompt_tokens_details"`
			ServerToolUse struct {
				WebSearchRequests int `json:"web_search_requests"`
			} `json:"server_tool_use"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "The answer is 42.", resp.Choices[0].Message.Content)
	assert.Equal(t, "long plan", resp.Choices[0].Message.ReasoningContent)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, 34768, resp.Usage.PromptTokens)
	assert.Equal(t, 33268, resp.Usage.CompletionTokens)
	assert.Equal(t, 33000, resp.Usage.PromptTokensDetails.CachedTokens)
	assert.Equal(t, 1000, resp.Usage.PromptTokensDetails.InitialPromptTokens)
	assert.Equal(t, 2, resp.Usage.ServerToolUse.WebSearchRequests)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 34768, entry.PromptTokens)
	assert.Equal(t, 33268, entry.CompletionTokens)
	assert.InDelta(t, argonTwoSegmentSpend, entry.Spend, 1e-12)
	assert.Equal(t, float64(1000), decodeMetadata(t, entry.Metadata)["initial_prompt_tokens"])
}

func sseFrames(payloads ...string) string {
	var b strings.Builder
	for _, payload := range payloads {
		b.WriteString("data: " + payload + "\r\n\r\n")
	}
	return b.String()
}

func TestProxyRequest_GeminiLongDecodeContinuesStreamed(t *testing.T) {
	upstream := newLongDecodeUpstream(t, func(n int, _ map[string]any, w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 0 {
			// A checkpoint token mid-stream, then the request's CONTINUATION.
			_, _ = w.Write([]byte(sseFrames(
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"long","thought":true}]},"continuationToken":"Y2hlY2twb2ludA=="}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":20000}}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":" plan","thought":true}]},"finishReason":"CONTINUATION","continuationToken":"dG9rLTE=","groundingMetadata":{"webSearchQueries":["argon"]}}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":32768,"totalTokenCount":33768}}`)))
			return
		}
		_, _ = w.Write([]byte(sseFrames(
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"The answer"}]}}],"usageMetadata":{"promptTokenCount":33768,"cachedContentTokenCount":33000,"candidatesTokenCount":200}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":" is 42."}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["argon"]}}],"usageMetadata":{"promptTokenCount":33768,"cachedContentTokenCount":33000,"candidatesTokenCount":500,"totalTokenCount":34268}}`)))
	})
	prx, dbStub := upstream.proxy()

	w := postJSON(prx, "/v1/chat/completions", strings.Replace(argonChatBody, `"model"`, `"stream": true, "model"`, 1))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, upstream.requests, 2)
	for _, path := range upstream.paths {
		assert.Equal(t, "/v1beta1/projects/grant/locations/global/publishers/google/models/gemini-4-argon:streamGenerateContent", path)
	}
	assert.Equal(t, "dG9rLTE=", upstream.requests[1]["continuationToken"], "the CONTINUATION token, not the checkpoint one")

	var content string
	var finishes []string
	var lastUsage map[string]any
	for _, frame := range strings.Split(w.Body.String(), "\n\n") {
		payload := strings.TrimPrefix(strings.TrimSpace(frame), "data: ")
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage map[string]any `json:"usage"`
		}
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk), payload)
		for _, choice := range chunk.Choices {
			content += choice.Delta.Content
			if choice.FinishReason != nil {
				finishes = append(finishes, *choice.FinishReason)
			}
		}
		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}
	}
	assert.Equal(t, "The answer is 42.", content)
	assert.Equal(t, []string{"stop"}, finishes)
	assert.Contains(t, w.Body.String(), "data: [DONE]")
	require.NotNil(t, lastUsage)
	assert.Equal(t, float64(34768), lastUsage["prompt_tokens"])
	assert.Equal(t, float64(33268), lastUsage["completion_tokens"])

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 34768, entry.PromptTokens)
	assert.Equal(t, 33268, entry.CompletionTokens)
	assert.InDelta(t, argonTwoSegmentSpend, entry.Spend, 1e-12)
}

// A follow-up that fails twice ends the generation where it stopped: the client
// gets the output so far as cut short, and the finished request is billed.
func TestProxyRequest_GeminiLongDecodeFailedContinuationKeepsUsage(t *testing.T) {
	saved := geminiContinuationRetryDelay
	geminiContinuationRetryDelay = 0
	t.Cleanup(func() { geminiContinuationRetryDelay = saved })
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "non-streamed", true: "streamed"}[stream], func(t *testing.T) {
			upstream := newLongDecodeUpstream(t, func(n int, _ map[string]any, w http.ResponseWriter) {
				if n > 0 {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error":{"code":500,"message":"Internal error","status":"INTERNAL"}}`))
					return
				}
				if stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte(sseFrames(argonFirstSegment)))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(argonFirstSegment))
			})
			prx, dbStub := upstream.proxy()
			body := argonChatBody
			if stream {
				body = strings.Replace(body, `"model"`, `"stream": true, "model"`, 1)
			}

			w := postJSON(prx, "/v1/chat/completions", body)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Len(t, upstream.requests, 3, "the first request, then a follow-up and its one retry")
			assert.Contains(t, w.Body.String(), `"finish_reason":"length"`)
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 1000, entry.PromptTokens)
			assert.Equal(t, 32768, entry.CompletionTokens)
			assert.InDelta(t, 1000*0.0000026+32768*0.000013+0.0182, entry.Spend, 1e-12)
		})
	}
}

func TestProxyRequest_GeminiLongDecodeContinuesResponsesAPI(t *testing.T) {
	upstream := newLongDecodeUpstream(t, func(n int, _ map[string]any, w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		if n == 0 {
			_, _ = w.Write([]byte(argonFirstSegment))
			return
		}
		_, _ = w.Write([]byte(argonSecondSegment))
	})
	prx, dbStub := upstream.proxy()

	w := postJSON(prx, "/v1/responses", `{"model": "gemini-4-argon", "max_output_tokens": 200000,
		"tools": [{"type": "web_search"}], "input": "Prove it."}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, upstream.requests, 2)
	assert.Equal(t, "dG9rLTE=", upstream.requests[1]["continuationToken"])

	var resp struct {
		Status string `json:"status"`
		Usage  struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "completed", resp.Status)
	assert.Equal(t, 34768, resp.Usage.InputTokens)
	assert.Equal(t, 33268, resp.Usage.OutputTokens)
	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, argonTwoSegmentSpend, dbStub.loggedEntries[0].Spend, 1e-12)
	assert.Equal(t, float64(1000), decodeMetadata(t, dbStub.loggedEntries[0].Metadata)["initial_prompt_tokens"])
}

func TestProxyRequest_GeminiLongDecodeContinuesStreamedResponsesAPI(t *testing.T) {
	upstream := newLongDecodeUpstream(t, func(n int, _ map[string]any, w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 0 {
			_, _ = w.Write([]byte(sseFrames(argonFirstSegment)))
			return
		}
		_, _ = w.Write([]byte(sseFrames(argonSecondSegment)))
	})
	prx, dbStub := upstream.proxy()

	w := postJSON(prx, "/v1/responses", `{"model": "gemini-4-argon", "stream": true, "max_output_tokens": 200000,
		"tools": [{"type": "web_search"}], "input": "Prove it."}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, upstream.requests, 2)
	assert.Equal(t, 1, strings.Count(w.Body.String(), "event: response.completed"))

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 34768, entry.PromptTokens)
	assert.Equal(t, 33268, entry.CompletionTokens)
	assert.InDelta(t, argonTwoSegmentSpend, entry.Spend, 1e-12)
	assert.Equal(t, float64(1000), decodeMetadata(t, entry.Metadata)["initial_prompt_tokens"])
}

// Other Gemini models are never continued, whatever their response says.
func TestProxyRequest_GeminiContinuationOnlyForLongDecodeModels(t *testing.T) {
	upstream := newLongDecodeUpstream(t, func(_ int, _ map[string]any, w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(strings.ReplaceAll(argonFirstSegment, "gemini-4-argon", "gemini-3.1-pro-preview")))
	})
	prx, _ := upstream.proxy()
	setTestModelPrice(prx, "gemini-3.1-pro-preview", argonTestPrice())
	w := postJSON(prx, "/v1/chat/completions", `{"model": "gemini-3.1-pro-preview", "messages": [{"role": "user", "content": "hi"}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Len(t, upstream.requests, 1)
}
