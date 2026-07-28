package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_ResponsesPassthroughPreservesBuiltInTools(t *testing.T) {
	tools := []struct {
		name       string
		tool       string
		toolChoice string
	}{
		{name: "web search", tool: `{"type":"web_search"}`, toolChoice: `{"type":"web_search"}`},
		{name: "file search", tool: `{"type":"file_search","vector_store_ids":["vs_1"]}`, toolChoice: `{"type":"file_search"}`},
	}
	providers := []config.ProviderType{
		config.ProviderTypeOpenAI,
		config.ProviderTypeProxy,
		config.ProviderTypeAIR,
	}

	for _, provider := range providers {
		for _, tt := range tools {
			t.Run(string(provider)+"/"+tt.name, func(t *testing.T) {
				var outbound map[string]interface{}
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					require.Equal(t, "/v1/responses", r.URL.Path)
					require.NoError(t, json.NewDecoder(r.Body).Decode(&outbound))
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{
						"id":"resp_1",
						"object":"response",
						"status":"completed",
						"model":"gpt-test",
						"output":[],
						"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
					}`))
				}))
				defer upstream.Close()

				prx := NewTestProxyBuilder().
					WithSingleCredential("upstream", provider, upstream.URL, "upstream-key").
					WithMasterKey("master-key").
					Build()
				req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
					`{"model":"gpt-test","input":"hello","tools":[`+tt.tool+`],"tool_choice":`+tt.toolChoice+`}`,
				))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				recorder := httptest.NewRecorder()

				prx.ProxyRequest(recorder, req)

				require.Equal(t, http.StatusOK, recorder.Code)
				outboundTools := outbound["tools"].([]interface{})
				require.Len(t, outboundTools, 1)
				assert.Equal(t, strings.ReplaceAll(tt.name, " ", "_"), outboundTools[0].(map[string]interface{})["type"])
				assert.Equal(t, strings.ReplaceAll(tt.name, " ", "_"), outbound["tool_choice"].(map[string]interface{})["type"])
			})
		}
	}
}

func TestProxyRequest_ChatCompletionsNormalizesWebSearchTools(t *testing.T) {
	tests := []struct {
		name                 string
		model                string
		tools                string
		wantWebSearchOptions bool
		wantFunction         bool
	}{
		{
			name:                 "search preview converts options",
			model:                "gpt-4o-search-preview",
			tools:                `[{"type":"web_search_preview","search_context_size":"high","user_location":{"type":"approximate","country":"RU"}}]`,
			wantWebSearchOptions: true,
		},
		{
			name:                 "function remains beside web search",
			model:                "gpt-4o-search-preview",
			tools:                `[{"type":"web_search"},{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]`,
			wantWebSearchOptions: true,
			wantFunction:         true,
		},
		{
			name:         "unsupported model follows documented drop policy",
			model:        "gpt-4o-mini",
			tools:        `[{"type":"web_search"},{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]`,
			wantFunction: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var outbound map[string]interface{}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, "/v1/chat/completions", r.URL.Path)
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				require.NoError(t, json.Unmarshal(body, &outbound))
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{
					"id":"chatcmpl_1",
					"object":"chat.completion",
					"created":1,
					"model":"gpt-test",
					"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
					"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
				}`))
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
				WithMasterKey("master-key").
				Build()
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
				`{"model":"`+tt.model+`","messages":[{"role":"user","content":"hello"}],"tools":`+tt.tools+`,"tool_choice":"auto"}`,
			))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()

			prx.ProxyRequest(recorder, req)

			require.Equal(t, http.StatusOK, recorder.Code)
			options, hasOptions := outbound["web_search_options"].(map[string]interface{})
			assert.Equal(t, tt.wantWebSearchOptions, hasOptions)
			if tt.name == "search preview converts options" {
				assert.Equal(t, "high", options["search_context_size"])
				assert.Equal(t, "RU", options["user_location"].(map[string]interface{})["country"])
			}
			tools, hasTools := outbound["tools"].([]interface{})
			assert.Equal(t, tt.wantFunction, hasTools)
			if tt.wantFunction {
				require.Len(t, tools, 1)
				assert.Equal(t, "function", tools[0].(map[string]interface{})["type"])
			}
		})
	}
}
