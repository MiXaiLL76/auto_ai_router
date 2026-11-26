package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The mock checks each model's thinking configuration and continuation request.
// Its usage includes search context, a repeated search and cached input on the
// continuation; all four entry points must log a consistent charge.
func TestGemini4ContinuesAndBills(t *testing.T) {
	for _, test := range []struct {
		model, prefixMismatchBehavior string
		promptTokens                  int
	}{
		{"gemini-4-hydrogen", "DROP", 65736},
		{"gemini-4-neon", "", 65762},
	} {
		t.Run(test.model, func(t *testing.T) {
			testGemini4ContinuesAndBills(t, test.model, test.prefixMismatchBehavior, test.promptTokens)
		})
	}
}

func testGemini4ContinuesAndBills(t *testing.T, model, prefixMismatchBehavior string, promptTokens int) {
	t.Helper()
	modelVersion := fmt.Sprintf(`"modelVersion":%q,`, model)
	firstCandidate := modelVersion + `"candidates":[{"content":{"role":"model","parts":[{"text":"plan","thought":true}]},"finishReason":"CONTINUATION","continuationToken":"dG9r","groundingMetadata":{"webSearchQueries":["one"]}}]`
	const firstUsage = `"usageMetadata":{"promptTokenCount":100,"thoughtsTokenCount":65536,"toolUsePromptTokenCount":11,"totalTokenCount":65647}`
	first := "{" + firstCandidate + "," + firstUsage + "}"
	last := "{" + modelVersion + `"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["one"]}}],"usageMetadata":{"promptTokenCount":65636,"cachedContentTokenCount":65000,"candidatesTokenCount":8,"toolUsePromptTokenCount":15,"totalTokenCount":65659}}`
	for _, endpoint := range []string{"/v1/chat/completions", "/v1/responses"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", endpoint, stream), func(t *testing.T) {
				u := newLongDecodeUpstream(t, func(n int, req map[string]any, w http.ResponseWriter) {
					cfg := req["generationConfig"].(map[string]any)
					tc := cfg["thinkingConfig"].(map[string]any)
					mismatch, _ := tc["prefixMismatchBehavior"].(string)
					if mismatch != prefixMismatchBehavior || tc["thinkingLevel"] != "XHIGH" {
						http.Error(w, "incorrect thinking config", http.StatusBadRequest)
						return
					}
					body := first
					if n == 1 {
						body = last
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						if n == 0 {
							body = sseFrames("{"+firstCandidate+"}", "{"+firstUsage+"}", "[DONE]")
						} else {
							body = sseFrames(body, `{}`)
						}
					} else {
						w.Header().Set("Content-Type", "application/json")
					}
					_, _ = w.Write([]byte(body))
				})
				prx, db := u.proxy()
				setTestModelPrice(prx, model, &pricing.ModelPrice{
					InputCostPerToken: 2e-6, OutputCostPerToken: 8e-6, CacheReadInputTokenCost: 0.5e-6,
					WebSearchBillingUnit: "per_query", SearchContextCostPerQuery: map[string]float64{"search_context_size_medium": 0.01, "search_context_size_low": 0.01, "search_context_size_high": 0.01},
				})
				req := map[string]any{"model": model, "stream": stream, "tools": []any{map[string]any{"type": "web_search"}}}
				if endpoint == "/v1/responses" {
					req["input"] = []any{
						map[string]any{"role": "user", "content": "shortened"},
						map[string]any{"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "a__thought__c2ln"},
						map[string]any{"type": "function_call_output", "call_id": "a__thought__c2ln", "output": "ok"},
					}
					req["instructions"] = "edited system"
					req["reasoning"] = map[string]any{"effort": "max"}
				} else {
					req["messages"] = []any{
						map[string]any{"role": "system", "content": "edited system"},
						map[string]any{"role": "user", "content": "shortened"},
						map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"type": "function", "id": "a", "function": map[string]any{"name": "lookup", "arguments": "{}"}, "provider_specific_fields": map[string]string{"thought_signature": "c2ln"}}}},
						map[string]any{"role": "tool", "tool_call_id": "a", "content": "ok"},
					}
					req["reasoning_effort"] = "xhigh"
				}
				body, err := json.Marshal(req)
				require.NoError(t, err)
				res := postJSON(prx, endpoint, string(body))
				require.Equal(t, http.StatusOK, res.Code, res.Body.String())
				require.Contains(t, res.Body.String(), "done")
				require.Len(t, u.requests, 2)
				require.Equal(t, "dG9r", u.requests[1]["continuationToken"])
				delete(u.requests[1], "continuationToken")
				require.Equal(t, u.requests[0], u.requests[1])
				require.Len(t, db.loggedEntries, 1)
				entry := db.loggedEntries[0]
				assert.Equal(t, promptTokens, entry.PromptTokens)
				assert.Equal(t, 65544, entry.CompletionTokens)
				assert.InDelta(t, float64(promptTokens-65000)*2e-6+65000*0.5e-6+65544*8e-6+2*0.01, entry.Spend, 1e-10)
			})
		}
	}
}

func TestHydrogenContinuationMixedToolBilling(t *testing.T) {
	const model = "gemini-4-hydrogen"
	const search = `"groundingMetadata":{"webSearchQueries":["q"]}`
	const url = `"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.org"}]}`
	for _, endpoint := range []string{"/v1/chat/completions", "/v1/responses"} {
		for _, stream := range []bool{false, true} {
			for _, searchFirst := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t/searchFirst=%t", endpoint, stream, searchFirst), func(t *testing.T) {
					u := newLongDecodeUpstream(t, func(n int, req map[string]any, w http.ResponseWriter) {
						grounding, tokens := url, 20
						if (n == 0) == searchFirst {
							grounding, tokens = search, 100
						}
						finish := `"finishReason":"STOP"`
						if n == 0 {
							finish = `"finishReason":"CONTINUATION","continuationToken":"dG9r"`
						}
						usage := fmt.Sprintf(`{"promptTokenCount":10,"toolUsePromptTokenCount":%d,"candidatesTokenCount":1,"totalTokenCount":%d}`, tokens, 11+tokens)
						candidate := fmt.Sprintf(`{"content":{"role":"model","parts":[{"text":"answer"}]},%s,%s}`, grounding, finish)
						body := fmt.Sprintf(`{"modelVersion":%q,"candidates":[%s],"usageMetadata":%s}`, model, candidate, usage)
						if stream {
							// Counters can arrive before the chunk that identifies the tool.
							w.Header().Set("Content-Type", "text/event-stream")
							body = sseFrames(fmt.Sprintf(`{"usageMetadata":%s}`, usage), fmt.Sprintf(`{"modelVersion":%q,"candidates":[%s]}`, model, candidate))
						}
						_, _ = w.Write([]byte(body))
					})
					prx, db := u.proxy()
					setTestModelPrice(prx, model, &pricing.ModelPrice{InputCostPerToken: 2e-6, OutputCostPerToken: 8e-6})
					req := map[string]any{"model": model, "stream": stream, "tools": []any{map[string]string{"type": "web_search"}, map[string]string{"type": "url_context"}}}
					if endpoint == "/v1/responses" {
						req["input"] = "compare sources"
					} else {
						req["messages"] = []map[string]string{{"role": "user", "content": "compare sources"}}
					}
					body, err := json.Marshal(req)
					require.NoError(t, err)
					res := postJSON(prx, endpoint, string(body))
					require.Equal(t, http.StatusOK, res.Code, res.Body.String())
					require.Len(t, u.requests, 2)
					require.Len(t, db.loggedEntries, 1)
					entry := db.loggedEntries[0]
					assert.Equal(t, 40, entry.PromptTokens)
					assert.Equal(t, 2, entry.CompletionTokens)
					assert.InDelta(t, 40*2e-6+2*8e-6, entry.Spend, 1e-10)
				})
			}
		}
	}
}
