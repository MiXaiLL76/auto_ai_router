package vertexresponses

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/require"
)

const hydrogen = "gemini-4-hydrogen"

func hydrogenWire(t *testing.T, api, model string, request map[string]any) ([]byte, error) {
	t.Helper()
	body, err := json.Marshal(request)
	require.NoError(t, err)
	if api == "responses" {
		return ResponsesRequestToVertex(body, model)
	}
	return vertex.OpenAIToVertex(body, false, false, model, "application/json")
}

func TestHydrogenWireThinking(t *testing.T) {
	for _, test := range []struct {
		model, effort, level string
		budget               int
	}{
		{hydrogen, "", "HIGH", 0}, {hydrogen, "low", "LOW", 0}, {hydrogen, "medium", "MEDIUM", 0},
		{hydrogen, "high", "HIGH", 0}, {"google/" + hydrogen, "xhigh", "XHIGH", 0}, {hydrogen, "max", "XHIGH", 0},
		{"gemini-4-argon", "max", "HIGH", 0}, {"gemini-4-carbon", "xhigh", "HIGH", 0},
		{"gemini-3.8-flash", "xhigh", "HIGH", 0}, {"gemini-2.5-pro", "max", "", 24576},
	} {
		for _, api := range []string{"chat", "responses"} {
			t.Run(api+"/"+test.model+"/"+test.effort, func(t *testing.T) {
				msg := map[string]any{"role": "user", "content": "hello"}
				req := map[string]any{"temperature": 0.2}
				if api == "chat" {
					req["messages"] = []any{msg}
					req["reasoning_effort"] = test.effort
				} else {
					req["input"] = []any{msg}
					req["reasoning"] = map[string]any{"effort": test.effort}
				}
				wire, err := hydrogenWire(t, api, test.model, req)
				require.NoError(t, err)
				var out struct {
					Generation map[string]any `json:"generationConfig"`
				}
				require.NoError(t, json.Unmarshal(wire, &out))
				tc := out.Generation["thinkingConfig"].(map[string]any)
				if test.level != "" {
					require.Equal(t, test.level, tc["thinkingLevel"])
					require.NotContains(t, tc, "thinkingBudget")
				} else {
					require.Equal(t, float64(test.budget), tc["thinkingBudget"])
				}
				if vertex.UsesPrefixBoundSignatures(test.model) {
					require.Equal(t, "DROP", tc["prefixMismatchBehavior"])
					require.NotContains(t, out.Generation, "temperature")
				} else {
					require.NotContains(t, tc, "prefixMismatchBehavior")
				}
			})
		}
	}
}

func TestHydrogenNativeOptions(t *testing.T) {
	for _, api := range []string{"chat", "responses"} {
		for _, test := range []struct {
			key         string
			value       any
			param, code string
		}{
			{"thinking_budget", 0, "thinking_budget", "unsupported_parameter"},
			{"thinkingBudget", -1, "thinking_budget", "unsupported_parameter"},
			{"thinkingConfig", map[string]any{"thinkingLevel": "MINIMAL"}, "thinking_level", "unsupported_value"},
			{"extra_body", map[string]any{"generation_config": map[string]any{"thinking_config": map[string]any{"thinking_budget": 1024}}}, "thinking_budget", "unsupported_parameter"},
			{"n", 2, "n", "unsupported_value"},
		} {
			t.Run(api+"/"+test.key, func(t *testing.T) {
				_, err := hydrogenWire(t, api, hydrogen, map[string]any{"input": "hello", "messages": []any{map[string]string{"role": "user", "content": "hello"}}, test.key: test.value})
				testhelpers.RequireValidationError(t, err, test.param, test.code)
			})
		}
		wire, err := hydrogenWire(t, api, hydrogen, map[string]any{"input": "hi", "messages": []any{map[string]string{"role": "user", "content": "hi"}},
			"thinking_config": map[string]any{"thinking_level": "xhigh", "include_thoughts": true}})
		require.NoError(t, err)
		require.Contains(t, string(wire), `"thinkingLevel":"XHIGH"`)
		require.Contains(t, string(wire), `"includeThoughts":true`)
	}
}

func TestHydrogenParallelCallsAfterHistoryEdit(t *testing.T) {
	for _, api := range []string{"chat", "responses"} {
		t.Run(api, func(t *testing.T) {
			const signed = "__thought__c2ln"
			// First replayed turn has an authentic signature; the second came
			// from a different provider. A stale signature on a sibling is ignored.
			req := map[string]any{}
			var history []any
			msg := func(role, text string) map[string]any { return map[string]any{"role": role, "content": text} }
			if api == "chat" {
				history = append(history, msg("system", "updated instructions"))
			} else {
				req["instructions"] = "updated instructions"
			}
			history = append(history, msg("user", "shortened history"))
			for turn := 0; turn < 2; turn++ {
				var calls, results []any
				for index := 0; index < 3; index++ {
					id := fmt.Sprintf("f%d-%d", turn, index)
					if index == 1 || (turn == 0 && index == 0) {
						id += signed
					}
					name := fmt.Sprintf("lookup%d", index)
					if api == "chat" {
						calls = append(calls, map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": "{}"}})
						results = append(results, map[string]any{"role": "tool", "tool_call_id": id, "content": "ok"})
					} else {
						calls = append(calls, map[string]any{"type": "function_call", "call_id": id, "name": name, "arguments": "{}"})
						results = append(results, map[string]any{"type": "function_call_output", "call_id": id, "output": "ok"})
					}
				}
				if api == "chat" {
					history = append(history, map[string]any{"role": "assistant", "content": "checking", "tool_calls": calls})
				} else {
					history = append(history, msg("assistant", "checking"))
					history = append(history, calls...)
				}
				for index := len(results) - 1; index >= 0; index-- {
					history = append(history, results[index])
				}
			}
			if api == "chat" {
				req["messages"] = history
			} else {
				req["input"] = history
			}
			wire, err := hydrogenWire(t, api, hydrogen, req)
			require.NoError(t, err)
			var out vertex.VertexRequest
			require.NoError(t, json.Unmarshal(wire, &out))
			require.Equal(t, "updated instructions", out.SystemInstruction.Parts[0].Text)
			require.Len(t, out.Contents, 5)
			for turn := 0; turn < 2; turn++ {
				calls, results := out.Contents[1+turn*2].Parts, out.Contents[2+turn*2].Parts
				require.Len(t, calls, 4)
				require.Len(t, results, 3)
				require.Equal(t, "checking", calls[0].Text)
				for index, part := range calls[1:] {
					id := fmt.Sprintf("f%d-%d", turn, index)
					require.Equal(t, id, part.FunctionCall.ID)
					require.Equal(t, id, results[index].FunctionResponse.ID)
					require.Equal(t, part.FunctionCall.Name, results[index].FunctionResponse.Name)
					if index != 0 {
						require.Nil(t, part.ThoughtSignature)
					} else if turn == 0 {
						require.Equal(t, []byte("sig"), part.ThoughtSignature)
					} else {
						require.Equal(t, []byte("skip_thought_signature_validator"), part.ThoughtSignature)
					}
				}
			}
		})
	}
}
