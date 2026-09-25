package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestForceWebSearchResults(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantHidden bool
		// wantResult is the search_result of each tool with a nested
		// web_search object after forcing; nil means the body is unchanged.
		wantResult []any
	}{
		{
			name:       "search_result omitted",
			body:       `{"model":"glm","messages":[],"tools":[{"type":"web_search","web_search":{"enable":true,"search_engine":"search-prime"}}]}`,
			wantHidden: true,
			wantResult: []any{true},
		},
		{
			name:       "search_result false",
			body:       `{"model":"glm","tools":[{"type":"web_search","web_search":{"enable":true,"search_result":false}}]}`,
			wantHidden: true,
			wantResult: []any{true},
		},
		{
			name:       "search_result string False",
			body:       `{"model":"glm","tools":[{"type":"web_search","web_search":{"enable":"True","search_result":"False"}}]}`,
			wantHidden: true,
			wantResult: []any{true},
		},
		{
			name: "search_result already true",
			body: `{"model":"glm","tools":[{"type":"web_search","web_search":{"enable":true,"search_result":true}}]}`,
		},
		{
			name: "search_result string True",
			body: `{"model":"glm","tools":[{"type":"web_search","web_search":{"enable":"True","search_result":"True"}}]}`,
		},
		{
			name: "flat Responses-style web_search tool",
			body: `{"model":"gpt-5","tools":[{"type":"web_search","search_context_size":"low"}]}`,
		},
		{
			name: "function tool named web_search",
			body: `{"model":"glm","tools":[{"type":"function","function":{"name":"web_search","parameters":{}}}]}`,
		},
		{
			name: "no tools",
			body: `{"model":"glm","messages":[{"role":"user","content":"\"web_search\""}]}`,
		},
		{
			name:       "only the hidden tool is changed",
			body:       `{"model":"glm","tools":[{"type":"function","function":{"name":"f","parameters":{}}},{"type":"web_search","web_search":{"enable":true}}]}`,
			wantHidden: true,
			wantResult: []any{true},
		},
		{
			name: "invalid JSON",
			body: `{"tools":[{"type":"web_search","web_search":{`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(tt.body)
			assert.Equal(t, tt.wantHidden, WebSearchResultsHidden(body))

			got := ForceWebSearchResults(body)
			if tt.wantResult == nil {
				assert.Equal(t, tt.body, string(got))
				return
			}
			var req struct {
				Tools []map[string]json.RawMessage `json:"tools"`
			}
			require.NoError(t, json.Unmarshal(got, &req))
			var results []any
			for _, tool := range req.Tools {
				var options map[string]any
				if json.Unmarshal(tool["web_search"], &options) == nil && options != nil {
					results = append(results, options["search_result"])
				}
			}
			assert.Equal(t, tt.wantResult, results)
			assert.False(t, WebSearchResultsHidden(got), "forcing must be idempotent")
		})
	}
}

func TestForceWebSearchResults_KeepsOtherFields(t *testing.T) {
	body := []byte(`{"model":"glm","seed":12345678901234567890,"messages":[{"role":"user","content":"<b>a & b</b>"}],` +
		`"tools":[{"type":"web_search","web_search":{"enable":true,"count":5,"search_prompt":"<keep>"}}]}`)

	got := ForceWebSearchResults(body)

	assert.Contains(t, string(got), `"seed":12345678901234567890`)
	assert.Contains(t, string(got), `"content":"<b>a & b</b>"`)
	assert.Contains(t, string(got), `"search_prompt":"<keep>"`)
	assert.Contains(t, string(got), `"count":5`)
	assert.Contains(t, string(got), `"search_result":true`)
}
