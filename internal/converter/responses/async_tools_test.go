package responses

import (
	"encoding/json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestPrepareCodexPassthroughPreservesAsyncFunctionFields(t *testing.T) {
	body := []byte(`{"tools":[{"type":"function","async":true,"defer_loading":true,"function":{"name":"lookup","strict":true,"parameters":{"type":"object"}}}]}`)
	var got map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(PrepareCodexPassthrough(body, false), &got))
	assert.JSONEq(t, `[{"type":"function","async":true,"defer_loading":true,"name":"lookup","strict":true,"parameters":{"type":"object"}}]`, string(got["tools"]))
}

func TestPrepareCodexPassthroughAdditionalToolsChoice(t *testing.T) {
	for _, choice := range []string{`"required"`, `{"type":"function","name":"lookup"}`} {
		for _, additional := range []string{`[]`, `[{"type":"function","name":"lookup","async":true}]`} {
			t.Run(choice+additional, func(t *testing.T) {
				var got map[string]json.RawMessage
				body := []byte(`{"tools":[],"additional_tools":` + additional + `,"tool_choice":` + choice + `}`)
				require.NoError(t, json.Unmarshal(PrepareCodexPassthrough(body, false), &got))
				assert.NotContains(t, got, "tools")
				assert.JSONEq(t, additional, string(got["additional_tools"]))
				if additional == `[]` {
					assert.NotContains(t, got, "tool_choice")
				} else {
					assert.JSONEq(t, choice, string(got["tool_choice"]))
				}
			})
		}
	}
}
