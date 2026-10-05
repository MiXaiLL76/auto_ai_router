package litellm

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The reasoning → reasoning_content fold follows converterutil.PickReasoningField,
// the same rule every other reader of the two spellings uses.
func TestReasoningFoldPicksPopulatedField(t *testing.T) {
	cases := []struct {
		name   string
		fields string
		want   any
	}{
		{"reasoning only", `"reasoning":"r"`, "r"},
		{"reasoning_content wins", `"reasoning_content":"rc","reasoning":"r"`, "rc"},
		{"empty reasoning_content falls back", `"reasoning_content":"","reasoning":"r"`, "r"},
		{"null reasoning_content falls back", `"reasoning_content":null,"reasoning":"r"`, "r"},
	}
	for _, tc := range cases {
		t.Run("message/"+tc.name, func(t *testing.T) {
			result := New().Transform(Context{Endpoint: "/v1/chat/completions", RequestedModel: "m"}, Response{
				StatusCode: http.StatusOK,
				Headers:    http.Header{"Content-Type": {"application/json"}},
				Body: []byte(`{"id":"x","created":1,"model":"m","choices":[{"index":0,"finish_reason":"stop",
					"message":{"role":"assistant","content":"ok",` + tc.fields + `}}]}`),
			})
			var body map[string]any
			require.NoError(t, json.Unmarshal(result.Body, &body))
			msg := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
			assert.Equal(t, tc.want, msg["reasoning_content"])
			assert.NotContains(t, msg, "reasoning")
		})
		t.Run("delta/"+tc.name, func(t *testing.T) {
			source := strings.NewReader(
				`data: {"id":"x","created":1,"choices":[{"index":0,"delta":{"role":"assistant",` + tc.fields + `}}]}` + "\n\n" +
					"data: [DONE]\n\n")
			output, err := io.ReadAll(New().Stream(Context{Endpoint: "/v1/chat/completions", RequestedModel: "m"}, source))
			require.NoError(t, err)
			frames := splitDataFrames(string(output))
			require.NotEmpty(t, frames)
			var chunk map[string]any
			require.NoError(t, json.Unmarshal([]byte(frames[0]), &chunk))
			delta := chunk["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
			assert.Equal(t, tc.want, delta["reasoning_content"])
			assert.NotContains(t, delta, "reasoning")
		})
	}
}
