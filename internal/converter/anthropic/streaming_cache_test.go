package anthropic

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTransformAnthropicStreamToOpenAI_PreservesCacheCreationUsage(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_cache","usage":{"input_tokens":100,"cache_creation_input_tokens":40,"cache_read_input_tokens":60,"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":30}}}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"

	var out bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToOpenAI(strings.NewReader(stream), "claude-test", &out))

	var usage map[string]interface{}
	for _, line := range strings.Split(out.String(), "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var chunk struct {
			Usage map[string]interface{} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk))
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}

	require.NotNil(t, usage)
	require.Equal(t, float64(200), usage["prompt_tokens"])
	details := usage["prompt_tokens_details"].(map[string]interface{})
	require.Equal(t, float64(60), details["cached_tokens"])
	require.Equal(t, float64(40), details["cache_creation_tokens"])
	ttlDetails := details["cache_creation_token_details"].(map[string]interface{})
	require.Equal(t, float64(10), ttlDetails["ephemeral_5m_input_tokens"])
	require.Equal(t, float64(30), ttlDetails["ephemeral_1h_input_tokens"])
}
