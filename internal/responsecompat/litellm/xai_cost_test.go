package litellm

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// xAI reports the purchase cost as cost_in_usd_ticks and, on the Responses
// API, also as cost_in_nano_usd. Neither may reach the client; xAI's own
// server-side tool usage counters stay.
const xaiResponsesUsage = `{"input_tokens":8,"output_tokens":13,"total_tokens":21,` +
	`"cost_in_usd_ticks":158500,"cost_in_nano_usd":15850,` +
	`"server_side_tool_usage_details":{"web_search_calls":1,"x_posts_fetched":4}}`

func TestXAIResponsesStreamDropsProviderCost(t *testing.T) {
	for _, terminal := range []string{"response.completed", "response.incomplete"} {
		t.Run(terminal, func(t *testing.T) {
			stream := `data: {"type":"` + terminal + `","response":{"id":"resp-1","model":"grok-4.7","usage":` + xaiResponsesUsage + `}}` + "\n\n" +
				"data: [DONE]\n\n"

			output, err := io.ReadAll(New().Stream(Context{
				Endpoint:       "/v1/responses",
				RequestedModel: "x-ai/grok-4.7",
				IncludeUsage:   true,
			}, strings.NewReader(stream)))

			require.NoError(t, err)
			assert.NotContains(t, string(output), "cost_in_usd_ticks")
			assert.NotContains(t, string(output), "cost_in_nano_usd")
			assert.Contains(t, string(output), `"x_posts_fetched":4`)
			assert.Contains(t, string(output), `"total_tokens":21`)
		})
	}
}

func TestXAIResponsesTransformDropsProviderCost(t *testing.T) {
	result := New().Transform(Context{
		Endpoint:       "/v1/responses",
		RequestedModel: "x-ai/grok-4.7",
	}, Response{
		StatusCode: http.StatusOK,
		Headers:    make(http.Header),
		Body:       []byte(`{"id":"resp-1","object":"response","model":"grok-4.7","output":[],"usage":` + xaiResponsesUsage + `}`),
	})

	require.Equal(t, http.StatusOK, result.StatusCode)
	assert.NotContains(t, string(result.Body), "cost_in_usd_ticks")
	assert.NotContains(t, string(result.Body), "cost_in_nano_usd")
	assert.Contains(t, string(result.Body), `"web_search_calls":1`)
}
