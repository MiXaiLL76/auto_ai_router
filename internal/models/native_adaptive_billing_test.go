package models

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	anthropicresponses "github.com/mixaill76/auto_ai_router/internal/converter/anthropic/responses"
)

// A single-rate row for a model with adaptive thinking reported inside output_tokens
// and per-query web search: no above_512k fields, no cache-write rate, no reasoning
// flags.
const singleRateThinkingPrice = `{
	"input_cost_per_token": 0.000000195,
	"output_cost_per_token": 0.00000078,
	"cache_read_input_token_cost": 0.0000000195,
	"rate": 1.3,
	"supports_web_search": true,
	"search_context_cost_per_query": {
		"search_context_size_low": 0.013,
		"search_context_size_medium": 0.013,
		"search_context_size_high": 0.013
	},
	"web_search_billing_unit": "per_query"
}`

func singleRateThinkingModelPrice(t *testing.T) *ModelPrice {
	t.Helper()
	var price ModelPrice
	require.NoError(t, json.Unmarshal([]byte(singleRateThinkingPrice), &price))
	return &price
}

func messagesBody(inputTokens, cacheRead, outputTokens, thinkingTokens, webSearches int) string {
	usage := `"input_tokens":` + strconv.Itoa(inputTokens) +
		`,"cache_read_input_tokens":` + strconv.Itoa(cacheRead) +
		`,"output_tokens":` + strconv.Itoa(outputTokens) +
		`,"output_tokens_details":{"thinking_tokens":` + strconv.Itoa(thinkingTokens) + `}`
	content := `{"type":"thinking","thinking":"t","signature":"s"}`
	for i := 0; i < webSearches; i++ {
		content += `,{"type":"server_tool_use","id":"srvtoolu_` + strconv.Itoa(i) + `","name":"web_search","input":{"query":"q"}}`
	}
	content += `,{"type":"text","text":"answer"}`
	return `{"id":"msg_1","type":"message","role":"assistant","model":"MiniMax-M3.1-Flash","content":[` + content +
		`],"stop_reason":"end_turn","usage":{` + usage + `}}`
}

func messagesStream(inputTokens, cacheRead, outputTokens, thinkingTokens, webSearches int) string {
	events := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"MiniMax-M3.1-Flash","usage":{"input_tokens":` + strconv.Itoa(inputTokens) +
			`,"cache_read_input_tokens":` + strconv.Itoa(cacheRead) + `,"output_tokens":0}}}`,
	}
	index := 0
	for i := 0; i < webSearches; i++ {
		events = append(events,
			`{"type":"content_block_start","index":`+strconv.Itoa(index)+`,"content_block":{"type":"server_tool_use","id":"srvtoolu_`+strconv.Itoa(i)+`","name":"web_search","input":{}}}`,
			`{"type":"content_block_stop","index":`+strconv.Itoa(index)+`}`)
		index++
	}
	events = append(events,
		`{"type":"content_block_start","index":`+strconv.Itoa(index)+`,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":`+strconv.Itoa(index)+`,"delta":{"type":"text_delta","text":"answer"}}`,
		`{"type":"content_block_stop","index":`+strconv.Itoa(index)+`}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":`+strconv.Itoa(outputTokens)+
			`,"output_tokens_details":{"thinking_tokens":`+strconv.Itoa(thinkingTokens)+`}}}`,
		`{"type":"message_stop"}`)
	var stream strings.Builder
	for _, event := range events {
		stream.WriteString("data: " + event + "\n\n")
	}
	return stream.String()
}

// passthroughStreamUsage relays stream as the /v1/messages passthrough does and
// extracts billing usage from its final message_delta, as the stream reader does.
func passthroughStreamUsage(t *testing.T, stream string) *converter.TokenUsage {
	t.Helper()
	var relayed strings.Builder
	require.NoError(t, anthropicconv.PassthroughMessagesStream(strings.NewReader(stream), &relayed, "minimax-m3.1-flash"))
	var usage *converter.TokenUsage
	for _, line := range strings.Split(relayed.String(), "\n") {
		if payload, ok := strings.CutPrefix(line, "data: "); ok && strings.Contains(payload, `"message_delta"`) {
			usage = converter.ExtractTokenUsage([]byte(payload))
		}
	}
	return usage
}

// usagesForAllShapes extracts billing usage the way the router bills the same upstream
// Messages response on each client API: the Chat Completions conversion (/v1/chat and
// non-streaming /v1/messages, which bills the Chat form), the Responses API conversion,
// and the relayed final message_delta of a streaming /v1/messages passthrough.
func usagesForAllShapes(t *testing.T, inputTokens, cacheRead, outputTokens, thinkingTokens, webSearches int) map[string]*converter.TokenUsage {
	t.Helper()
	body := messagesBody(inputTokens, cacheRead, outputTokens, thinkingTokens, webSearches)
	chatBody, err := anthropicconv.AnthropicToOpenAI([]byte(body), "minimax-m3.1-flash")
	require.NoError(t, err)
	response, err := anthropicresponses.AnthropicToResponsesResponse([]byte(body), "minimax-m3.1-flash", "", 0)
	require.NoError(t, err)
	responsesBody, err := json.Marshal(response)
	require.NoError(t, err)
	return map[string]*converter.TokenUsage{
		"chat":            converter.ExtractTokenUsage(chatBody),
		"responses":       converter.ExtractTokenUsage(responsesBody),
		"messages stream": passthroughStreamUsage(t, messagesStream(inputTokens, cacheRead, outputTokens, thinkingTokens, webSearches)),
	}
}

// 100 new + 900 cached input + 100 output (60 of them thinking) = $0.00011505: thinking
// is billed once, inside output, on every client API.
func TestSingleRateThinkingBilling_ControlSum(t *testing.T) {
	price := singleRateThinkingModelPrice(t)
	for shape, usage := range usagesForAllShapes(t, 100, 900, 100, 60, 0) {
		t.Run(shape, func(t *testing.T) {
			require.NotNil(t, usage)
			assert.Equal(t, 1000, usage.PromptTokens)
			assert.Equal(t, 900, usage.CachedInputTokens)
			assert.Equal(t, 100, usage.CompletionTokens)
			assert.Equal(t, 60, usage.ReasoningTokens)
			assert.InDelta(t, 0.00011505, price.CalculateCost(usage), 1e-12)
		})
	}

	// The same bill without the thinking breakdown: it splits output, adds nothing.
	for shape, usage := range usagesForAllShapes(t, 100, 900, 100, 0, 0) {
		assert.InDelta(t, 0.00011505, price.CalculateCost(usage), 1e-12, shape)
	}
}

func TestSingleRateThinkingBilling_WebSearch(t *testing.T) {
	price := singleRateThinkingModelPrice(t)
	for _, searches := range []int{0, 1, 3} {
		for shape, usage := range usagesForAllShapes(t, 100, 900, 100, 60, searches) {
			assert.Equal(t, searches, usage.WebSearchRequests, "%s, %d searches", shape, searches)
			assert.InDelta(t, 0.00011505+0.013*float64(searches), price.CalculateCost(usage), 1e-12, "%s, %d searches", shape, searches)
		}
	}
}

// One token rate for the whole context: the same per-token price below and above 512K.
func TestSingleRateThinkingBilling_NoLongContextTier(t *testing.T) {
	price := singleRateThinkingModelPrice(t)
	for _, input := range []int{400_000, 512_000, 512_001, 900_000} {
		usage := converter.ExtractTokenUsage([]byte(messagesBody(input, 100_000, 2_000, 500, 0)))
		require.NotNil(t, usage)
		want := float64(input)*0.000000195 + 100_000*0.0000000195 + 2_000*0.00000078
		assert.InDelta(t, want, price.CalculateCost(usage), 1e-9, "input %d", input)
	}
}
