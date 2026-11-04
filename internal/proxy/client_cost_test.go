package proxy

import (
	"bufio"
	"encoding/json"
	"io"
	"strconv"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	litellmmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestShouldExposeCostToClient(t *testing.T) {
	assert.False(t, shouldExposeCostToClient(nil))
	assert.False(t, shouldExposeCostToClient(&RequestLogContext{}))
	assert.False(t, shouldExposeCostToClient(&RequestLogContext{
		OrganizationPolicy: &models.OrganizationPolicy{ReturnResponseCostInBody: false},
	}))
	assert.True(t, shouldExposeCostToClient(&RequestLogContext{
		OrganizationPolicy: &models.OrganizationPolicy{ReturnResponseCostInBody: true},
	}))
}

func TestClientFacingCost(t *testing.T) {
	p := &Proxy{}

	t.Run("nil usage", func(t *testing.T) {
		cost, ok := p.clientFacingCost(&RequestLogContext{}, nil)
		assert.False(t, ok)
		assert.Zero(t, cost)
	})

	t.Run("zero usage", func(t *testing.T) {
		cost, ok := p.clientFacingCost(&RequestLogContext{}, &converter.TokenUsage{})
		assert.False(t, ok)
		assert.Zero(t, cost)
	})

	t.Run("no price available", func(t *testing.T) {
		logCtx := &RequestLogContext{}
		usage := &converter.TokenUsage{PromptTokens: 100, CompletionTokens: 50}
		cost, ok := p.clientFacingCost(logCtx, usage)
		assert.False(t, ok)
		assert.Zero(t, cost)
	})

	t.Run("computes margin-adjusted total from the already-resolved price", func(t *testing.T) {
		logCtx := &RequestLogContext{
			ModelPrice: &models.ModelPrice{
				InputCostPerToken:  0.000001,
				OutputCostPerToken: 0.000002,
			},
			TokenInfo: &litellmmodels.TokenInfo{
				CostMarginConfigs: []litellmmodels.CostMarginConfig{
					{"global": litellmmodels.CostMargin{Percentage: 0.1}},
				},
			},
		}
		usage := &converter.TokenUsage{PromptTokens: 1000, CompletionTokens: 1000}
		// base = 1000*0.000001 + 1000*0.000002 = 0.003; +10% margin = 0.0033
		cost, ok := p.clientFacingCost(logCtx, usage)
		require.True(t, ok)
		assert.InDelta(t, 0.0033, cost, 1e-9)
	})
}

func TestInjectCostIntoJSONUsage(t *testing.T) {
	t.Run("adds cost to an existing usage object", func(t *testing.T) {
		body := []byte(`{"id":"resp1","usage":{"prompt_tokens":10,"completion_tokens":5}}`)
		out, ok := injectCostIntoJSONUsage(body, 0.00123)
		require.True(t, ok)

		var decoded struct {
			ID    string `json:"id"`
			Usage struct {
				PromptTokens     int     `json:"prompt_tokens"`
				CompletionTokens int     `json:"completion_tokens"`
				Cost             float64 `json:"cost"`
			} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal(out, &decoded))
		assert.Equal(t, "resp1", decoded.ID)
		assert.Equal(t, 10, decoded.Usage.PromptTokens)
		assert.Equal(t, 5, decoded.Usage.CompletionTokens)
		assert.InDelta(t, 0.00123, decoded.Usage.Cost, 1e-12)
	})

	t.Run("leaves body untouched when there is no usage key", func(t *testing.T) {
		body := []byte(`{"id":"resp1"}`)
		out, ok := injectCostIntoJSONUsage(body, 1.0)
		assert.False(t, ok)
		assert.Equal(t, body, out)
	})

	t.Run("leaves body untouched when usage is null", func(t *testing.T) {
		body := []byte(`{"id":"resp1","usage":null}`)
		out, ok := injectCostIntoJSONUsage(body, 1.0)
		assert.False(t, ok)
		assert.Equal(t, body, out)
	})

	t.Run("leaves body untouched on malformed JSON", func(t *testing.T) {
		body := []byte(`not json`)
		out, ok := injectCostIntoJSONUsage(body, 1.0)
		assert.False(t, ok)
		assert.Equal(t, body, out)
	})
}

func TestInjectCostIntoSSEUsageChunk(t *testing.T) {
	t.Run("rebuilds a single-payload usage chunk with cost added", func(t *testing.T) {
		payload := []byte(`{"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		chunk, ok := injectCostIntoSSEUsageChunk([][]byte{payload}, 0.0042)
		require.True(t, ok)
		assert.True(t, len(chunk) > 0)

		s := string(chunk)
		require.Contains(t, s, "data: ")
		require.Contains(t, s, "\n\n")

		jsonPart := s[len("data: ") : len(s)-2]
		var decoded struct {
			Usage struct {
				TotalTokens int     `json:"total_tokens"`
				Cost        float64 `json:"cost"`
			} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal([]byte(jsonPart), &decoded))
		assert.Equal(t, 15, decoded.Usage.TotalTokens)
		assert.InDelta(t, 0.0042, decoded.Usage.Cost, 1e-12)
	})

	t.Run("bails out on zero or multiple payloads", func(t *testing.T) {
		_, ok := injectCostIntoSSEUsageChunk(nil, 1.0)
		assert.False(t, ok)

		payload := []byte(`{"usage":{"total_tokens":1}}`)
		_, ok = injectCostIntoSSEUsageChunk([][]byte{payload, payload}, 1.0)
		assert.False(t, ok)
	})

	t.Run("bails out when the single payload has no usage key", func(t *testing.T) {
		payload := []byte(`{"id":"chatcmpl-1","choices":[]}`)
		_, ok := injectCostIntoSSEUsageChunk([][]byte{payload}, 1.0)
		assert.False(t, ok)
	})
}

func testLogCtxWithFlatPrice() *RequestLogContext {
	return &RequestLogContext{
		ModelPrice: &models.ModelPrice{
			InputCostPerToken:  0.000001,
			OutputCostPerToken: 0.000002,
		},
	}
}

// readAllSmallChunks drives Read with a tiny destination buffer, regardless
// of how the source itself chooses to fragment -- this is what actually
// exercises the reader's own pending/pendingErr bookkeeping (bufio.Reader.
// ReadSlice already reassembles physical lines on its own).
func readAllSmallChunks(t *testing.T, r io.Reader, chunkSize int) string {
	t.Helper()
	var out []byte
	buf := make([]byte, chunkSize)
	for {
		n, err := r.Read(buf)
		out = append(out, buf[:n]...)
		if err != nil {
			require.ErrorIs(t, err, io.EOF)
			break
		}
	}
	return string(out)
}

func usageLineCost(t *testing.T, out string) (float64, bool) {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, `"usage"`) || !strings.Contains(line, "total_tokens") {
			continue
		}
		payload := strings.TrimPrefix(strings.TrimSpace(line), "data: ")
		var decoded struct {
			Usage struct {
				Cost *float64 `json:"cost"`
			} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal([]byte(payload), &decoded))
		if decoded.Usage.Cost == nil {
			return 0, false
		}
		return *decoded.Usage.Cost, true
	}
	t.Fatalf("no usage line found in output:\n%s", out)
	return 0, false
}

func TestCostInjectingSSEReader(t *testing.T) {
	const stream = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150}}\n\n" +
		"data: [DONE]\n\n"
	// 100*0.000001 + 50*0.000002 = 0.0001 + 0.0001 = 0.0002
	const wantCost = 0.0002

	t.Run("whole stream read at once", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		r := newCostInjectingSSEReader(strings.NewReader(stream), p, logCtx, converter.TokenUsageExtractionOptions{})
		out, err := io.ReadAll(r)
		require.NoError(t, err)

		assert.Contains(t, string(out), `"content":"hi"`)
		assert.Contains(t, string(out), "data: [DONE]")
		cost, ok := usageLineCost(t, string(out))
		require.True(t, ok)
		assert.InDelta(t, wantCost, cost, 1e-12)
	})

	t.Run("fragmented one byte at a time, driven through a tiny destination buffer", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		r := newCostInjectingSSEReader(iotest.OneByteReader(strings.NewReader(stream)), p, logCtx, converter.TokenUsageExtractionOptions{})
		out := readAllSmallChunks(t, r, 3)

		assert.Contains(t, out, `"content":"hi"`)
		assert.Contains(t, out, "data: [DONE]")
		cost, ok := usageLineCost(t, out)
		require.True(t, ok)
		assert.InDelta(t, wantCost, cost, 1e-12)
	})

	t.Run("no price available: usage line passes through without a cost field", func(t *testing.T) {
		p := &Proxy{}
		logCtx := &RequestLogContext{} // no ModelPrice resolvable
		r := newCostInjectingSSEReader(strings.NewReader(stream), p, logCtx, converter.TokenUsageExtractionOptions{})
		out, err := io.ReadAll(r)
		require.NoError(t, err)

		_, ok := usageLineCost(t, string(out))
		assert.False(t, ok)
	})

	t.Run("merges with usage already accumulated from earlier chunks", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		// Simulates a completion-tokens-bearing chunk that already arrived
		// and was merged into logCtx.TokenUsage before this line is read.
		logCtx.TokenUsage = &converter.TokenUsage{CompletionTokens: 50}
		// This line alone only carries the prompt side of the final total.
		promptOnly := "data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":0,\"total_tokens\":100}}\n\n"
		r := newCostInjectingSSEReader(strings.NewReader(promptOnly), p, logCtx, converter.TokenUsageExtractionOptions{})
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		cost, ok := usageLineCost(t, string(out))
		require.True(t, ok)
		// Cost reflects the merge of logCtx.TokenUsage's 50 completion tokens
		// with this line's own 100 prompt tokens, not this line's usage alone
		// (which would price at 100*0.000001 = 0.0001, not wantCost).
		assert.InDelta(t, wantCost, cost, 1e-12)
	})

	t.Run("oversized line falls back to plain passthrough", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		big := "data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"total_tokens\":1},\"padding\":\"" + strings.Repeat("x", 2048) + "\"}\n\n"
		r := &costInjectingSSEReader{
			source:       bufio.NewReader(strings.NewReader(big)),
			proxy:        p,
			logCtx:       logCtx,
			maxLineBytes: 1024,
		}
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		assert.Equal(t, big, string(out))
	})

	t.Run("injects cost only once even if usage appears again", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		twice := "data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150}}\n\n" +
			"data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150}}\n\n"
		r := newCostInjectingSSEReader(strings.NewReader(twice), p, logCtx, converter.TokenUsageExtractionOptions{})
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		assert.Equal(t, 1, strings.Count(string(out), `"cost"`))
	})
}

// responsesAPICompletedEventLine builds a minimal "response.completed" SSE
// line with the given input/output token counts, matching the wire shape
// handlePassthroughResponsesStreaming's own onChunk closure decodes.
func responsesAPICompletedEventLine(inputTokens, outputTokens int) string {
	return "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{" +
		"\"input_tokens\":" + strconv.Itoa(inputTokens) + "," +
		"\"output_tokens\":" + strconv.Itoa(outputTokens) + "," +
		"\"total_tokens\":" + strconv.Itoa(inputTokens+outputTokens) + "," +
		"\"input_tokens_details\":{\"cached_tokens\":0}," +
		"\"output_tokens_details\":{\"reasoning_tokens\":0}" +
		"}}}\n\n"
}

func nestedUsageCost(t *testing.T, out string) (float64, bool) {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, `"response.completed"`) && !strings.Contains(line, `"response.incomplete"`) {
			continue
		}
		payload := strings.TrimPrefix(strings.TrimSpace(line), "data: ")
		var decoded struct {
			Response struct {
				Usage struct {
					Cost *float64 `json:"cost"`
				} `json:"usage"`
			} `json:"response"`
		}
		require.NoError(t, json.Unmarshal([]byte(payload), &decoded))
		if decoded.Response.Usage.Cost == nil {
			return 0, false
		}
		return *decoded.Response.Usage.Cost, true
	}
	t.Fatalf("no response.completed/incomplete line found in output:\n%s", out)
	return 0, false
}

func TestResponsesAPICostInjectingReader(t *testing.T) {
	// 100*0.000001 + 50*0.000002 = 0.0001 + 0.0001 = 0.0002
	const wantCost = 0.0002
	stream := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n" +
		responsesAPICompletedEventLine(100, 50) +
		"data: [DONE]\n\n"
	opts := converter.TokenUsageExtractionOptions{}

	t.Run("adds cost to response.usage on the completed event", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		r := newResponsesAPICostInjectingReader(strings.NewReader(stream), p, logCtx, opts, true)
		out, err := io.ReadAll(r)
		require.NoError(t, err)

		assert.Contains(t, string(out), `"delta":"hi"`)
		assert.Contains(t, string(out), "data: [DONE]")
		cost, ok := nestedUsageCost(t, string(out))
		require.True(t, ok)
		assert.InDelta(t, wantCost, cost, 1e-12)
	})

	t.Run("fragmented one byte at a time", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		r := newResponsesAPICostInjectingReader(iotest.OneByteReader(strings.NewReader(stream)), p, logCtx, opts, true)
		out := readAllSmallChunks(t, r, 5)

		assert.Contains(t, out, `"delta":"hi"`)
		cost, ok := nestedUsageCost(t, out)
		require.True(t, ok)
		assert.InDelta(t, wantCost, cost, 1e-12)
	})

	t.Run("response.incomplete also gets cost added", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		incomplete := strings.Replace(responsesAPICompletedEventLine(100, 50), "response.completed", "response.incomplete", 1)
		r := newResponsesAPICostInjectingReader(strings.NewReader(incomplete), p, logCtx, opts, true)
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		cost, ok := nestedUsageCost(t, string(out))
		require.True(t, ok)
		assert.InDelta(t, wantCost, cost, 1e-12)
	})

	t.Run("no price available: completed event passes through without a cost field", func(t *testing.T) {
		p := &Proxy{}
		logCtx := &RequestLogContext{}
		r := newResponsesAPICostInjectingReader(strings.NewReader(stream), p, logCtx, opts, true)
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		_, ok := nestedUsageCost(t, string(out))
		assert.False(t, ok)
	})

	t.Run("a plain content-delta event is never touched", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		onlyDelta := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"
		r := newResponsesAPICostInjectingReader(strings.NewReader(onlyDelta), p, logCtx, opts, true)
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		assert.Equal(t, onlyDelta, string(out))
	})

	t.Run("oversized completed-event line falls back to plain passthrough", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		big := "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}," +
			"\"padding\":\"" + strings.Repeat("x", 2048) + "\"}}\n\n"
		r := &responsesAPICostInjectingReader{
			source:       bufio.NewReader(strings.NewReader(big)),
			proxy:        p,
			logCtx:       logCtx,
			maxLineBytes: 1024,
		}
		out, err := io.ReadAll(r)
		require.NoError(t, err)
		assert.Equal(t, big, string(out))
	})
}

func TestInjectCostIntoNestedResponseUsage(t *testing.T) {
	t.Run("adds cost under response.usage", func(t *testing.T) {
		payload := []byte(`{"type":"response.completed","response":{"id":"r1","usage":{"input_tokens":10,"output_tokens":5}}}`)
		out, ok := injectCostIntoNestedResponseUsage(payload, 0.0042)
		require.True(t, ok)

		var decoded struct {
			Response struct {
				ID    string `json:"id"`
				Usage struct {
					InputTokens int     `json:"input_tokens"`
					Cost        float64 `json:"cost"`
				} `json:"usage"`
			} `json:"response"`
		}
		require.NoError(t, json.Unmarshal(out, &decoded))
		assert.Equal(t, "r1", decoded.Response.ID)
		assert.Equal(t, 10, decoded.Response.Usage.InputTokens)
		assert.InDelta(t, 0.0042, decoded.Response.Usage.Cost, 1e-12)
	})

	t.Run("bails out when there is no response key", func(t *testing.T) {
		payload := []byte(`{"type":"response.completed"}`)
		_, ok := injectCostIntoNestedResponseUsage(payload, 1.0)
		assert.False(t, ok)
	})

	t.Run("bails out when response.usage is absent", func(t *testing.T) {
		payload := []byte(`{"type":"response.completed","response":{"id":"r1"}}`)
		_, ok := injectCostIntoNestedResponseUsage(payload, 1.0)
		assert.False(t, ok)
	})
}
