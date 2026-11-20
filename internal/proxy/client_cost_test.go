package proxy

import (
	"bufio"
	"bytes"
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

	// Mirrors logSpendToLiteLLMDB's own rule ("LiteLLM bills a fixed margin
	// only on priced responses"): a fixed-amount margin must not turn a
	// genuinely free ($0 base cost) request into a non-zero one here, or
	// the client-visible cost would disagree with what's actually billed.
	t.Run("a fixed margin never applies to a zero base cost", func(t *testing.T) {
		logCtx := &RequestLogContext{
			ModelPrice: &models.ModelPrice{
				InputCostPerToken:  0,
				OutputCostPerToken: 0,
			},
			TokenInfo: &litellmmodels.TokenInfo{
				CostMarginConfigs: []litellmmodels.CostMarginConfig{
					{"global": litellmmodels.CostMargin{FixedAmount: 0.01}},
				},
			},
		}
		usage := &converter.TokenUsage{PromptTokens: 1000, CompletionTokens: 1000}
		cost, ok := p.clientFacingCost(logCtx, usage)
		require.True(t, ok)
		assert.Zero(t, cost)
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

	// Regression (Bug 1): a sibling object with byte-identical content to
	// the real top-level "usage" used to confuse a content-based search --
	// the cost would land in metadata.usage (the earlier, coincidental
	// match) instead of the real usage object the client actually reads.
	t.Run("targets the real top-level usage, not an identical-looking nested one", func(t *testing.T) {
		body := []byte(`{"metadata":{"usage":{"prompt_tokens":10}},"usage":{"prompt_tokens":10}}`)
		out, ok := injectCostIntoJSONUsage(body, 0.0042)
		require.True(t, ok)

		var decoded struct {
			Metadata struct {
				Usage struct {
					Cost *float64 `json:"cost"`
				} `json:"usage"`
			} `json:"metadata"`
			Usage struct {
				Cost *float64 `json:"cost"`
			} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal(out, &decoded))
		assert.Nil(t, decoded.Metadata.Usage.Cost, "metadata.usage must stay untouched")
		require.NotNil(t, decoded.Usage.Cost, "the real top-level usage must get the cost")
		assert.InDelta(t, 0.0042, *decoded.Usage.Cost, 1e-12)
	})

	// Regression (Bug 2): a provider-reported usage.cost (non-compat mode,
	// where only the LiteLLM transform strips it) used to end up duplicated
	// as a second "cost" key -- invalid JSON, and ambiguous for any parser
	// that doesn't happen to take "last key wins".
	t.Run("replaces an existing provider-reported cost in place, no duplicate key", func(t *testing.T) {
		body := []byte(`{"id":"resp1","usage":{"prompt_tokens":10,"cost":9.99}}`)
		out, ok := injectCostIntoJSONUsage(body, 0.0042)
		require.True(t, ok)
		assert.Equal(t, 1, strings.Count(string(out), `"cost"`), "must not write a second cost key")

		var decoded struct {
			Usage struct {
				PromptTokens int     `json:"prompt_tokens"`
				Cost         float64 `json:"cost"`
			} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal(out, &decoded))
		assert.Equal(t, 10, decoded.Usage.PromptTokens)
		assert.InDelta(t, 0.0042, decoded.Usage.Cost, 1e-12)
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
		chunk := []byte("data: " + string(payload) + "\n\n")
		out, ok := injectCostIntoSSEUsageChunk(chunk, [][]byte{payload}, 0.0042)
		require.True(t, ok)

		s := string(out)
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
		_, ok := injectCostIntoSSEUsageChunk([]byte("data: {}\n\n"), nil, 1.0)
		assert.False(t, ok)

		payload := []byte(`{"usage":{"total_tokens":1}}`)
		chunk := []byte("data: " + string(payload) + "\ndata: " + string(payload) + "\n\n")
		_, ok = injectCostIntoSSEUsageChunk(chunk, [][]byte{payload, payload}, 1.0)
		assert.False(t, ok)
	})

	t.Run("bails out when the single payload has no usage key, either shape", func(t *testing.T) {
		payload := []byte(`{"id":"chatcmpl-1","choices":[]}`)
		chunk := []byte("data: " + string(payload) + "\n\n")
		_, ok := injectCostIntoSSEUsageChunk(chunk, [][]byte{payload}, 1.0)
		assert.False(t, ok)
	})

	// Regression: an Anthropic Messages API chunk is written as a single
	// fmt.Fprintf("event: %s\ndata: %s\n\n", ...) call -- the whole thing
	// reaches tokenCapturingWriter.Write as one []byte. The old
	// implementation rebuilt the chunk from only the "data:" payload,
	// silently dropping the preceding "event:" line a Messages API client
	// needs to know what kind of event this is.
	t.Run("preserves a bundled event: line (Anthropic Messages API shape)", func(t *testing.T) {
		payload := []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
		chunk := []byte("event: message_delta\ndata: " + string(payload) + "\n\n")
		out, ok := injectCostIntoSSEUsageChunk(chunk, [][]byte{payload}, 0.0007)
		require.True(t, ok)

		s := string(out)
		assert.True(t, strings.HasPrefix(s, "event: message_delta\ndata: "), "event: line must survive: %q", s)
		assert.True(t, strings.HasSuffix(s, "\n\n"))
		var decoded struct {
			Type  string `json:"type"`
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				OutputTokens int     `json:"output_tokens"`
				Cost         float64 `json:"cost"`
			} `json:"usage"`
		}
		jsonPart := strings.TrimSuffix(strings.TrimPrefix(s, "event: message_delta\ndata: "), "\n\n")
		require.NoError(t, json.Unmarshal([]byte(jsonPart), &decoded))
		assert.Equal(t, "message_delta", decoded.Type)
		assert.Equal(t, "end_turn", decoded.Delta.StopReason)
		assert.Equal(t, 5, decoded.Usage.OutputTokens)
		assert.InDelta(t, 0.0007, decoded.Usage.Cost, 1e-12)
	})

	// Regression: handleNativeResponsesStreaming/handleResponsesAPIStreaming
	// go through handleTransformedStreaming/tokenCapturingWriter too, but
	// their wire shape nests usage at response.usage, not top-level "usage".
	// The old implementation only ever tried the top-level key, so native
	// Responses API streaming never got a cost at all.
	t.Run("falls back to nested response.usage for Responses-API-shaped chunks", func(t *testing.T) {
		payload := []byte(`{"type":"response.completed","response":{"id":"r1","usage":{"input_tokens":10,"output_tokens":5}}}`)
		chunk := []byte("data: " + string(payload) + "\n\n")
		out, ok := injectCostIntoSSEUsageChunk(chunk, [][]byte{payload}, 0.0042)
		require.True(t, ok)

		var decoded struct {
			Response struct {
				Usage struct {
					Cost float64 `json:"cost"`
				} `json:"usage"`
			} `json:"response"`
		}
		jsonPart := strings.TrimSuffix(strings.TrimPrefix(string(out), "data: "), "\n\n")
		require.NoError(t, json.Unmarshal([]byte(jsonPart), &decoded))
		assert.InDelta(t, 0.0042, decoded.Response.Usage.Cost, 1e-12)
	})
}

func TestAppendCostField(t *testing.T) {
	t.Run("inserts before the closing brace with a leading comma", func(t *testing.T) {
		out, ok := appendCostField([]byte(`{"a":1,"b":2}`), 0.5)
		require.True(t, ok)
		assert.JSONEq(t, `{"a":1,"b":2,"cost":0.5}`, string(out))
	})

	t.Run("an empty object gets no leading comma", func(t *testing.T) {
		out, ok := appendCostField([]byte(`{}`), 0.5)
		require.True(t, ok)
		assert.JSONEq(t, `{"cost":0.5}`, string(out))
	})

	t.Run("formats like X-Total-Cost's strconv.FormatFloat, no exponent, no float64 noise", func(t *testing.T) {
		// 1000*0.000001 + 1000*0.000002 computed as float64 addition --
		// exactly the kind of value that json.Marshal(float64) could print
		// with 17 significant digits (binary floating-point artifacts);
		// strconv.FormatFloat(_, 'f', -1, 64) prints the shortest decimal
		// that round-trips instead.
		cost := 1000*0.000001 + 1000*0.000002
		out, ok := appendCostField([]byte(`{}`), cost)
		require.True(t, ok)
		assert.Contains(t, string(out), `"cost":0.003}`)

		small, ok := appendCostField([]byte(`{}`), 0.0000001)
		require.True(t, ok)
		assert.NotContains(t, string(small), "e-", "must not use exponent notation")
	})

	t.Run("rejects non-object or garbage bytes", func(t *testing.T) {
		_, ok := appendCostField([]byte(`[1,2]`), 1.0)
		assert.False(t, ok)
		_, ok = appendCostField([]byte(`{not json}`), 1.0)
		assert.False(t, ok)
	})
}

func TestSpliceBytesFrom(t *testing.T) {
	out, ok := spliceBytesFrom([]byte("abc-DEF-ghi"), 0, []byte("DEF"), []byte("xyzw"))
	require.True(t, ok)
	assert.Equal(t, "abc-xyzw-ghi", string(out))

	_, ok = spliceBytesFrom([]byte("abc"), 0, []byte("not-there"), []byte("x"))
	assert.False(t, ok)

	// Regression (Bug 1): searching from byte 0 of a document that repeats
	// the exact same byte sequence earlier would patch the wrong
	// occurrence. Searching from a tracked offset instead finds the right
	// one even though "DEF" appears twice.
	t.Run("searches only from the given offset, ignoring an earlier identical match", func(t *testing.T) {
		frame := []byte("DEF-abc-DEF-ghi")
		out, ok := spliceBytesFrom(frame, 8, []byte("DEF"), []byte("xyzw"))
		require.True(t, ok)
		assert.Equal(t, "DEF-abc-xyzw-ghi", string(out))
	})

	t.Run("from out of range is rejected", func(t *testing.T) {
		_, ok := spliceBytesFrom([]byte("abc"), 10, []byte("a"), []byte("b"))
		assert.False(t, ok)
		_, ok = spliceBytesFrom([]byte("abc"), -1, []byte("a"), []byte("b"))
		assert.False(t, ok)
	})
}

func TestTopLevelEntries(t *testing.T) {
	t.Run("only top-level keys, not ones nested deeper under the same name", func(t *testing.T) {
		entries, ok := topLevelEntries([]byte(`{"metadata":{"usage":{"prompt_tokens":10}},"usage":{"prompt_tokens":20}}`))
		require.True(t, ok)
		usage, has := entries["usage"]
		require.True(t, has)
		assert.JSONEq(t, `{"prompt_tokens":20}`, string(usage.value))
		// searchFrom must point at (or after) this top-level "usage"'s own
		// value, not the one nested under "metadata" earlier in the bytes.
		assert.GreaterOrEqual(t, usage.searchFrom, 45)
	})

	t.Run("a repeated key resolves to the last occurrence, matching a standard decoder", func(t *testing.T) {
		entries, ok := topLevelEntries([]byte(`{"cost":1,"cost":2}`))
		require.True(t, ok)
		cost, has := entries["cost"]
		require.True(t, has)
		assert.Equal(t, "2", string(bytes.TrimSpace(cost.value)))
	})

	t.Run("rejects non-object input", func(t *testing.T) {
		_, ok := topLevelEntries([]byte(`[1,2,3]`))
		assert.False(t, ok)
		_, ok = topLevelEntries([]byte(`not json`))
		assert.False(t, ok)
	})
}

// Regression: a prior implementation decode+remarshaled the whole body via
// encoding/json, which re-sorts map keys and HTML-escapes "<"/">"/"&"
// anywhere in the document (Marshal's default) -- not just inside "usage".
// A message containing those characters, or a body whose fields weren't
// already alphabetical, would come back byte-different for reasons that
// have nothing to do with cost.
func TestInjectCostIntoJSONUsage_PreservesUnrelatedBytes(t *testing.T) {
	body := []byte(`{"id":"resp1","choices":[{"message":{"content":"if (a < b && b > c) { ok(); }"}}],"usage":{"prompt_tokens":10}}`)
	out, ok := injectCostIntoJSONUsage(body, 0.0001)
	require.True(t, ok)
	assert.Contains(t, string(out), `"content":"if (a < b && b > c) { ok(); }"`,
		"content containing <, >, & must survive unescaped")
	assert.True(t, strings.HasPrefix(string(out), `{"id":"resp1","choices":`),
		"field order before usage must be untouched: %q", out)
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

	// Regression: extractTokenUsageFromPayloads reads a usage object's own
	// "cost" key as the provider's reported price (ProviderCostUSD). This
	// reader's Read() returns the already-cost-injected line, and
	// streamToClient's onChunk/onLine observer (simulated below, faithfully
	// mirroring handleStreamingWithTokens's own closure) extracts from
	// exactly what Read() returned -- so without the
	// snapshot-before/restore-on-next-Read correction, AIR's own injected
	// usage.cost would end up misreported as what the *provider* charged,
	// in spend metadata and the Kafka event.
	t.Run("never pollutes ProviderCostUSD even though the observer reads the injected bytes", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		r := newCostInjectingSSEReader(strings.NewReader(stream), p, logCtx, converter.TokenUsageExtractionOptions{})

		var usageLines streamUsageLines
		var payloadBuf [][]byte
		onLine := func(line []byte) {
			payloadBuf = splitSSEPayloads(line, payloadBuf)
			if usage := extractTokenUsageFromPayloads(payloadBuf, converter.TokenUsageExtractionOptions{}); usage != nil {
				if logCtx.TokenUsage == nil {
					logCtx.TokenUsage = &converter.TokenUsage{}
				}
				logCtx.TokenUsage.MergeNonZero(usage)
			}
		}

		buf := make([]byte, 7) // deliberately small and line-misaligned
		for {
			n, err := r.Read(buf)
			if n > 0 {
				usageLines.Observe(buf[:n], onLine)
			}
			if err != nil {
				require.ErrorIs(t, err, io.EOF)
				usageLines.Finalize(onLine)
				break
			}
		}

		require.NotNil(t, logCtx.TokenUsage)
		assert.Zero(t, logCtx.TokenUsage.ProviderCostUSD,
			"AIR's own injected usage.cost must never be mistaken for a provider-reported cost")
		// The cost itself must still have reached the client.
		assert.Equal(t, 1, strings.Count(string(mustReadAll(t, newCostInjectingSSEReader(strings.NewReader(stream), p, testLogCtxWithFlatPrice(), converter.TokenUsageExtractionOptions{}))), `"cost"`))
	})

	// A genuine provider-reported cost (an aggregator-style upstream, in
	// non-compat mode where only the LiteLLM transform would otherwise
	// strip it) must not be lost just because the guard also has to
	// prevent AIR's own injected figure from being mistaken for it. The
	// wire still shows AIR's own billed cost (the client is billed by the
	// router, not by whatever the upstream charged AIR) but
	// ProviderCostUSD -- internal-only, used for provider_reported_cost in
	// spend metadata and the Kafka event -- must end up holding the real
	// 9.99, not 0 and not AIR's own figure.
	t.Run("preserves a genuine provider-reported cost in ProviderCostUSD", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		withProviderCost := "data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150,\"cost\":9.99}}\n\n"
		r := newCostInjectingSSEReader(strings.NewReader(withProviderCost), p, logCtx, converter.TokenUsageExtractionOptions{})

		var usageLines streamUsageLines
		var payloadBuf [][]byte
		onLine := func(line []byte) {
			payloadBuf = splitSSEPayloads(line, payloadBuf)
			if usage := extractTokenUsageFromPayloads(payloadBuf, converter.TokenUsageExtractionOptions{}); usage != nil {
				if logCtx.TokenUsage == nil {
					logCtx.TokenUsage = &converter.TokenUsage{}
				}
				logCtx.TokenUsage.MergeNonZero(usage)
			}
		}

		buf := make([]byte, 7)
		var forwarded []byte
		for {
			n, err := r.Read(buf)
			if n > 0 {
				forwarded = append(forwarded, buf[:n]...)
				usageLines.Observe(buf[:n], onLine)
			}
			if err != nil {
				require.ErrorIs(t, err, io.EOF)
				usageLines.Finalize(onLine)
				break
			}
		}

		require.NotNil(t, logCtx.TokenUsage)
		assert.InDelta(t, 9.99, logCtx.TokenUsage.ProviderCostUSD, 1e-12,
			"the genuine provider-reported cost must survive for internal reconciliation")
		assert.NotContains(t, string(forwarded), "9.99",
			"the wire value is still AIR's own billed cost, not the provider's")
	})
}

func mustReadAll(t *testing.T, r io.Reader) []byte {
	t.Helper()
	out, err := io.ReadAll(r)
	require.NoError(t, err)
	return out
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

	// Same regression as costInjectingSSEReader's own ProviderCostUSD test,
	// for the nested response.usage shape: handlePassthroughResponsesStreaming's
	// own onChunk closure calls MergeUsageExtensions(ExtractTokenUsageWithOptions(
	// payload, ...)) on exactly the bytes Read() returned, which is where
	// ProviderCostUSD gets read from response.usage.cost when present.
	t.Run("never pollutes ProviderCostUSD even though the observer reads the injected bytes", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		r := newResponsesAPICostInjectingReader(strings.NewReader(stream), p, logCtx, opts, true)

		lines := newResponsesStreamLines()
		onLine := func(line []byte) {
			jsonData := responsesStreamDataPayload(line)
			if jsonData == nil {
				return
			}
			event, ok := parseResponsesTerminalEvent(jsonData)
			if !ok {
				return
			}
			usage := event.typedUsage()
			if usage == nil {
				return
			}
			payload := bytes.Clone(jsonData)
			if logCtx.TokenUsage == nil {
				logCtx.TokenUsage = &converter.TokenUsage{}
			}
			logCtx.TokenUsage.PromptTokens = usage.InputTokens
			logCtx.TokenUsage.CompletionTokens = usage.OutputTokens
			logCtx.TokenUsage.MergeUsageExtensions(converter.ExtractTokenUsageWithOptions(payload, converter.TokenUsageExtractionOptions{}))
		}

		buf := make([]byte, 7)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				lines.Observe(buf[:n], onLine)
			}
			if err != nil {
				require.ErrorIs(t, err, io.EOF)
				lines.Finalize(onLine)
				break
			}
		}

		require.NotNil(t, logCtx.TokenUsage)
		assert.Zero(t, logCtx.TokenUsage.ProviderCostUSD,
			"AIR's own injected response.usage.cost must never be mistaken for a provider-reported cost")
	})

	// Same genuine-provider-cost-preservation regression as
	// costInjectingSSEReader's own test, for the nested response.usage shape.
	t.Run("preserves a genuine provider-reported cost in ProviderCostUSD", func(t *testing.T) {
		p := &Proxy{}
		logCtx := testLogCtxWithFlatPrice()
		withProviderCost := "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{" +
			"\"input_tokens\":100,\"output_tokens\":50,\"total_tokens\":150," +
			"\"input_tokens_details\":{\"cached_tokens\":0}," +
			"\"output_tokens_details\":{\"reasoning_tokens\":0}," +
			"\"cost\":9.99" +
			"}}}\n\n"
		r := newResponsesAPICostInjectingReader(strings.NewReader(withProviderCost), p, logCtx, opts, true)

		lines := newResponsesStreamLines()
		onLine := func(line []byte) {
			jsonData := responsesStreamDataPayload(line)
			if jsonData == nil {
				return
			}
			event, ok := parseResponsesTerminalEvent(jsonData)
			if !ok {
				return
			}
			usage := event.typedUsage()
			if usage == nil {
				return
			}
			payload := bytes.Clone(jsonData)
			if logCtx.TokenUsage == nil {
				logCtx.TokenUsage = &converter.TokenUsage{}
			}
			logCtx.TokenUsage.PromptTokens = usage.InputTokens
			logCtx.TokenUsage.CompletionTokens = usage.OutputTokens
			logCtx.TokenUsage.MergeUsageExtensions(converter.ExtractTokenUsageWithOptions(payload, converter.TokenUsageExtractionOptions{}))
		}

		buf := make([]byte, 7)
		var forwarded []byte
		for {
			n, err := r.Read(buf)
			if n > 0 {
				forwarded = append(forwarded, buf[:n]...)
				lines.Observe(buf[:n], onLine)
			}
			if err != nil {
				require.ErrorIs(t, err, io.EOF)
				lines.Finalize(onLine)
				break
			}
		}

		require.NotNil(t, logCtx.TokenUsage)
		assert.InDelta(t, 9.99, logCtx.TokenUsage.ProviderCostUSD, 1e-12,
			"the genuine provider-reported cost must survive for internal reconciliation")
		assert.NotContains(t, string(forwarded), "9.99",
			"the wire value is still AIR's own billed cost, not the provider's")
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
