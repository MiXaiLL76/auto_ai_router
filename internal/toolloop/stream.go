package toolloop

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/sse"
)

// CollectStreamToolCalls reads a Chat Completions SSE stream and returns the
// tool calls the model made in it.
//
// The stream is read whole: the router has to know whether the model asked
// for a tool before it can decide whether to relay the stream to the client
// as it is or to run the tool first. A stream that ends without a tool call
// is replayed to the client byte for byte, so the client sees exactly what
// the provider sent.
//
// It returns the stream's bytes, the tool calls and the usage the stream
// carried. The calls are not filtered: the caller decides which of them it
// owns (see CallsFromStream).
func CollectStreamToolCalls(resp *http.Response) ([]byte, []ToolCall, *openai.OpenAIUsage, error) {
	if resp == nil || resp.Body == nil {
		return nil, nil, nil, errors.New("toolloop: no stream to read")
	}
	defer func() { _ = resp.Body.Close() }()

	var buffered bytes.Buffer
	if _, err := io.Copy(&buffered, resp.Body); err != nil {
		return nil, nil, nil, err
	}
	raw := buffered.Bytes()

	calls, usage := callsFromStream(raw)
	return raw, calls, usage, nil
}

// CallsFromStream reads the tool calls out of a buffered stream.
func CallsFromStream(raw []byte, runner Runner) []ToolCall {
	calls, _ := callsFromStream(raw)
	owned := make([]ToolCall, 0, len(calls))
	for _, call := range calls {
		if runner.Owns(call.Name) {
			owned = append(owned, call)
		}
	}
	return owned
}

// callsFromStream folds a stream's tool call pieces back into one call per
// index, and reads the stream's usage.
//
// The pieces of one streamed tool call arrive under the same index: the id
// and the name only on the first piece, the arguments split across the rest.
// The index is therefore the key -- the id is not (a later piece may carry it
// empty, and two calls may share one id in a provider's stream).
func callsFromStream(raw []byte) ([]ToolCall, *openai.OpenAIUsage) {
	calls := map[string]*ToolCall{}
	var order []string
	usage := &openai.OpenAIUsage{}
	for _, payload := range sse.SplitPayloads(raw, nil) {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *openai.OpenAIUsage `json:"usage"`
		}
		if err := json.Unmarshal(payload, &chunk); err != nil {
			continue
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			for _, delta := range choice.Delta.ToolCalls {
				key := strconv.Itoa(delta.Index)
				call, seen := calls[key]
				if !seen {
					call = &ToolCall{ID: delta.ID}
					calls[key] = call
					order = append(order, key)
				}
				if delta.ID != "" {
					call.ID = delta.ID
				}
				if delta.Function.Name != "" {
					call.Name = delta.Function.Name
				}
				call.Arguments += delta.Function.Arguments
			}
		}
	}

	folded := make([]ToolCall, 0, len(order))
	for _, key := range order {
		folded = append(folded, *calls[key])
	}
	return folded, usage
}
