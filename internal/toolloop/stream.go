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
// It returns the stream's bytes, the tool calls of the ROUTER'S OWN and the
// usage the stream carried.
//
// The calls are filtered here, and only here: a caller that got the raw
// list could mistake a client's own tool call for the router's and loop on a
// request it has no tool to run -- sending the very same request upstream a
// second time and billing the client for both sends.
//
// A stream that breaks mid-way returns the bytes that DID arrive together
// with the error, not an empty body: the provider's connection can die after
// it has already streamed most of its answer (measured against Moonshot and
// DeepSeek, whose streams break at a fixed 30 s and 45 s), and throwing the
// received text away gives the client a bare 502 instead of the answer the
// model had already produced. The caller decides what to do with the error;
// it can no longer assume the bytes are empty.
//
// The tool calls are returned with them as well, so a caller can tell whether
// the partial stream carries a call of the ROUTER'S OWN -- one the client
// never declared, and so never one to relay.
func CollectStreamToolCalls(resp *http.Response, runner Runner) ([]byte, []ToolCall, *openai.OpenAIUsage, error) {
	if resp == nil || resp.Body == nil {
		return nil, nil, nil, errors.New("toolloop: no stream to read")
	}
	defer func() { _ = resp.Body.Close() }()

	var buffered bytes.Buffer
	_, err := io.Copy(&buffered, resp.Body)
	raw := buffered.Bytes()

	calls, usage := callsFromStream(raw)
	if err != nil {
		// The bytes that arrived are still the answer so far: hand them over
		// with the error rather than as nothing.
		return raw, OwnedCalls(calls, runner), usage, err
	}

	return raw, OwnedCalls(calls, runner), usage, nil
}

// HasUnnamedToolCall reports whether the stream carries a tool call whose
// name never arrived.
//
// A streamed tool call carries its name on the first piece and its
// arguments over the next ones, so a stream that breaks in between leaves
// the call nameless. An unnamed call cannot be told apart from one of the
// router's own -- and one of the router's own must never reach the client,
// which never declared it. So a nameless call counts as the router's.
func HasUnnamedToolCall(raw []byte) bool {
	calls, _ := callsFromStream(raw)
	for _, call := range calls {
		if call.Name == "" {
			return true
		}
	}
	return false
}

// CallsFromStream reads the router's own tool calls out of a buffered stream.
func CallsFromStream(raw []byte, runner Runner) []ToolCall {
	calls, _ := callsFromStream(raw)
	return OwnedCalls(calls, runner)
}

// ReadStream reads a response's stream whole and returns its bytes and the
// usage it carried.
//
// It is for the step that reads the model's NEXT answer: the calls in it are
// not needed there, only the stream itself and its figures.
//
// Unlike CollectStreamToolCalls it keeps the old contract on a broken
// stream: the bytes are dropped and only the error comes back. A step of the
// loop that breaks half-way is not an answer -- it may carry the tool call
// the client never declared, which must not reach the client -- so the
// partial stream of a step is never relayed.
func ReadStream(resp *http.Response) ([]byte, *openai.OpenAIUsage, error) {
	if resp == nil || resp.Body == nil {
		return nil, nil, errors.New("toolloop: no stream to read")
	}
	defer func() { _ = resp.Body.Close() }()

	var buffered bytes.Buffer
	if _, err := io.Copy(&buffered, resp.Body); err != nil {
		return nil, nil, err
	}
	raw := buffered.Bytes()
	_, usage := callsFromStream(raw)
	return raw, usage, nil
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
