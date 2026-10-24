package proxy

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	json "github.com/goccy/go-json"
)

// chatStreamAggregator rebuilds the chat.completion a non-streamed request
// would have received from the chat.completion.chunk events of the stream
// the loop gate asked for instead:
//
//   - text deltas (content, reasoning_content / reasoning, refusal) are
//     concatenated per choice; the reasoning field keeps the name the deltas
//     used;
//   - tool_calls are assembled by index, arguments concatenated;
//   - content is null when a choice made tool calls and its text is only
//     whitespace (the streaming tool parser emits blanks before a call), as
//     non-streamed vLLM returns;
//   - usage comes from the final chunk;
//   - an error event inside the stream becomes the response: its status and
//     body, not a 200.
type chatStreamAggregator struct {
	id                string
	created           int64
	model             string
	systemFingerprint json.RawMessage
	usage             json.RawMessage
	loopGate          *loopGateMarker
	choices           map[int]*aggChoice

	errStatus int
	errBody   []byte
	sawChunk  bool
}

type aggChoice struct {
	index          int
	role           string
	content        strings.Builder
	contentSeen    bool
	reasoning      strings.Builder
	reasoningField string
	refusal        strings.Builder
	refusalSeen    bool
	toolCalls      map[int]*aggToolCall
	finishReason   *string
	stopReason     json.RawMessage
	logprobs       []json.RawMessage
	logprobsSeen   bool
}

type aggToolCall struct {
	index     int
	id        string
	typ       string
	name      strings.Builder
	arguments strings.Builder
}

type aggStreamChunk struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	SystemFingerprint json.RawMessage `json:"system_fingerprint"`
	Usage             json.RawMessage `json:"usage"`
	Error             json.RawMessage `json:"error"`
	LoopGate          *loopGateMarker `json:"air_loop_gate"`
	Choices           []struct {
		Index int `json:"index"`
		Delta struct {
			Role             *string `json:"role"`
			Content          *string `json:"content"`
			ReasoningContent *string `json:"reasoning_content"`
			Reasoning        *string `json:"reasoning"`
			Refusal          *string `json:"refusal"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      *string `json:"name"`
					Arguments *string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string         `json:"finish_reason"`
		StopReason   json.RawMessage `json:"stop_reason"`
		Logprobs     *struct {
			Content []json.RawMessage `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
}

func newChatStreamAggregator() *chatStreamAggregator {
	return &chatStreamAggregator{choices: make(map[int]*aggChoice)}
}

// consume reads the whole SSE stream. A read error is returned as is, and
// ErrResponseBodyTooLarge once more than limit bytes were read (limit <= 0 =
// no limit); an error event ends consumption and is kept for the response.
func (a *chatStreamAggregator) consume(r io.Reader, limit int64) error {
	br := bufio.NewReaderSize(r, 64*1024)
	var read int64
	for {
		line, err := br.ReadBytes('\n')
		read += int64(len(line))
		if limit > 0 && read > limit {
			return ErrResponseBodyTooLarge
		}
		if len(line) > 0 {
			if payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), sseDataPrefix); ok {
				payload = bytes.TrimSpace(payload)
				if len(payload) > 0 && !bytes.Equal(payload, sseDoneSentinel) {
					a.add(payload)
					if a.errStatus != 0 {
						return nil
					}
				}
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func (a *chatStreamAggregator) add(payload []byte) {
	var chunk aggStreamChunk
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return
	}
	if status, ok := chatStreamErrorStatus(chunk, payload); ok {
		a.errStatus = status
		a.errBody = append([]byte(nil), payload...)
		return
	}
	a.sawChunk = true
	if chunk.ID != "" {
		a.id = chunk.ID
	}
	if chunk.Created != 0 {
		a.created = chunk.Created
	}
	if chunk.Model != "" {
		a.model = chunk.Model
	}
	if !isJSONNull(chunk.SystemFingerprint) {
		a.systemFingerprint = append(json.RawMessage(nil), chunk.SystemFingerprint...)
	}
	if !isJSONNull(chunk.Usage) {
		a.usage = append(json.RawMessage(nil), chunk.Usage...)
	}
	if chunk.LoopGate != nil {
		a.loopGate = chunk.LoopGate
	}
	for i := range chunk.Choices {
		in := &chunk.Choices[i]
		c := a.choices[in.Index]
		if c == nil {
			c = &aggChoice{index: in.Index, role: "assistant", toolCalls: make(map[int]*aggToolCall)}
			a.choices[in.Index] = c
		}
		d := &in.Delta
		if d.Role != nil && *d.Role != "" {
			c.role = *d.Role
		}
		if d.Content != nil {
			c.content.WriteString(*d.Content)
			c.contentSeen = true
		}
		switch {
		case d.ReasoningContent != nil:
			c.reasoning.WriteString(*d.ReasoningContent)
			if c.reasoningField == "" {
				c.reasoningField = "reasoning_content"
			}
		case d.Reasoning != nil:
			c.reasoning.WriteString(*d.Reasoning)
			if c.reasoningField == "" {
				c.reasoningField = "reasoning"
			}
		}
		if d.Refusal != nil {
			c.refusal.WriteString(*d.Refusal)
			c.refusalSeen = true
		}
		for _, tc := range d.ToolCalls {
			call := c.toolCalls[tc.Index]
			if call == nil {
				call = &aggToolCall{index: tc.Index, typ: "function"}
				c.toolCalls[tc.Index] = call
			}
			if tc.ID != "" {
				call.id = tc.ID
			}
			if tc.Type != "" {
				call.typ = tc.Type
			}
			if tc.Function.Name != nil {
				call.name.WriteString(*tc.Function.Name)
			}
			if tc.Function.Arguments != nil {
				call.arguments.WriteString(*tc.Function.Arguments)
			}
		}
		if in.FinishReason != nil && *in.FinishReason != "" {
			fr := *in.FinishReason
			c.finishReason = &fr
		}
		if !isJSONNull(in.StopReason) {
			c.stopReason = append(json.RawMessage(nil), in.StopReason...)
		}
		if in.Logprobs != nil {
			c.logprobsSeen = true
			c.logprobs = append(c.logprobs, in.Logprobs.Content...)
		}
	}
}

// chatStreamErrorStatus recognizes an error event: vLLM sends
// {"error": {"message", "type", "code"}} (older: {"object": "error", ...}).
func chatStreamErrorStatus(chunk aggStreamChunk, payload []byte) (int, bool) {
	isErr := chunk.Object == "error" || !isJSONNull(chunk.Error)
	if !isErr {
		return 0, false
	}
	var codes struct {
		Code  json.RawMessage `json:"code"`
		Error struct {
			Code json.RawMessage `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(payload, &codes)
	for _, raw := range []json.RawMessage{codes.Error.Code, codes.Code} {
		var code int
		if json.Unmarshal(raw, &code) == nil && code >= 400 && code <= 599 {
			return code, true
		}
	}
	return statusCodeFromProviderStreamError(string(payload)), true
}

// choice0 returns choice 0, the one the loop gate watches; nil when the
// stream had none.
func (a *chatStreamAggregator) choice0() *aggChoice {
	return a.choices[0]
}

type aggCompletion struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	Choices           []aggChoiceOut  `json:"choices"`
	Usage             json.RawMessage `json:"usage,omitempty"`
	SystemFingerprint json.RawMessage `json:"system_fingerprint,omitempty"`
	LoopGate          *loopGateMarker `json:"air_loop_gate,omitempty"`
}

type aggChoiceOut struct {
	Index        int             `json:"index"`
	Message      aggMessage      `json:"message"`
	Logprobs     json.RawMessage `json:"logprobs"`
	FinishReason *string         `json:"finish_reason"`
	StopReason   json.RawMessage `json:"stop_reason,omitempty"`
}

type aggMessage struct {
	Role             string           `json:"role"`
	Content          *string          `json:"content"`
	ReasoningContent *string          `json:"reasoning_content,omitempty"`
	Reasoning        *string          `json:"reasoning,omitempty"`
	Refusal          *string          `json:"refusal,omitempty"`
	ToolCalls        []aggToolCallOut `json:"tool_calls,omitempty"`
}

type aggToolCallOut struct {
	ID       string `json:"id,omitempty"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

var aggLogprobsNull = json.RawMessage("null")

// completion returns the rebuilt chat.completion body.
func (a *chatStreamAggregator) completion() ([]byte, error) {
	out := aggCompletion{
		ID:                a.id,
		Object:            "chat.completion",
		Created:           a.created,
		Model:             a.model,
		Usage:             a.usage,
		SystemFingerprint: a.systemFingerprint,
		LoopGate:          a.loopGate,
	}
	indexes := make([]int, 0, len(a.choices))
	for i := range a.choices {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)
	out.Choices = make([]aggChoiceOut, 0, len(indexes))
	for _, i := range indexes {
		out.Choices = append(out.Choices, a.choices[i].out())
	}
	return json.Marshal(out)
}

func (c *aggChoice) out() aggChoiceOut {
	msg := aggMessage{Role: c.role}
	content := c.content.String()
	switch {
	case len(c.toolCalls) > 0 && strings.TrimSpace(content) == "":
		// content stays null
	case c.contentSeen || len(c.toolCalls) == 0:
		msg.Content = &content
	}
	if c.reasoningField != "" {
		reasoning := c.reasoning.String()
		if c.reasoningField == "reasoning" {
			msg.Reasoning = &reasoning
		} else {
			msg.ReasoningContent = &reasoning
		}
	}
	if c.refusalSeen {
		refusal := c.refusal.String()
		msg.Refusal = &refusal
	}
	if len(c.toolCalls) > 0 {
		indexes := make([]int, 0, len(c.toolCalls))
		for i := range c.toolCalls {
			indexes = append(indexes, i)
		}
		slices.Sort(indexes)
		for _, i := range indexes {
			tc := c.toolCalls[i]
			call := aggToolCallOut{ID: tc.id, Type: tc.typ}
			call.Function.Name = tc.name.String()
			call.Function.Arguments = tc.arguments.String()
			msg.ToolCalls = append(msg.ToolCalls, call)
		}
	}
	choice := aggChoiceOut{Index: c.index, Message: msg, Logprobs: aggLogprobsNull, FinishReason: c.finishReason, StopReason: c.stopReason}
	if c.logprobsSeen {
		lp, err := json.Marshal(struct {
			Content []json.RawMessage `json:"content"`
		}{Content: c.logprobs})
		if err == nil {
			choice.Logprobs = lp
		}
	}
	return choice
}

// aggregatedResponse returns resp with its body replaced by body as a
// non-streamed JSON response with the given status.
func aggregatedResponse(resp *http.Response, status int, body []byte) *http.Response {
	header := resp.Header.Clone()
	header.Set("Content-Type", "application/json")
	header.Del("Content-Length")
	header.Del("Content-Encoding")
	header.Del("Transfer-Encoding")
	return &http.Response{
		Status:        fmt.Sprintf("%d %s", status, http.StatusText(status)),
		StatusCode:    status,
		Proto:         resp.Proto,
		ProtoMajor:    resp.ProtoMajor,
		ProtoMinor:    resp.ProtoMinor,
		Header:        header,
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		Request:       resp.Request,
	}
}
