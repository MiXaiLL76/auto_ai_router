package loopgate

import (
	"bytes"
	"hash/maphash"
	"strings"
	"time"
	"unicode"

	"github.com/goccy/go-json"
)

// Channel names, as reported in logs and metrics.
const (
	ChannelContent   = "content"
	ChannelReasoning = "reasoning"
)

// Format is the wire format of a watched stream.
type Format int

const (
	// FormatNone means the stream is not watched.
	FormatNone Format = iota
	// FormatChat is chat.completion.chunk events (OpenAI Chat Completions).
	FormatChat
	// FormatResponses is OpenAI Responses API events (response.*).
	FormatResponses
	// FormatMessages is Anthropic Messages API events.
	FormatMessages
)

// String returns the format name used in logs.
func (f Format) String() string {
	switch f {
	case FormatChat:
		return "chat"
	case FormatResponses:
		return "responses"
	case FormatMessages:
		return "messages"
	default:
		return "none"
	}
}

// Firing is a channel's first firing.
type Firing struct {
	Channel string
	Fire
}

// Detector watches one streamed chat completion. The answer text and the
// reasoning text are counted separately; tool call arguments are not counted
// at all, so strict JSON-schema arguments never look like a loop. Only choice
// 0 is watched. Not safe for concurrent use.
type Detector struct {
	start        time.Time
	content      *Channel
	reasoning    *Channel
	finishReason string
	meta         ChunkMeta
	answered     bool
	openBlock    int // Messages: index of the content block being streamed, -1 = none
	sequence     int // Responses: highest sequence_number seen
}

// ChunkMeta identifies the completion a stream belongs to, as the last chunk
// carrying each field reported it.
type ChunkMeta struct {
	ID      string
	Created int64
	Model   string
}

// NewDetector returns a detector whose elapsed time counts from start.
func NewDetector(cfg Config, start time.Time) *Detector {
	seed := maphash.MakeSeed()
	return &Detector{
		start:     start,
		content:   NewChannel(cfg, seed),
		reasoning: NewChannel(cfg, seed),
		openBlock: -1,
	}
}

// OpenBlock returns the index of the Messages content block being streamed.
func (d *Detector) OpenBlock() (int, bool) { return d.openBlock, d.openBlock >= 0 }

// Sequence returns the highest Responses sequence_number seen.
func (d *Detector) Sequence() int { return d.sequence }

// Observe feeds one SSE data payload in the given format.
func (d *Detector) Observe(format Format, payload []byte, now time.Time) []Firing {
	switch format {
	case FormatChat:
		return d.ObserveChatChunk(payload, now)
	case FormatResponses:
		return d.ObserveResponsesEvent(payload, now)
	case FormatMessages:
		return d.ObserveMessagesEvent(payload, now)
	default:
		return nil
	}
}

func (d *Detector) feedContent(text string) {
	d.content.Feed(text)
	if !d.answered && strings.TrimSpace(text) != "" {
		d.answered = true
	}
}

// ResponsesEvent is what the detector reads from one Responses API event, for
// a caller that has parsed the event itself (ObserveResponses).
type ResponsesEvent struct {
	Type           string
	SequenceNumber int
	Delta          *string
	ItemType       string             // the item of response.output_item.added
	Response       *ResponsesResponse // the response of response.* lifecycle events
}

// ResponsesResponse is the part of a Responses API response object the
// detector reads.
type ResponsesResponse struct {
	ID               string
	CreatedAt        int64
	Model            string
	Status           string
	IncompleteReason string
}

type responsesEvent struct {
	Type           string  `json:"type"`
	SequenceNumber int     `json:"sequence_number"`
	Delta          *string `json:"delta"`
	Item           *struct {
		Type string `json:"type"`
	} `json:"item"`
	Response *struct {
		ID                string `json:"id"`
		CreatedAt         int64  `json:"created_at"`
		Model             string `json:"model"`
		Status            string `json:"status"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
	} `json:"response"`
}

var typeKey = []byte(`"type"`)

// ObserveResponsesEvent feeds one Responses API event: output_text and
// refusal deltas are the answer, reasoning deltas (reasoning_text,
// reasoning_summary_text) the reasoning; a function call or any other
// non-message output item starts the answer without being counted.
func (d *Detector) ObserveResponsesEvent(payload []byte, now time.Time) []Firing {
	if !bytes.Contains(payload, typeKey) {
		return nil
	}
	var raw responsesEvent
	if err := json.Unmarshal(payload, &raw); err != nil {
		return nil
	}
	ev := ResponsesEvent{Type: raw.Type, SequenceNumber: raw.SequenceNumber, Delta: raw.Delta}
	if raw.Item != nil {
		ev.ItemType = raw.Item.Type
	}
	if r := raw.Response; r != nil {
		ev.Response = &ResponsesResponse{ID: r.ID, CreatedAt: r.CreatedAt, Model: r.Model, Status: r.Status}
		if r.IncompleteDetails != nil {
			ev.Response.IncompleteReason = r.IncompleteDetails.Reason
		}
	}
	return d.ObserveResponses(ev, now)
}

// ObserveResponses is ObserveResponsesEvent for an event the caller parsed.
func (d *Detector) ObserveResponses(ev ResponsesEvent, now time.Time) []Firing {
	if ev.SequenceNumber > d.sequence {
		d.sequence = ev.SequenceNumber
	}
	if r := ev.Response; r != nil {
		d.setMeta(ChunkMeta{ID: r.ID, Created: r.CreatedAt, Model: r.Model})
	}
	switch {
	case ev.Type == "response.output_text.delta" || ev.Type == "response.refusal.delta":
		if ev.Delta != nil {
			d.feedContent(*ev.Delta)
		}
	case strings.HasPrefix(ev.Type, "response.reasoning") && strings.HasSuffix(ev.Type, ".delta"):
		if ev.Delta != nil {
			d.reasoning.Feed(*ev.Delta)
		}
	case ev.Type == "response.function_call_arguments.delta":
		d.answered = true
	case ev.Type == "response.output_item.added":
		if ev.ItemType != "" && ev.ItemType != "reasoning" && ev.ItemType != "message" {
			d.answered = true
		}
	case ev.Type == "response.completed" || ev.Type == "response.incomplete" || ev.Type == "response.failed":
		if r := ev.Response; r != nil {
			d.finishReason = r.Status
			if r.IncompleteReason != "" {
				d.finishReason = r.IncompleteReason
			}
		}
	}
	return d.check(now)
}

// setMeta keeps the non-empty fields of m.
func (d *Detector) setMeta(m ChunkMeta) {
	if m.ID != "" {
		d.meta.ID = m.ID
	}
	if m.Created != 0 {
		d.meta.Created = m.Created
	}
	if m.Model != "" {
		d.meta.Model = m.Model
	}
}

type messagesEvent struct {
	Type    string `json:"type"`
	Index   *int   `json:"index"`
	Message *struct {
		ID    string `json:"id"`
		Model string `json:"model"`
	} `json:"message"`
	ContentBlock *struct {
		Type string `json:"type"`
	} `json:"content_block"`
	Delta *struct {
		Type       string  `json:"type"`
		Text       *string `json:"text"`
		Thinking   *string `json:"thinking"`
		StopReason *string `json:"stop_reason"`
	} `json:"delta"`
}

// ObserveMessagesEvent feeds one Anthropic Messages API event: text deltas
// are the answer, thinking deltas the reasoning; a tool_use block starts the
// answer without its input being counted.
func (d *Detector) ObserveMessagesEvent(payload []byte, now time.Time) []Firing {
	if !bytes.Contains(payload, typeKey) {
		return nil
	}
	var ev messagesEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil
	}
	switch ev.Type {
	case "message_start":
		if ev.Message != nil {
			d.meta.ID, d.meta.Model = ev.Message.ID, ev.Message.Model
		}
	case "content_block_start":
		if ev.Index != nil {
			d.openBlock = *ev.Index
		}
		if ev.ContentBlock != nil && ev.ContentBlock.Type != "text" && ev.ContentBlock.Type != "thinking" && ev.ContentBlock.Type != "redacted_thinking" {
			d.answered = true
		}
	case "content_block_delta":
		if ev.Delta == nil {
			break
		}
		switch ev.Delta.Type {
		case "text_delta":
			if ev.Delta.Text != nil {
				d.feedContent(*ev.Delta.Text)
			}
		case "thinking_delta":
			if ev.Delta.Thinking != nil {
				d.reasoning.Feed(*ev.Delta.Thinking)
			}
		case "input_json_delta":
			d.answered = true
		}
	case "content_block_stop":
		d.openBlock = -1
	case "message_delta":
		if ev.Delta != nil && ev.Delta.StopReason != nil {
			d.finishReason = *ev.Delta.StopReason
		}
	}
	return d.check(now)
}

// Content returns the answer text channel.
func (d *Detector) Content() *Channel { return d.content }

// Reasoning returns the reasoning text channel.
func (d *Detector) Reasoning() *Channel { return d.reasoning }

// FinishReason returns the finish_reason of choice 0, "" until it arrives.
func (d *Detector) FinishReason() string { return d.finishReason }

// AnswerStarted reports whether the stream carried answer text (anything but
// whitespace) or a tool call for choice 0. Until then the client has seen at
// most reasoning.
func (d *Detector) AnswerStarted() bool { return d.answered }

// Meta returns the id, created and model of the stream's chunks.
func (d *Detector) Meta() ChunkMeta { return d.meta }

// LoopingChannel returns the channel looping at ratio cut (reasoning first),
// or "" when neither is.
func (d *Detector) LoopingChannel(cut float64) string {
	switch {
	case d.reasoning.Looping(cut):
		return ChannelReasoning
	case d.content.Looping(cut):
		return ChannelContent
	default:
		return ""
	}
}

type chatChunk struct {
	ID      string `json:"id"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Content          *string         `json:"content"`
			ReasoningContent *string         `json:"reasoning_content"`
			Reasoning        *string         `json:"reasoning"`
			ToolCalls        json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

var (
	deltaKey       = []byte(`"delta"`)
	jsonNull       = []byte(`null`)
	emptyJSONArray = []byte(`[]`)
)

// ObserveChatChunk feeds the data payload of one chat.completion.chunk SSE
// event (without the "data: " prefix) and returns the firings it caused.
// Anything that is not a chat chunk is ignored.
func (d *Detector) ObserveChatChunk(payload []byte, now time.Time) []Firing {
	if !bytes.Contains(payload, deltaKey) {
		return nil
	}
	var chunk chatChunk
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return nil
	}
	d.setMeta(ChunkMeta{ID: chunk.ID, Created: chunk.Created, Model: chunk.Model})
	for i := range chunk.Choices {
		choice := &chunk.Choices[i]
		if choice.Index != 0 {
			continue
		}
		delta := ChatDelta{
			Content:   choice.Delta.Content,
			Reasoning: choice.Delta.ReasoningContent,
			ToolCalls: len(choice.Delta.ToolCalls) > 0 && !bytes.Equal(choice.Delta.ToolCalls, jsonNull) && !bytes.Equal(choice.Delta.ToolCalls, emptyJSONArray),
		}
		// vLLM names the field reasoning_content (older) or reasoning (newer).
		if delta.Reasoning == nil {
			delta.Reasoning = choice.Delta.Reasoning
		}
		if choice.FinishReason != nil {
			delta.FinishReason = *choice.FinishReason
		}
		d.feedChat(delta)
	}
	return d.check(now)
}

// ChatDelta is what the detector reads from choice 0 of one
// chat.completion.chunk, for a caller that has parsed the chunk itself
// (ObserveChat). The zero value is a chunk without choice 0.
type ChatDelta struct {
	Content      *string
	Reasoning    *string // reasoning_content, else reasoning
	ToolCalls    bool    // the delta carries a tool call
	FinishReason string
}

// ObserveChat is ObserveChatChunk for a chunk the caller parsed: meta are its
// id, created and model, delta its choice 0.
func (d *Detector) ObserveChat(meta ChunkMeta, delta ChatDelta, now time.Time) []Firing {
	d.setMeta(meta)
	d.feedChat(delta)
	return d.check(now)
}

func (d *Detector) feedChat(delta ChatDelta) {
	if delta.FinishReason != "" {
		d.finishReason = delta.FinishReason
	}
	if delta.Content != nil {
		d.feedContent(*delta.Content)
	}
	if delta.ToolCalls {
		d.answered = true
	}
	if delta.Reasoning != nil {
		d.reasoning.Feed(*delta.Reasoning)
	}
}

// ObserveText feeds the complete reasoning and answer text of a non-streamed
// response as if they had streamed (reasoning first), checking every few
// words, so a firing reports the word a stream would have fired on. Returns
// the firings.
func (d *Detector) ObserveText(content, reasoning, finishReason string, now time.Time) []Firing {
	var out []Firing
	feed := func(c *Channel, text string) {
		for len(text) > 0 {
			end := min(observeTextStep, len(text))
			if i := strings.IndexFunc(text[end:], unicode.IsSpace); i >= 0 {
				end += i + 1
			} else {
				end = len(text)
			}
			c.Feed(text[:end])
			text = text[end:]
			out = append(out, d.check(now)...)
		}
	}
	feed(d.reasoning, reasoning)
	feed(d.content, content)
	if strings.TrimSpace(content) != "" {
		d.answered = true
	}
	if finishReason != "" {
		d.finishReason = finishReason
	}
	return append(out, d.Finish(now)...)
}

// observeTextStep is how many bytes ObserveText feeds between checks.
const observeTextStep = 64

// Finish counts the words still pending at the end of the stream and returns
// the firings that caused.
func (d *Detector) Finish(now time.Time) []Firing {
	d.content.Flush()
	d.reasoning.Flush()
	return d.check(now)
}

func (d *Detector) check(now time.Time) []Firing {
	elapsed := now.Sub(d.start)
	var out []Firing
	if f := d.reasoning.Check(elapsed); f != nil {
		out = append(out, Firing{Channel: ChannelReasoning, Fire: *f})
	}
	if f := d.content.Check(elapsed); f != nil {
		out = append(out, Firing{Channel: ChannelContent, Fire: *f})
	}
	return out
}
