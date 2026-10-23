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
	}
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
	if chunk.ID != "" {
		d.meta.ID = chunk.ID
	}
	if chunk.Created != 0 {
		d.meta.Created = chunk.Created
	}
	if chunk.Model != "" {
		d.meta.Model = chunk.Model
	}
	for i := range chunk.Choices {
		choice := &chunk.Choices[i]
		if choice.Index != 0 {
			continue
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			d.finishReason = *choice.FinishReason
		}
		if choice.Delta.Content != nil {
			d.content.Feed(*choice.Delta.Content)
			if !d.answered && strings.TrimSpace(*choice.Delta.Content) != "" {
				d.answered = true
			}
		}
		if len(choice.Delta.ToolCalls) > 0 && !bytes.Equal(choice.Delta.ToolCalls, jsonNull) && !bytes.Equal(choice.Delta.ToolCalls, emptyJSONArray) {
			d.answered = true
		}
		// vLLM names the field reasoning_content (older) or reasoning (newer).
		if choice.Delta.ReasoningContent != nil {
			d.reasoning.Feed(*choice.Delta.ReasoningContent)
		} else if choice.Delta.Reasoning != nil {
			d.reasoning.Feed(*choice.Delta.Reasoning)
		}
	}
	return d.check(now)
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
