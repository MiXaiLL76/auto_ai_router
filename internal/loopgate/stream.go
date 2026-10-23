package loopgate

import (
	"bytes"
	"hash/maphash"
	"time"

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

type chatChunk struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Content          *string `json:"content"`
			ReasoningContent *string `json:"reasoning_content"`
			Reasoning        *string `json:"reasoning"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

var deltaKey = []byte(`"delta"`)

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
