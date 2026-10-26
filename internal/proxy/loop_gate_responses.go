package proxy

import (
	"bytes"
	"io"
	"sort"
	"strings"

	json "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/loopgate"
)

// loopGateResponsesOutput rebuilds the output of a Responses API stream from
// its events, so a cut stream can close the items it left open and end with
// a response.incomplete that carries the whole output: the OpenAI SDKs build
// the final Response from that event, and the response store keeps it for
// previous_response_id. Kept only for an enforced passthrough Responses
// stream (the text is what the client already received).
type loopGateResponsesOutput struct {
	response    map[string]json.RawMessage // response object of response.created / in_progress
	inputTokens int                        // usage.input_tokens reported so far, 0 = unknown
	items       map[int]*loopGateResponsesItem
}

type loopGateResponsesItem struct {
	outputIndex int
	id          string
	typ         string
	item        map[string]json.RawMessage // from output_item.added, or the final one from output_item.done
	done        bool
	content     []*loopGateResponsesPart
	summary     []*loopGateResponsesPart
	arguments   strings.Builder
}

type loopGateResponsesPart struct {
	index int
	typ   string
	part  map[string]json.RawMessage // from content_part.added / reasoning_summary_part.added; nil = never announced
	text  strings.Builder
	done  bool
}

type loopGateResponsesEvent struct {
	Type           string                     `json:"type"`
	SequenceNumber int                        `json:"sequence_number"`
	OutputIndex    int                        `json:"output_index"`
	ContentIndex   int                        `json:"content_index"`
	SummaryIndex   int                        `json:"summary_index"`
	ItemID         string                     `json:"item_id"`
	Delta          string                     `json:"delta"`
	Item           map[string]json.RawMessage `json:"item"`
	Part           map[string]json.RawMessage `json:"part"`
	Response       map[string]json.RawMessage `json:"response"`
}

func newLoopGateResponsesOutput() *loopGateResponsesOutput {
	return &loopGateResponsesOutput{items: make(map[int]*loopGateResponsesItem)}
}

// add applies one Responses API event payload and returns what the loop gate
// detector reads of it, so the event is parsed once; ok is false when the
// payload is not an event.
func (r *loopGateResponsesOutput) add(payload []byte) (loopgate.ResponsesEvent, bool) {
	if !bytes.Contains(payload, sseTypeNeedle) {
		return loopgate.ResponsesEvent{}, false
	}
	var ev loopGateResponsesEvent
	if json.Unmarshal(payload, &ev) != nil {
		return loopgate.ResponsesEvent{}, false
	}
	r.apply(&ev)
	return ev.detected(), true
}

// detected returns what the loop gate detector reads of ev.
func (ev *loopGateResponsesEvent) detected() loopgate.ResponsesEvent {
	out := loopgate.ResponsesEvent{Type: ev.Type, SequenceNumber: ev.SequenceNumber, Delta: &ev.Delta}
	if ev.Item != nil {
		out.ItemType = rawString(ev.Item["type"])
	}
	if ev.Response != nil {
		var created int64
		_ = json.Unmarshal(ev.Response["created_at"], &created)
		var incomplete struct {
			Reason string `json:"reason"`
		}
		_ = json.Unmarshal(ev.Response["incomplete_details"], &incomplete)
		out.Response = &loopgate.ResponsesResponse{
			ID: rawString(ev.Response["id"]), CreatedAt: created, Model: rawString(ev.Response["model"]),
			Status: rawString(ev.Response["status"]), IncompleteReason: incomplete.Reason,
		}
	}
	return out
}

func (r *loopGateResponsesOutput) apply(ev *loopGateResponsesEvent) {
	if ev.Type == "response.created" || ev.Type == "response.in_progress" {
		if ev.Response != nil {
			r.response = ev.Response
			r.noteUsage(ev.Response["usage"])
		}
		return
	}
	if !strings.HasPrefix(ev.Type, "response.output_") && !strings.HasPrefix(ev.Type, "response.content_part.") &&
		!strings.HasPrefix(ev.Type, "response.refusal.") && !strings.HasPrefix(ev.Type, "response.reasoning_") &&
		!strings.HasPrefix(ev.Type, "response.function_call_arguments.") {
		return
	}
	it := r.item(ev.OutputIndex)
	if it.id == "" && ev.ItemID != "" {
		it.id = ev.ItemID
	}
	switch ev.Type {
	case "response.output_item.added":
		it.item = ev.Item
		it.id = rawString(ev.Item["id"])
		it.typ = rawString(ev.Item["type"])
	case "response.output_item.done":
		it.item, it.done = ev.Item, true
	case "response.content_part.added":
		it.part(&it.content, ev.ContentIndex, rawString(ev.Part["type"])).part = ev.Part
	case "response.content_part.done":
		it.part(&it.content, ev.ContentIndex, "").done = true
	case "response.output_text.delta":
		it.part(&it.content, ev.ContentIndex, "output_text").text.WriteString(ev.Delta)
	case "response.refusal.delta":
		it.part(&it.content, ev.ContentIndex, "refusal").text.WriteString(ev.Delta)
	case "response.reasoning_text.delta":
		it.part(&it.content, ev.ContentIndex, "reasoning_text").text.WriteString(ev.Delta)
	case "response.reasoning_summary_part.added":
		it.part(&it.summary, ev.SummaryIndex, "summary_text").part = ev.Part
	case "response.reasoning_summary_part.done":
		it.part(&it.summary, ev.SummaryIndex, "summary_text").done = true
	case "response.reasoning_summary_text.delta":
		it.part(&it.summary, ev.SummaryIndex, "summary_text").text.WriteString(ev.Delta)
	case "response.function_call_arguments.delta":
		it.arguments.WriteString(ev.Delta)
	}
}

func (r *loopGateResponsesOutput) noteUsage(raw json.RawMessage) {
	var usage struct {
		InputTokens int `json:"input_tokens"`
	}
	if len(raw) > 0 && json.Unmarshal(raw, &usage) == nil && usage.InputTokens > 0 {
		r.inputTokens = usage.InputTokens
	}
}

func (r *loopGateResponsesOutput) item(index int) *loopGateResponsesItem {
	it := r.items[index]
	if it == nil {
		it = &loopGateResponsesItem{outputIndex: index}
		r.items[index] = it
	}
	return it
}

// part returns the part at index of parts, adding it with typ when missing
// (typ "" keeps the type of an existing part).
func (it *loopGateResponsesItem) part(parts *[]*loopGateResponsesPart, index int, typ string) *loopGateResponsesPart {
	for _, p := range *parts {
		if p.index == index {
			if p.typ == "" {
				p.typ = typ
			}
			return p
		}
	}
	p := &loopGateResponsesPart{index: index, typ: typ}
	*parts = append(*parts, p)
	return p
}

func rawString(raw json.RawMessage) string {
	var s string
	if len(raw) > 0 && json.Unmarshal(raw, &s) == nil {
		return s
	}
	return ""
}

// sortedItems returns the items in output_index order.
func (r *loopGateResponsesOutput) sortedItems() []*loopGateResponsesItem {
	items := make([]*loopGateResponsesItem, 0, len(r.items))
	for _, it := range r.items {
		items = append(items, it)
	}
	sort.Slice(items, func(i, j int) bool { return items[i].outputIndex < items[j].outputIndex })
	return items
}

// closeOpen writes to w the events that close every item left open (the
// *.done events of its open parts, then response.output_item.done with status
// "incomplete"), numbered from *seq on, and returns the full output: finished
// items as the upstream sent them, closed ones as rebuilt here.
func (r *loopGateResponsesOutput) closeOpen(w io.Writer, seq *int) []any {
	output := []any{}
	emit := func(ev map[string]interface{}) {
		_ = responses.WriteSSEEvent(w, ev["type"].(string), ev, seq)
	}
	for _, it := range r.sortedItems() {
		if it.done {
			output = append(output, it.item)
			continue
		}
		item := make(map[string]any, len(it.item)+3)
		for k, v := range it.item {
			item[k] = v
		}
		if it.id != "" {
			item["id"] = it.id
		}
		if it.typ != "" {
			item["type"] = it.typ
		}
		item["status"] = "incomplete"

		content := make([]any, 0, len(it.content))
		for _, p := range sortedParts(it.content) {
			part := p.built()
			content = append(content, part)
			if p.done {
				continue
			}
			switch p.typ {
			case "refusal":
				emit(responses.BuildRefusalDoneEvent(it.id, it.outputIndex, p.index, p.text.String()))
			case "reasoning_text":
				emit(responses.BuildReasoningTextDoneEvent(it.id, it.outputIndex, p.index, p.text.String()))
			default:
				ev := responses.BuildOutputTextDoneEvent(it.id, it.outputIndex, p.index, p.text.String())
				ev["logprobs"] = []any{}
				emit(ev)
			}
			if p.part != nil {
				ev := responses.BuildContentPartDoneEvent(it.id, it.outputIndex, p.index, "")
				ev["part"] = part
				emit(ev)
			}
		}
		summary := make([]any, 0, len(it.summary))
		for _, p := range sortedParts(it.summary) {
			part := p.built()
			summary = append(summary, part)
			if p.done {
				continue
			}
			emit(responses.BuildReasoningSummaryTextDoneEvent(it.id, it.outputIndex, p.index, p.text.String()))
			if p.part != nil {
				ev := responses.BuildReasoningSummaryPartDoneEvent(it.id, it.outputIndex, p.index, "")
				ev["part"] = part
				emit(ev)
			}
		}
		switch it.typ {
		case "message":
			item["content"] = content
		case "reasoning":
			item["summary"] = summary
			if len(content) > 0 {
				item["content"] = content
			}
		case "function_call":
			args := it.arguments.String()
			item["arguments"] = args
			emit(responses.BuildFunctionCallArgumentsDoneEvent(it.id, it.outputIndex, args))
		}
		emit(responses.BuildOutputItemDoneEvent(it.outputIndex, item))
		output = append(output, item)
	}
	return output
}

func sortedParts(parts []*loopGateResponsesPart) []*loopGateResponsesPart {
	sort.Slice(parts, func(i, j int) bool { return parts[i].index < parts[j].index })
	return parts
}

// built returns the part with its accumulated text.
func (p *loopGateResponsesPart) built() map[string]any {
	part := make(map[string]any, len(p.part)+2)
	for k, v := range p.part {
		part[k] = v
	}
	typ := p.typ
	if typ == "" {
		typ = "output_text"
	}
	part["type"] = typ
	switch typ {
	case "refusal":
		part["refusal"] = p.text.String()
	case "output_text":
		part["text"] = p.text.String()
		if _, ok := part["annotations"]; !ok {
			part["annotations"] = []any{}
		}
	default:
		part["text"] = p.text.String()
	}
	return part
}
