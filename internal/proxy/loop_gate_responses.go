package proxy

import (
	"bytes"
	"sort"
	"strings"

	json "github.com/goccy/go-json"
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
	Type         string                     `json:"type"`
	OutputIndex  int                        `json:"output_index"`
	ContentIndex int                        `json:"content_index"`
	SummaryIndex int                        `json:"summary_index"`
	ItemID       string                     `json:"item_id"`
	Delta        string                     `json:"delta"`
	Item         map[string]json.RawMessage `json:"item"`
	Part         map[string]json.RawMessage `json:"part"`
	Response     map[string]json.RawMessage `json:"response"`
}

func newLoopGateResponsesOutput() *loopGateResponsesOutput {
	return &loopGateResponsesOutput{items: make(map[int]*loopGateResponsesItem)}
}

// add applies one Responses API event payload.
func (r *loopGateResponsesOutput) add(payload []byte) {
	if !bytes.Contains(payload, sseTypeNeedle) {
		return
	}
	var ev loopGateResponsesEvent
	if json.Unmarshal(payload, &ev) != nil {
		return
	}
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

// closeOpen returns the events that close every item left open (the *.done
// events of its open parts, then response.output_item.done with status
// "incomplete"), and the full output: finished items as the upstream sent
// them, closed ones as rebuilt here. nextSeq numbers the events.
func (r *loopGateResponsesOutput) closeOpen(nextSeq func() int) ([]loopGateSSEEvent, []any) {
	var events []loopGateSSEEvent
	var output []any
	emit := func(name string, fields map[string]any) {
		fields["type"] = name
		fields["sequence_number"] = nextSeq()
		events = append(events, loopGateSSEEvent{name: name, data: fields})
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
			base := func() map[string]any {
				return map[string]any{"item_id": it.id, "output_index": it.outputIndex, "content_index": p.index}
			}
			switch p.typ {
			case "refusal":
				ev := base()
				ev["refusal"] = p.text.String()
				emit("response.refusal.done", ev)
			case "reasoning_text":
				ev := base()
				ev["text"] = p.text.String()
				emit("response.reasoning_text.done", ev)
			default:
				ev := base()
				ev["text"] = p.text.String()
				ev["logprobs"] = []any{}
				emit("response.output_text.done", ev)
			}
			if p.part != nil {
				ev := base()
				ev["part"] = part
				emit("response.content_part.done", ev)
			}
		}
		summary := make([]any, 0, len(it.summary))
		for _, p := range sortedParts(it.summary) {
			part := p.built()
			summary = append(summary, part)
			if p.done {
				continue
			}
			emit("response.reasoning_summary_text.done", map[string]any{
				"item_id": it.id, "output_index": it.outputIndex, "summary_index": p.index, "text": p.text.String(),
			})
			if p.part != nil {
				emit("response.reasoning_summary_part.done", map[string]any{
					"item_id": it.id, "output_index": it.outputIndex, "summary_index": p.index, "part": part,
				})
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
			emit("response.function_call_arguments.done", map[string]any{
				"item_id": it.id, "output_index": it.outputIndex, "arguments": args,
			})
		}
		emit("response.output_item.done", map[string]any{"output_index": it.outputIndex, "item": item})
		output = append(output, item)
	}
	if output == nil {
		output = []any{}
	}
	return events, output
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
