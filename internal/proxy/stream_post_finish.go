package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime"
	"net/http"
	"slices"
)

// chatPostFinishFilter holds an incomplete SSE event and tracks which choices
// have finished in this response.
type chatPostFinishFilter struct {
	pending []byte
	scan    int
	line    int
	choices map[int]chatPostFinishChoice
	report  func(int, string, []string)
}

type chatPostFinishChoice struct {
	reason   string
	reported bool
}

func (p *Proxy) newChatPostFinishFilter(ctx context.Context, headers http.Header, endpoint string, status int, model string, logCtx *RequestLogContext) *chatPostFinishFilter {
	if endpoint != "/v1/chat/completions" || status < http.StatusOK || status >= http.StatusMultipleChoices {
		return nil
	}
	mediaType, _, err := mime.ParseMediaType(headers.Get("Content-Type"))
	if err != nil || mediaType != "text/event-stream" {
		return nil
	}
	headers.Del("Content-Length")
	dropRepresentationIntegrityHeaders(headers)
	return &chatPostFinishFilter{
		choices: make(map[int]chatPostFinishChoice),
		report: func(index int, reason string, fields []string) {
			p.logger.WarnContext(ctx, "Dropped choice after finish_reason in Chat Completions stream",
				"request_id", logContextRequestID(logCtx), "model", clientVisibleResponseModel(logCtx, model),
				"choice_index", index, "finish_reason", reason, "late_fields", fields)
		},
	}
}

func (f *chatPostFinishFilter) filter(chunk []byte, final bool) []byte {
	if f == nil {
		return chunk
	}
	f.pending = append(f.pending, chunk...)
	var output []byte
	eventStart := 0
	for f.scan < len(f.pending) {
		next := bytes.IndexByte(f.pending[f.scan:], '\n')
		if next < 0 {
			f.scan = len(f.pending)
			break
		}
		end := f.scan + next
		if len(bytes.TrimSuffix(f.pending[f.line:end], []byte("\r"))) == 0 {
			output = append(output, f.filterEvent(f.pending[eventStart:end+1])...)
			eventStart = end + 1
		}
		f.scan = end + 1
		f.line = f.scan
	}
	if final {
		output = append(output, f.filterEvent(f.pending[eventStart:])...)
		f.pending = nil
		return output
	}
	if eventStart > 0 {
		f.pending = f.pending[:copy(f.pending, f.pending[eventStart:])]
		f.scan -= eventStart
		f.line -= eventStart
	}
	return output
}

func (f *chatPostFinishFilter) filterEvent(event []byte) []byte {
	var payload []byte
	hasData := false
	for line := range bytes.SplitSeq(event, []byte("\n")) {
		if data, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			data = bytes.TrimPrefix(bytes.TrimSuffix(data, []byte("\r")), []byte(" "))
			if hasData {
				payload = slices.Concat(payload, []byte("\n"), data)
			} else {
				payload = data
				hasData = true
			}
		}
	}
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 || payload[0] != '{' {
		return event
	}

	var body map[string]json.RawMessage
	var choices []json.RawMessage
	if json.Unmarshal(payload, &body) != nil || json.Unmarshal(body["choices"], &choices) != nil {
		return event
	}
	kept := choices[:0]
	// Commit finishes after the event so the records in the terminal event
	// itself are preserved.
	var finished []int
	for _, raw := range choices {
		var choice map[string]json.RawMessage
		var index *int
		if json.Unmarshal(raw, &choice) != nil || json.Unmarshal(choice["index"], &index) != nil || index == nil {
			kept = append(kept, raw)
			continue
		}
		choiceIndex := *index
		state := f.choices[choiceIndex]
		if state.reason != "" && !slices.Contains(finished, choiceIndex) {
			if !state.reported {
				f.report(choiceIndex, state.reason, lateChoiceFields(choice))
				state.reported = true
				f.choices[choiceIndex] = state
			}
			continue
		}
		if state.reason == "" {
			var reason string
			if json.Unmarshal(choice["finish_reason"], &reason) == nil && reason != "" {
				f.choices[choiceIndex] = chatPostFinishChoice{reason: reason}
				finished = append(finished, choiceIndex)
			}
		}
		kept = append(kept, raw)
	}
	usage := body["usage"]
	hasError := hasSSEValue(body["error"])
	splitUsage := len(finished) > 0 && hasSSEValue(usage) && !hasError
	if len(kept) == len(choices) && !splitUsage {
		return event
	}
	if len(kept) == 0 && !hasSSEValue(usage) && !hasError {
		return nil
	}

	body["choices"], _ = json.Marshal(kept)
	if splitUsage {
		delete(body, "usage")
	}
	filtered, _ := json.Marshal(body)
	// Replace the data fields while retaining SSE metadata and framing.
	var output []byte
	for line := range bytes.SplitAfterSeq(event, []byte("\n")) {
		if !bytes.HasPrefix(line, []byte("data:")) {
			output = append(output, line...)
		} else if filtered != nil {
			output = append(output, "data: "...)
			output = append(output, filtered...)
			output = append(output, line[len(bytes.TrimRight(line, "\r\n")):]...)
			filtered = nil
		}
	}
	if splitUsage {
		usageChunk := map[string]json.RawMessage{"choices": json.RawMessage("[]"), "usage": usage}
		for _, field := range []string{"id", "object", "created", "model", "system_fingerprint", "service_tier"} {
			if value, ok := body[field]; ok {
				usageChunk[field] = value
			}
		}
		stats, _ := json.Marshal(usageChunk)
		if !bytes.HasSuffix(output, []byte("\n\n")) && !bytes.HasSuffix(output, []byte("\n\r\n")) {
			// An EOF tail may have no event delimiter.
			ending := []byte("\n\n")
			if bytes.HasSuffix(output, []byte("\n")) {
				ending = ending[:1]
			}
			output = append(output, ending...)
		}
		output = append(output, "data: "...)
		output = append(output, stats...)
		output = append(output, "\n\n"...)
	}
	return output
}

func hasSSEValue(value json.RawMessage) bool {
	return len(value) != 0 && !bytes.Equal(value, []byte("null"))
}

func lateChoiceFields(choice map[string]json.RawMessage) []string {
	var delta map[string]json.RawMessage
	_ = json.Unmarshal(choice["delta"], &delta)
	fields := make([]string, 0, len(choice)+len(delta))
	for field := range choice {
		switch field {
		case "index":
		case "delta":
			for name := range delta {
				fields = append(fields, name)
			}
		default:
			fields = append(fields, field)
		}
	}
	slices.Sort(fields)
	return fields
}

type chatPostFinishWriter struct {
	dst    io.Writer
	filter *chatPostFinishFilter
}

func (w chatPostFinishWriter) Write(chunk []byte) (int, error) {
	if data := w.filter.filter(chunk, false); len(data) > 0 {
		if _, err := w.dst.Write(data); err != nil {
			return 0, err
		}
	}
	return len(chunk), nil
}
