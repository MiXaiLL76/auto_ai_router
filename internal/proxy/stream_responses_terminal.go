package proxy

import (
	"bytes"
	"encoding/json"
	"math"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// newResponsesStreamLines returns the SSE line reassembly of the passthrough
// Responses handlers. Unlike other usage lines it has no length bound: the
// terminal event carries the usage and is read whole, however large.
// Reassembly appends each read and searches only it for the line end, so a
// multi-MB line costs linear time (string concatenation used to recopy and
// rescan the whole line on every 8 KB read).
func newResponsesStreamLines() streamUsageLines {
	return streamUsageLines{maxLineBytes: math.MaxInt}
}

// responsesStreamDataPayload returns the JSON of an SSE data: line, with or
// without the (spec-optional) space after the colon and the line ending, or
// nil for any other line, empty data and [DONE].
func responsesStreamDataPayload(line []byte) []byte {
	if !bytes.HasPrefix(line, sseDataPrefix) {
		return nil
	}
	payload := bytes.TrimSpace(line[len(sseDataPrefix):])
	if len(payload) == 0 || bytes.Equal(payload, sseDoneSentinel) {
		return nil
	}
	return payload
}

// The terminal events of a Responses stream, by their quoted type value. A
// payload holding neither cannot be one of them (type values are plain ASCII,
// which JSON encoders do not escape), so the passthrough handlers skip the
// JSON decode of every other event, the bulk of a stream.
var (
	responsesCompletedTypeToken  = []byte(`"response.completed"`)
	responsesIncompleteTypeToken = []byte(`"response.incomplete"`)
)

func isTerminalResponsesEventType(eventType string) bool {
	return eventType == "response.completed" || eventType == "response.incomplete"
}

// mayBeTerminalResponsesEvent reports whether payload, a stream event's JSON,
// can be a terminal Responses event.
func mayBeTerminalResponsesEvent(payload []byte) bool {
	return bytes.Contains(payload, responsesCompletedTypeToken) || bytes.Contains(payload, responsesIncompleteTypeToken)
}

// responsesTerminalEvent is a terminal Responses stream event
// (response.completed or response.incomplete) decoded once for everything the
// passthrough handlers read from it. The usage objects are kept raw: each
// reader decodes the small object into its own shape, while the large nested
// output (reasoning, generated images) is walked once, for the item types.
type responsesTerminalEvent struct {
	Type  string          `json:"type"`
	Error json.RawMessage `json:"error"`
	// Top-level usage and output, which the stream usage extractor also reads.
	Usage    json.RawMessage `json:"usage"`
	Output   json.RawMessage `json:"output"`
	Response struct {
		Usage  json.RawMessage           `json:"usage"`
		Output []streamingResponseOutput `json:"output"`
	} `json:"response"`
}

// parseResponsesTerminalEvent decodes payload, a stream event's JSON, when it
// is a terminal Responses event.
func parseResponsesTerminalEvent(payload []byte) (*responsesTerminalEvent, bool) {
	if !mayBeTerminalResponsesEvent(payload) {
		return nil, false
	}
	var event responsesTerminalEvent
	if json.Unmarshal(payload, &event) != nil || !isTerminalResponsesEventType(event.Type) {
		return nil, false
	}
	return &event, true
}

// typedUsage returns the response's usage, or nil when it has none or the
// usage does not decode.
func (e *responsesTerminalEvent) typedUsage() *responses.Usage {
	var usage *responses.Usage
	if len(e.Response.Usage) == 0 || json.Unmarshal(e.Response.Usage, &usage) != nil {
		return nil
	}
	return usage
}

// finalChunkParse returns what finalizeStreamingLog reads from payload, the
// event's JSON: what the OpenAI stream usage extractor and
// extractStreamErrorEvent return for it.
func (e *responsesTerminalEvent) finalChunkParse(payload []byte, audioInputAlreadyExcludesCachedAudio bool) *finalChunkParse {
	return &finalChunkParse{
		usage:      e.streamUsage(&openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: audioInputAlreadyExcludesCachedAudio}),
		errorEvent: e.streamErrorEvent(payload),
	}
}

// streamUsage is extractor.ExtractUsage for the event: a top-level Chat
// Completions usage wins over the Responses one. As there, a member that does
// not decode into the shape it is read in voids that shape's result.
func (e *responsesTerminalEvent) streamUsage(extractor *openAIStreamUsageExtractor) *StreamUsageInfo {
	var chatUsage *chatCompletionStreamUsage
	if decodeOptionalJSON(e.Usage, &chatUsage) {
		if info := extractor.chatCompletionUsageInfo(chatUsage); info != nil {
			return info
		}
	}
	var topUsage, nestedUsage *responsesAPIUsage
	var topOutput []streamingResponseOutput
	if !decodeOptionalJSON(e.Usage, &topUsage) || !decodeOptionalJSON(e.Output, &topOutput) ||
		!decodeOptionalJSON(e.Response.Usage, &nestedUsage) {
		return nil
	}
	return extractor.responsesAPIUsageInfo(topUsage, topOutput, nestedUsage, e.Response.Output)
}

// streamErrorEvent is extractStreamErrorEvent for the event: its type is not
// an error type, so only a non-null top-level "error" makes it an error event.
func (e *responsesTerminalEvent) streamErrorEvent(payload []byte) string {
	if isStreamErrorEvent(e.Type, len(e.Error) > 0 && string(e.Error) != "null") {
		return string(payload)
	}
	return ""
}

// decodeOptionalJSON decodes raw into v, leaving v unset when raw is absent,
// and reports whether raw decoded.
func decodeOptionalJSON(raw json.RawMessage, v any) bool {
	return len(raw) == 0 || json.Unmarshal(raw, v) == nil
}

// terminalResponseJSON returns the response object of payload, a terminal
// event's JSON, as the upstream sent it, or nil when it has none. Unlike
// decodeTerminalResponse it does not need to fit responses.Response.
func terminalResponseJSON(payload []byte) []byte {
	var event struct {
		Response json.RawMessage `json:"response"`
	}
	if json.Unmarshal(payload, &event) != nil {
		return nil
	}
	if trimmed := bytes.TrimSpace(event.Response); len(trimmed) == 0 || trimmed[0] != '{' {
		return nil
	}
	return event.Response
}

// decodeTerminalResponse decodes the response object of payload, a terminal
// event's JSON. Only a stored response needs it whole.
func decodeTerminalResponse(payload []byte) (*responses.Response, bool) {
	var event struct {
		Response responses.Response `json:"response"`
	}
	if json.Unmarshal(payload, &event) != nil {
		return nil, false
	}
	return &event.Response, true
}
