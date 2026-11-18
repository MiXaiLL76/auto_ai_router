package responses

import (
	"encoding/json"
	"fmt"
	"io"
)

// BuildInProgressResponse returns the map[string]interface{} payload used in
// response.created and response.in_progress SSE events.
//
// The caller is responsible for adding any provider-specific fields (e.g.
// "store", "previous_response_id") to the returned map after this call.
func BuildInProgressResponse(responseID, model string, createdAt int64) map[string]interface{} {
	return ResponseToMap(NewResponse(ResponseParams{
		ID:        responseID,
		Model:     model,
		CreatedAt: createdAt,
		Status:    "in_progress",
	}))
}

// CompletedResponseParams holds the pre-computed, provider-independent fields
// needed to assemble a final *Response at stream completion.  The caller is
// responsible for all provider-specific mapping (stop-reason → Status/
// IncompleteDetails, raw chunks → Output, raw usage → *Usage) before filling
// this struct.
type CompletedResponseParams struct {
	ID        string
	Model     string
	CreatedAt int64

	// Pre-computed status fields.
	Status            string
	IncompleteDetails *IncompleteDetails

	// Pre-built output items.
	Output []OutputItem

	// Pre-computed token usage (may be nil).
	Usage *Usage

	// Metadata and conversation linkage.
	Metadata           map[string]string
	PreviousResponseID interface{} // nil or string

	// CompletedAt is set by providers that track a completion timestamp.
	// Leave nil to omit (e.g. Chat Completions path).
	CompletedAt *int64

	// Store and ToolChoice are only populated by the Chat Completions path.
	Store      bool
	ToolChoice interface{}

	// Text is the request's text config to echo (ResponsesMetadata.Text).
	Text interface{}
}

// BuildCompletedResponse constructs the typed *Response for a stream-completed
// event from pre-computed, provider-neutral fields.  All provider-specific
// conversions (stop-reasons, usage shapes, output item types) must be done by
// the caller before invoking this function.
func BuildCompletedResponse(p CompletedResponseParams) *Response {
	return NewResponse(ResponseParams{
		ID:                 p.ID,
		Model:              p.Model,
		CreatedAt:          p.CreatedAt,
		Status:             p.Status,
		CompletedAt:        p.CompletedAt,
		IncompleteDetails:  p.IncompleteDetails,
		Output:             p.Output,
		Usage:              p.Usage,
		Metadata:           p.Metadata,
		PreviousResponseID: p.PreviousResponseID,
		Store:              p.Store,
		ToolChoice:         p.ToolChoice,
		Text:               p.Text,
	})
}

// WriteSSEEvent JSON-marshals data, increments *seqNum, and writes a single
// SSE frame in the format:
//
//	event: <eventType>\ndata: <json>\n\n
//
// seqNum must not be nil. The caller owns the sequenceNumber field inside its
// own accumulator and passes a pointer so the counter stays in sync.
func WriteSSEEvent(w io.Writer, eventType string, data interface{}, seqNum *int) error {
	payload, err := injectSequenceNumber(data, *seqNum+1)
	if err != nil {
		return fmt.Errorf("WriteSSEEvent marshal %s: %w", eventType, err)
	}
	*seqNum++
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, payload)
	return err
}

func injectSequenceNumber(data interface{}, seq int) ([]byte, error) {
	switch v := data.(type) {
	case map[string]interface{}:
		if _, exists := v["sequence_number"]; !exists {
			v["sequence_number"] = seq
		}
		return json.Marshal(v)
	default:
		raw, err := json.Marshal(data)
		if err != nil {
			return nil, err
		}
		var obj map[string]interface{}
		if err := json.Unmarshal(raw, &obj); err != nil {
			return nil, err
		}
		if _, exists := obj["sequence_number"]; !exists {
			obj["sequence_number"] = seq
		}
		return json.Marshal(obj)
	}
}
