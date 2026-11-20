package anthropicresponses

import (
	"io"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

func anthropicRefusalItem(reason string, details *anthropic.StopDetails) *responses.OutputItem {
	refusal := anthropic.RefusalMessage(reason, details)
	if refusal == "" {
		return nil
	}
	return &responses.OutputItem{
		Type: "message", ID: responses.GenerateItemID("msg_"), Status: "completed", Role: "assistant",
		Content: []responses.OutputContent{{Type: "refusal", Refusal: refusal}},
	}
}

// A classifier can stop a response after text or tools have already streamed.
// Announce a separate refusal item after those items, retaining their indices
// and making the refusal visible to incremental clients and final snapshots.
func emitAnthropicRefusal(w io.Writer, acc *anthropicStreamAccumulator, outputIndex int) error {
	item := acc.refusalItem
	if item == nil {
		return nil
	}
	part := item.Content[0]
	events := []map[string]interface{}{
		responses.BuildMessageItemAddedEvent(outputIndex, item.ID),
		{"type": "response.content_part.added", "item_id": item.ID, "output_index": outputIndex,
			"content_index": 0, "part": map[string]interface{}{"type": "refusal", "refusal": ""}},
		{"type": "response.refusal.delta", "item_id": item.ID, "output_index": outputIndex,
			"content_index": 0, "delta": part.Refusal},
		responses.BuildRefusalDoneEvent(item.ID, outputIndex, 0, part.Refusal),
		{"type": "response.content_part.done", "item_id": item.ID, "output_index": outputIndex,
			"content_index": 0, "part": part},
		{"type": "response.output_item.done", "output_index": outputIndex, "item": item},
	}
	for _, event := range events {
		if err := writeAnthropicSSE(w, event["type"].(string), event, acc); err != nil {
			return err
		}
	}
	return nil
}
