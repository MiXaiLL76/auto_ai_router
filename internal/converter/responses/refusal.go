package responses

import "io"

func emitRefusalStartEvents(w io.Writer, acc *streamAccumulator) error {
	acc.refusalItemID = GenerateItemID("msg_")
	acc.refusalOutputIndex = acc.allocOutputIndex()
	if err := writeSSEWithSeq(w, "response.output_item.added",
		BuildMessageItemAddedEvent(acc.refusalOutputIndex, acc.refusalItemID), acc); err != nil {
		return err
	}
	return writeSSEWithSeq(w, "response.content_part.added", map[string]interface{}{
		"type": "response.content_part.added", "item_id": acc.refusalItemID,
		"output_index": acc.refusalOutputIndex, "content_index": 0,
		"part": map[string]interface{}{"type": "refusal", "refusal": ""},
	}, acc)
}

func completedRefusalItem(acc *streamAccumulator) OutputItem {
	return OutputItem{
		Type: "message", ID: acc.refusalItemID, Status: "completed", Role: "assistant",
		Content: []OutputContent{{Type: "refusal", Refusal: acc.fullRefusal}},
	}
}
