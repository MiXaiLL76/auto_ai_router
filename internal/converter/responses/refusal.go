package responses

import (
	"io"
	"strings"
)

func refusalInputText(item map[string]interface{}) string {
	parts, _ := item["content"].([]interface{})
	var text []string
	for _, raw := range parts {
		part, ok := raw.(map[string]interface{})
		if !ok || (part["type"] != "refusal" && part["type"] != "output_refusal") {
			return ""
		}
		value, _ := part["refusal"].(string)
		text = append(text, value)
	}
	return strings.Join(text, "\n\n")
}

func appendAssistantText(message map[string]interface{}, text string) {
	switch content := message["content"].(type) {
	case []interface{}:
		message["content"] = append(content, map[string]interface{}{"type": "text", "text": text})
	case string:
		message["content"] = content + "\n\n" + text
	default:
		message["content"] = text
	}
}

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
