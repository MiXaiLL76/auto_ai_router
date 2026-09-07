package openai

import "encoding/json"

func IsGptImage1MiniModel(modelID string) bool {
	return matchModelFamily(modelID, "gpt-image-1-mini")
}

func normalizeImageMiniQuality(quality string) string {
	switch quality {
	case "standard":
		return "medium"
	case "hd":
		return "high"
	default:
		return quality
	}
}

func RewriteImageMiniJSON(body []byte, modelID string, edit bool) []byte {
	if !IsGptImage1MiniModel(modelID) {
		return body
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || fields == nil {
		return body
	}
	changed := false
	var quality string
	if json.Unmarshal(fields["quality"], &quality) == nil {
		if normalized := normalizeImageMiniQuality(quality); normalized != quality {
			fields["quality"], _ = json.Marshal(normalized)
			changed = true
		}
	}
	if _, exists := fields["input_fidelity"]; edit && exists {
		delete(fields, "input_fidelity")
		changed = true
	}
	if !changed {
		return body
	}
	result, err := json.Marshal(fields)
	if err != nil {
		return body
	}
	return result
}
