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

// watermarkingImageFamilies are image model families that stamp a visible
// watermark on generated images unless the request explicitly opts out.
var watermarkingImageFamilies = []string{"seedream", "dola-seedream", "seededit"}

// AddsImageWatermark reports whether modelID belongs to an image family that
// watermarks its output by default.
func AddsImageWatermark(modelID string) bool {
	for _, family := range watermarkingImageFamilies {
		if matchModelFamily(modelID, family) {
			return true
		}
	}
	return false
}

// DisableImageWatermark forces "watermark": false on a JSON image request,
// overriding any client-supplied value. Other fields are forwarded byte-for-byte;
// bodies that are not a JSON object are returned unchanged — multipart edits get
// the same opt-out from RewriteImageEditMultipart.
func DisableImageWatermark(body []byte) []byte {
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || fields == nil {
		return body
	}
	if current, exists := fields["watermark"]; exists && string(current) == "false" {
		return body
	}
	fields["watermark"] = json.RawMessage("false")
	result, err := json.Marshal(fields)
	if err != nil {
		return body
	}
	return result
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
