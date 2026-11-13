// Package publicrefusal exposes classifier decisions without forwarding
// provider-controlled prose or private extension fields to the client.
package publicrefusal

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

const Unknown = "The model declined this request; no public reason was provided."

// Explanation is intentionally an allowlist, not a scrubber for arbitrary text.
// A provider can echo credentials, URLs or customer input in explanation or even
// category. Only a recognized category determines the public explanation.
func Explanation(category string) string {
	switch category {
	case "reasoning_extraction":
		return "The safety classifier identified a request to reproduce internal reasoning."
	case "frontier_llm":
		return "The safety classifier identified a request related to developing competing AI models."
	case "cyber":
		return "The safety classifier identified a potential cybersecurity harm."
	case "bio":
		return "The safety classifier identified a potential biological harm."
	case "general_harms":
		return "The safety classifier identified a potential usage-policy violation."
	default:
		return Unknown
	}
}

func Message(category string) string {
	message := Explanation(category)
	if message != Unknown {
		message += " (category: " + category + ")"
	}
	return message
}

// CategoryFromMessage restores the public category across AIR's Chat adapter.
// Unknown text and unknown categories never become native refusal details.
func CategoryFromMessage(message string) string {
	for _, category := range []string{"reasoning_extraction", "frontier_llm", "cyber", "bio", "general_harms"} {
		if message == Message(category) {
			return category
		}
	}
	return ""
}

func Details(category string) map[string]any {
	var publicCategory any
	if Explanation(category) != Unknown {
		publicCategory = category
	}
	return map[string]any{"type": "refusal", "category": publicCategory, "explanation": Explanation(category)}
}

// SanitizeEnvelope only visits protocol envelopes, never model text, tool
// arguments, thinking blocks or arbitrary user-defined objects.
func SanitizeEnvelope(obj map[string]any) bool {
	changed := false
	if reason, _ := obj["stop_reason"].(string); reason == "refusal" {
		details, _ := obj["stop_details"].(map[string]any)
		category, _ := details["category"].(string)
		public := Details(category)
		if len(details) != len(public) || details["type"] != public["type"] ||
			details["category"] != public["category"] || details["explanation"] != public["explanation"] {
			obj["stop_details"] = public
			changed = true
		}
	} else if _, present := obj["stop_details"]; present && obj["stop_details"] != nil {
		// Details are defined only for refusal; never expose misplaced extensions.
		obj["stop_details"] = nil
		changed = true
	}
	for _, key := range []string{"message", "delta"} {
		if child, ok := obj[key].(map[string]any); ok && SanitizeEnvelope(child) {
			changed = true
		}
	}
	return changed
}

func SanitizeJSON(body []byte) ([]byte, bool, error) {
	var obj map[string]any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&obj); err != nil {
		return nil, false, err
	}
	// Reject a valid object followed by garbage; never pass the suffix through.
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, false, errors.New("invalid refusal response JSON")
	}
	if !SanitizeEnvelope(obj) {
		return body, false, nil
	}
	clean, err := json.Marshal(obj)
	return clean, true, err
}
