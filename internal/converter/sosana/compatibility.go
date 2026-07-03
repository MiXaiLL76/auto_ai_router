package sosana

import (
	"bytes"
	"encoding/json"
	"fmt"
	"mime"
	"mime/multipart"
	"strings"
)

const maxInputImages = 14

var unsupportedImageFields = []string{
	"tools",
	"tool_choice",
	"google_search",
	"thinking_level",
	"thinking_budget",
	"thinking_config",
	"thinking",
	"reasoning_effort",
	"generation_config",
	"temperature",
	"top_p",
	"top_k",
	"seed",
	"max_tokens",
	"stop",
	"stream",
	"messages",
	"extra_body",
	"image_size",
	"image",
	"images",
	"image_urls",
	"reference_images",
}

// UnsupportedRequest returns a short reason when a request needs image features
// that Sosana Banana does not expose in its public API.
func UnsupportedRequest(path string, body []byte, contentType string) string {
	switch {
	case strings.Contains(path, "/images/generations"):
		return unsupportedGenerationRequest(body)
	case strings.Contains(path, "/images/edits"):
		return unsupportedEditRequest(body, contentType)
	default:
		return "sosana provider supports only image generation"
	}
}

func unsupportedGenerationRequest(body []byte) string {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return ""
	}
	return unsupportedImageFieldsInJSON(raw)
}

func unsupportedEditRequest(body []byte, contentType string) string {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(mediaType, "multipart/form-data") {
		return ""
	}
	boundary := params["boundary"]
	if boundary == "" {
		return ""
	}

	fields := make(map[string]string)
	imageCount := 0
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, err := reader.NextPart()
		if err != nil {
			break
		}

		formName := part.FormName()
		if formName == "" {
			continue
		}
		data, err := readLimited(part, maxMultipartImageBytes)
		if err != nil {
			return err.Error()
		}
		if part.FileName() == "" {
			fields[formName] = strings.TrimSpace(string(data))
			continue
		}
		if formName == "mask" {
			return "mask is unsupported"
		}
		if formName != "image" && formName != "images" && formName != "image[]" {
			continue
		}
		imageCount++
		if detectImageMIMEType(part.Header.Get("Content-Type"), data) != "image/png" {
			return "only PNG input images are supported"
		}
	}
	if imageCount > maxInputImages {
		return "too many input images"
	}
	return unsupportedImageFieldsInForm(fields)
}

func unsupportedImageFieldsInJSON(raw map[string]json.RawMessage) string {
	if reason := unsupportedJSONImageCount(raw["n"]); reason != "" {
		return reason
	}
	if reason := unsupportedJSONResponseFormat(raw["response_format"]); reason != "" {
		return reason
	}
	if reason := unsupportedJSONOutputFormat(raw["output_format"]); reason != "" {
		return reason
	}
	for _, field := range []string{"quality", "style", "background", "moderation"} {
		if hasJSONValue(raw[field]) {
			return field + " is unsupported"
		}
	}
	if hasJSONValue(raw["output_compression"]) {
		return "output_compression is unsupported"
	}
	for _, field := range unsupportedImageFields {
		if hasJSONValue(raw[field]) {
			return field + " is unsupported"
		}
	}
	return ""
}

func unsupportedImageFieldsInForm(fields map[string]string) string {
	if err := validateImageCountString(fields["n"]); err != nil {
		return err.Error()
	}
	if err := validateResponseFormat(fields["response_format"]); err != nil {
		return err.Error()
	}
	if err := validateOutputFormat(fields["output_format"]); err != nil {
		return err.Error()
	}
	for _, field := range []string{"quality", "style", "background", "moderation"} {
		if strings.TrimSpace(fields[field]) != "" {
			return field + " is unsupported"
		}
	}
	if _, ok := fields["output_compression"]; ok {
		return "output_compression is unsupported"
	}
	for _, field := range unsupportedImageFields {
		if strings.TrimSpace(fields[field]) != "" {
			return field + " is unsupported"
		}
	}
	return ""
}

func unsupportedJSONImageCount(raw json.RawMessage) string {
	if !hasJSONValue(raw) {
		return ""
	}
	var n int
	if err := json.Unmarshal(raw, &n); err == nil {
		if n == 1 {
			return ""
		}
		return "sosana supports n=1 only"
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		if f == 1 {
			return ""
		}
		return "sosana supports n=1 only"
	}
	return "invalid image count"
}

func unsupportedJSONResponseFormat(raw json.RawMessage) string {
	if !hasJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "response_format is unsupported"
	}
	if strings.EqualFold(strings.TrimSpace(value), "b64_json") || strings.TrimSpace(value) == "" {
		return ""
	}
	if strings.EqualFold(strings.TrimSpace(value), "url") {
		return "response_format=url is unsupported for this image model"
	}
	return "response_format is unsupported"
}

func unsupportedJSONOutputFormat(raw json.RawMessage) string {
	if !hasJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "output_format is unsupported"
	}
	if outputFormatAllowed(value) {
		return ""
	}
	return "output_format is unsupported"
}

func validateOutputFormat(raw string) error {
	if outputFormatAllowed(raw) {
		return nil
	}
	return fmt.Errorf("output_format is unsupported")
}

func outputFormatAllowed(raw string) bool {
	value := strings.ToLower(strings.TrimSpace(raw))
	return value == "" || value == "png"
}

func hasJSONValue(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && !bytes.Equal(raw, []byte("null"))
}
