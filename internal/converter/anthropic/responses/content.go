package anthropicresponses

import (
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// contentPartToAnthropic converts a Responses API content part map to an Anthropic ContentBlock.
// Returns nil when the part type is unknown and should be silently skipped.
func contentPartToAnthropic(partMap map[string]interface{}) (*anthropic.ContentBlock, error) {
	partType, _ := partMap["type"].(string)
	switch partType {
	case "input_text", "output_text", "text":
		text, _ := partMap["text"].(string)
		return &anthropic.ContentBlock{Type: "text", Text: text}, nil

	case "input_image":
		return convertInputImageToAnthropic(partMap)

	case "input_audio":
		// Anthropic does not support audio input natively.
		return nil, converterutil.NewRequestValidationError("input_audio", "input_audio is not supported by Anthropic")

	case "input_file":
		return convertInputFileToAnthropic(partMap)

	case "reasoning_text", "summary_text":
		text, _ := partMap["text"].(string)
		return &anthropic.ContentBlock{Type: "thinking", Thinking: text}, nil

	default:
		return nil, nil
	}
}

func convertInputImageToAnthropic(partMap map[string]interface{}) (*anthropic.ContentBlock, error) {
	var imgURL string
	switch v := partMap["image_url"].(type) {
	case string:
		imgURL = v
	case map[string]interface{}:
		imgURL, _ = v["url"].(string)
	}

	if imgURL == "" {
		return nil, fmt.Errorf("input_image: missing image_url")
	}

	// data: URL → base64 inline
	if strings.HasPrefix(imgURL, "data:") {
		mimeType, data, err := parseDataURL(imgURL)
		if err != nil {
			return nil, err
		}
		if mimeType == "application/pdf" {
			return nil, converterutil.NewRequestValidationError("input_image.image_url", "PDF data URLs must be sent as input_file, not input_image")
		}
		if !isAllowedImageMediaType(mimeType) {
			return nil, converterutil.NewRequestValidationError("input_image.image_url", fmt.Sprintf("unsupported image media type %q for Anthropic", mimeType))
		}
		return &anthropic.ContentBlock{
			Type: "image",
			Source: &anthropic.MediaSource{
				Type:      "base64",
				MediaType: mimeType,
				Data:      data,
			},
		}, nil
	}

	// Regular URL → download and base64-encode; Anthropic does not support url sources.
	block := anthropic.DownloadAndEncodeImage(imgURL)
	if block == nil {
		return nil, fmt.Errorf("input_image: failed to download image from %s", imgURL)
	}
	return block, nil
}

func convertInputFileToAnthropic(partMap map[string]interface{}) (*anthropic.ContentBlock, error) {
	fileData, _ := partMap["file_data"].(string)
	if fileData != "" {
		return anthropic.ConvertDataURLToDocument(fileData, "input_file.file_data")
	}

	fileURL, _ := partMap["file_url"].(string)
	if fileURL != "" {
		return &anthropic.ContentBlock{
			Type: "document",
			Source: &anthropic.MediaSource{
				Type: "url",
				URL:  fileURL,
			},
		}, nil
	}

	if fileID, _ := partMap["file_id"].(string); fileID != "" {
		return nil, converterutil.NewRequestValidationError("input_file.file_id", "input_file.file_id is not supported for Anthropic-backed models")
	}

	return nil, converterutil.NewRequestValidationError("input_file", "missing supported file source")
}

// parseDataURL parses a data: URL into mimeType and base64-encoded data string.
func parseDataURL(dataURL string) (mimeType, b64data string, err error) {
	return anthropic.ParseBase64DataURL(dataURL, "input_image.image_url")
}

func isAllowedImageMediaType(mediaType string) bool {
	switch strings.ToLower(mediaType) {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}
