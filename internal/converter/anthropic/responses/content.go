package anthropicresponses

import (
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// redactedThinkingPrefix marks a reasoning item's encrypted_content that carries a
// redacted_thinking block's data rather than a thinking block's signature.
const redactedThinkingPrefix = "redacted_thinking:"

// thinkingEncryptedContent is the encrypted_content of the reasoning item that stands
// for an Anthropic thinking or redacted_thinking block; reasoningItemToThinking
// turns it back into the same block.
func thinkingEncryptedContent(block anthropic.ContentBlock) string {
	if block.Type == "redacted_thinking" {
		return redactedThinkingPrefix + block.Data
	}
	return block.Signature
}

// reasoningOutputItem builds the reasoning item for a thinking block. A block with
// empty text (display "omitted") still carries its signature and must be kept.
func reasoningOutputItem(id, thinking, encryptedContent string) responses.OutputItem {
	item := responses.OutputItem{Type: "reasoning", ID: id, Status: "completed", EncryptedContent: encryptedContent}
	if thinking != "" {
		item.Summary = []responses.OutputContent{{Type: "summary_text", Text: thinking}}
	}
	return item
}

// reasoningItemToThinking turns a reasoning input item back into its Anthropic block,
// or returns nil when the item carries neither text nor encrypted_content.
func reasoningItemToThinking(item map[string]interface{}) *anthropic.ContentBlock {
	encrypted, _ := item["encrypted_content"].(string)
	if data, ok := strings.CutPrefix(encrypted, redactedThinkingPrefix); ok {
		return &anthropic.ContentBlock{Type: "redacted_thinking", Data: data}
	}
	// The signature is verified against the thinking text it was issued for, so the
	// text travels with it: reasoningOutputItem put the full thinking into the summary.
	text := responses.ReasoningItemText(item)
	if text == "" && encrypted == "" {
		return nil
	}
	return &anthropic.ContentBlock{Type: "thinking", Thinking: text, Signature: encrypted}
}

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
		if fileID, _ := partMap["file_id"].(string); fileID != "" {
			return nil, converterutil.NewRequestValidationError("input_image.file_id", "file_id is not supported for this route")
		}
		return nil, converterutil.NewRequestValidationError("input_image", "missing supported image source")
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
		return nil, converterutil.NewRequestValidationError("input_file.file_id", "file_id is not supported for this route")
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
