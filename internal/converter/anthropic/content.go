package anthropic

import "strings"

// convertOpenAIContentToAnthropic converts an OpenAI message content value (string or
// []interface{} of content blocks) into a slice of Anthropic ContentBlocks.
func convertOpenAIContentToAnthropic(content interface{}) []ContentBlock {
	switch c := content.(type) {
	case string:
		if c == "" {
			return nil
		}
		return []ContentBlock{{Type: "text", Text: c}}

	case []interface{}:
		var blocks []ContentBlock
		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			blockType, _ := blockMap["type"].(string)
			switch blockType {
			case "text":
				text, _ := blockMap["text"].(string)
				if text != "" {
					blocks = append(blocks, ContentBlock{Type: "text", Text: text})
				}

			case "image_url":
				imageURL, ok := blockMap["image_url"].(map[string]interface{})
				if !ok {
					continue
				}
				url, _ := imageURL["url"].(string)
				if url == "" {
					continue
				}
				if cb := convertImageURLToAnthropic(url); cb != nil {
					blocks = append(blocks, *cb)
				}

			case "input_audio":
				// Anthropic does not support audio input; emit a text placeholder.
				audioData, ok := blockMap["input_audio"].(map[string]interface{})
				if !ok {
					continue
				}
				format, _ := audioData["format"].(string)
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: "[Audio input: " + format + " format - not supported by Anthropic API]",
				})

			case "video_url":
				videoURL, ok := blockMap["video_url"].(map[string]interface{})
				if !ok {
					continue
				}
				url, _ := videoURL["url"].(string)
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: "[Video: " + url + "]",
				})

			case "file":
				fileObj, ok := blockMap["file"].(map[string]interface{})
				if !ok {
					continue
				}
				fileID, _ := fileObj["file_id"].(string)
				if fileID == "" {
					continue
				}
				// Data-URL files can be forwarded as Anthropic document blocks.
				if strings.HasPrefix(fileID, "data:") {
					if cb := convertDataURLToDocument(fileID); cb != nil {
						blocks = append(blocks, *cb)
					}
				}
			}
		}
		return blocks
	}
	return nil
}

// convertImageURLToAnthropic converts an OpenAI image_url value to an Anthropic image block.
// Supports base64 data URLs and plain HTTPS/HTTP URLs.
func convertImageURLToAnthropic(url string) *ContentBlock {
	if strings.HasPrefix(url, "data:") {
		parts := strings.SplitN(url, ",", 2)
		if len(parts) != 2 {
			return nil
		}
		header := parts[0] // e.g. "data:image/jpeg;base64"
		data := parts[1]

		mediaType := extractMediaType(header)
		if mediaType == "" {
			mediaType = "image/jpeg"
		}

		return &ContentBlock{
			Type: "image",
			Source: &MediaSource{
				Type:      "base64",
				MediaType: mediaType,
				Data:      data,
			},
		}
	}

	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return &ContentBlock{
			Type: "image",
			Source: &MediaSource{
				Type: "url",
				URL:  url,
			},
		}
	}

	return nil
}

// convertDataURLToDocument converts a data URL into an Anthropic document block.
// Only application/* and text/* MIME types are forwarded; others are ignored.
func convertDataURLToDocument(dataURL string) *ContentBlock {
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return nil
	}
	header := parts[0]
	data := parts[1]

	mediaType := extractMediaType(header)
	if mediaType == "" {
		return nil
	}

	if !strings.HasPrefix(mediaType, "application/") && !strings.HasPrefix(mediaType, "text/") {
		return nil
	}

	return &ContentBlock{
		Type: "document",
		Source: &MediaSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		},
	}
}

// extractMediaType extracts the MIME type from a data URL header
// (e.g. "data:image/jpeg;base64" → "image/jpeg").
func extractMediaType(header string) string {
	idx := strings.Index(header, ":")
	if idx < 0 {
		return ""
	}
	rest := header[idx+1:] // e.g. "image/jpeg;base64"
	if idx2 := strings.Index(rest, ";"); idx2 >= 0 {
		return rest[:idx2]
	}
	return rest
}
