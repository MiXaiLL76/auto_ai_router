// Package vertex converts requests and responses between the router's internal form and Google Vertex AI.
package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"google.golang.org/genai"
)

// extractTextContent extracts the first text block from OpenAI message content
func extractTextContent(content interface{}) string {
	parts := converterutil.ExtractTextBlocks(content)
	if len(parts) == 0 {
		return ""
	}
	return parts[0]
}

// convertContentToParts converts OpenAI content format to genai.Part slice.
// The only error case is a block whose content is unusable on its own terms
// (currently: an inline data: URL payload too large to forward) — malformed
// or unrecognized blocks are still skipped silently, as before.
func convertContentToParts(content interface{}) ([]*genai.Part, error) {
	switch c := content.(type) {
	case string:
		return []*genai.Part{{Text: c}}, nil
	case []interface{}:
		var parts []*genai.Part
		type partHandler func(map[string]interface{}) (*genai.Part, error)

		handlers := map[string]partHandler{
			"text": func(block map[string]interface{}) (*genai.Part, error) {
				text, ok := block["text"].(string)
				if !ok {
					return nil, nil
				}
				return &genai.Part{Text: text}, nil
			},
			"image_url": func(block map[string]interface{}) (*genai.Part, error) {
				imageURL, ok := block["image_url"].(map[string]interface{})
				if !ok {
					return nil, nil
				}
				url, ok := imageURL["url"].(string)
				if !ok {
					return nil, nil
				}
				// Try to parse as data URL first, then as regular URL
				part, err := parseDataURLToPart(url)
				if err != nil {
					return nil, err
				}
				if part == nil {
					// If not a data URL, treat as regular URL (http/https)
					part = parseURLToPart(url, imageURL)
				}
				return part, nil
			},
			"input_audio": func(block map[string]interface{}) (*genai.Part, error) {
				audioData, ok := block["input_audio"].(map[string]interface{})
				if !ok {
					return nil, nil
				}
				data, ok := audioData["data"].(string)
				if !ok {
					return nil, nil
				}

				// check base64 payload size before decoding
				if len(data) > 20*1024*1024 {
					return nil, nil
				}

				// Decode base64 audio data
				decodedData, err := base64.StdEncoding.DecodeString(data)
				if err != nil {
					return nil, nil
				}

				// Determine MIME type from format field or default to wav
				mimeType := "audio/wav"
				if format, ok := audioData["format"].(string); ok && format != "" {
					mimeType = getAudioMimeType(format)
				}

				return &genai.Part{
					InlineData: &genai.Blob{
						MIMEType: mimeType,
						Data:     decodedData,
					},
				}, nil
			},
			"video_url": func(block map[string]interface{}) (*genai.Part, error) {
				videoURL, ok := block["video_url"].(map[string]interface{})
				if !ok {
					return nil, nil
				}
				url, ok := videoURL["url"].(string)
				if !ok {
					return nil, nil
				}

				// Determine MIME type from format field or URL extension
				mimeType := ""
				if format, ok := videoURL["format"].(string); ok && format != "" {
					mimeType = format
				} else {
					mimeType = getMimeTypeFromURL(url)
				}

				if mimeType == "" {
					// Default to mp4 if we can't determine
					mimeType = "video/mp4"
				}

				return &genai.Part{
					FileData: &genai.FileData{
						MIMEType: mimeType,
						FileURI:  url,
					},
				}, nil
			},
			"file": func(block map[string]interface{}) (*genai.Part, error) {
				fileObj, ok := block["file"].(map[string]interface{})
				if !ok {
					return nil, nil
				}
				fileID, ok := fileObj["file_id"].(string)
				if !ok {
					return nil, nil
				}

				// Try to parse as data URL first, then as regular URL
				part, err := parseDataURLToPart(fileID)
				if err != nil {
					return nil, err
				}
				if part == nil {
					// If not a data URL, treat as regular URL (http/https)
					part = parseURLToPart(fileID, fileObj)
				}
				return part, nil
			},
		}

		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			contentType, _ := blockMap["type"].(string)
			if handler, ok := handlers[contentType]; ok {
				part, err := handler(blockMap)
				if err != nil {
					return nil, err
				}
				if part != nil {
					parts = append(parts, part)
				}
			}
		}
		return parts, nil
	}
	// use json.Marshal instead of fmt.Sprintf for structured content
	if data, err := json.Marshal(content); err == nil {
		return []*genai.Part{{Text: string(data)}}, nil
	}
	return []*genai.Part{{Text: fmt.Sprintf("%v", content)}}, nil
}
