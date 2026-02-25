package vertex

import (
	"encoding/base64"
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

// convertContentToParts converts OpenAI content format to genai.Part slice
func convertContentToParts(content interface{}) []*genai.Part {
	switch c := content.(type) {
	case string:
		return []*genai.Part{{Text: c}}
	case []interface{}:
		var parts []*genai.Part
		type partHandler func(map[string]interface{}) *genai.Part

		handlers := map[string]partHandler{
			"text": func(block map[string]interface{}) *genai.Part {
				text, ok := block["text"].(string)
				if !ok {
					return nil
				}
				return &genai.Part{Text: text}
			},
			"image_url": func(block map[string]interface{}) *genai.Part {
				imageURL, ok := block["image_url"].(map[string]interface{})
				if !ok {
					return nil
				}
				url, ok := imageURL["url"].(string)
				if !ok {
					return nil
				}
				// Try to parse as data URL first, then as regular URL
				part := parseDataURLToPart(url)
				if part == nil {
					// If not a data URL, treat as regular URL (http/https)
					part = parseURLToPart(url, imageURL)
				}
				return part
			},
			"input_audio": func(block map[string]interface{}) *genai.Part {
				audioData, ok := block["input_audio"].(map[string]interface{})
				if !ok {
					return nil
				}
				data, ok := audioData["data"].(string)
				if !ok {
					return nil
				}

				// Decode base64 audio data
				decodedData, err := base64.StdEncoding.DecodeString(data)
				if err != nil {
					return nil
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
				}
			},
			"video_url": func(block map[string]interface{}) *genai.Part {
				videoURL, ok := block["video_url"].(map[string]interface{})
				if !ok {
					return nil
				}
				url, ok := videoURL["url"].(string)
				if !ok {
					return nil
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
				}
			},
			"file": func(block map[string]interface{}) *genai.Part {
				fileObj, ok := block["file"].(map[string]interface{})
				if !ok {
					return nil
				}
				fileID, ok := fileObj["file_id"].(string)
				if !ok {
					return nil
				}

				// Try to parse as data URL first, then as regular URL
				part := parseDataURLToPart(fileID)
				if part == nil {
					// If not a data URL, treat as regular URL (http/https)
					part = parseURLToPart(fileID, fileObj)
				}
				return part
			},
		}

		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			contentType, _ := blockMap["type"].(string)
			if handler, ok := handlers[contentType]; ok {
				if part := handler(blockMap); part != nil {
					parts = append(parts, part)
				}
			}
		}
		return parts
	}
	return []*genai.Part{{Text: fmt.Sprintf("%v", content)}}
}
