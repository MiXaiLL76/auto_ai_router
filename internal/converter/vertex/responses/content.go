package vertexresponses

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"google.golang.org/genai"
)

// contentPartToVertexParts converts a Responses API content part map to genai.Part(s).
// Returns nil parts (not an error) for unknown types that should be silently skipped.
func contentPartToVertexParts(partMap map[string]interface{}) ([]*genai.Part, error) {
	partType, _ := partMap["type"].(string)
	switch partType {
	case "input_text", "output_text", "text":
		text, _ := partMap["text"].(string)
		return []*genai.Part{{Text: text}}, nil

	case "input_image":
		return convertInputImageParts(partMap)

	case "input_audio":
		return convertInputAudioPart(partMap)

	case "input_file":
		return convertInputFilePart(partMap)

	default:
		return nil, nil
	}
}

func convertInputImageParts(partMap map[string]interface{}) ([]*genai.Part, error) {
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

	// Try data: URL first (inline base64)
	if part := parseDataURLToPart(imgURL); part != nil {
		return []*genai.Part{part}, nil
	}

	// Regular https:// URL → fileData
	return []*genai.Part{{
		FileData: &genai.FileData{
			MIMEType: detectMIMEFromURL(imgURL),
			FileURI:  imgURL,
		},
	}}, nil
}

func convertInputAudioPart(partMap map[string]interface{}) ([]*genai.Part, error) {
	data, _ := partMap["data"].(string)
	format, _ := partMap["format"].(string)
	if data == "" {
		return nil, fmt.Errorf("input_audio: missing data")
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("input_audio: base64 decode: %w", err)
	}
	return []*genai.Part{{
		InlineData: &genai.Blob{
			MIMEType: audioFormatToMIME(format),
			Data:     decoded,
		},
	}}, nil
}

func convertInputFilePart(partMap map[string]interface{}) ([]*genai.Part, error) {
	fileURL, _ := partMap["file_url"].(string)
	if fileURL != "" {
		return []*genai.Part{{
			FileData: &genai.FileData{
				MIMEType: detectMIMEFromURL(fileURL),
				FileURI:  fileURL,
			},
		}}, nil
	}

	if fileID, _ := partMap["file_id"].(string); fileID != "" {
		return nil, converterutil.NewRequestValidationError("input_file.file_id", "file_id is not supported for this route")
	}

	return nil, converterutil.NewRequestValidationError("input_file", "missing supported file source")
}

// parseDataURLToPart decodes a data: URL into an InlineData Part.
// Returns nil if the string is not a data URL.
func parseDataURLToPart(dataURL string) *genai.Part {
	if !strings.HasPrefix(dataURL, "data:") {
		return nil
	}
	rest := strings.TrimPrefix(dataURL, "data:")
	semi := strings.Index(rest, ";")
	if semi < 0 {
		return nil
	}
	mimeType := rest[:semi]
	after := rest[semi+1:]
	if !strings.HasPrefix(after, "base64,") {
		return nil
	}
	decoded, err := base64.StdEncoding.DecodeString(after[7:])
	if err != nil {
		return nil
	}
	return &genai.Part{
		InlineData: &genai.Blob{MIMEType: mimeType, Data: decoded},
	}
}

// detectMIMEFromURL guesses a MIME type from common file extensions in a URL.
func detectMIMEFromURL(url string) string {
	lower := strings.ToLower(url)
	switch {
	case strings.Contains(lower, ".jpg") || strings.Contains(lower, ".jpeg"):
		return "image/jpeg"
	case strings.Contains(lower, ".png"):
		return "image/png"
	case strings.Contains(lower, ".gif"):
		return "image/gif"
	case strings.Contains(lower, ".webp"):
		return "image/webp"
	case strings.Contains(lower, ".pdf"):
		return "application/pdf"
	case strings.Contains(lower, ".mp4"):
		return "video/mp4"
	case strings.Contains(lower, ".mp3"):
		return "audio/mp3"
	case strings.Contains(lower, ".wav"):
		return "audio/wav"
	default:
		return "application/octet-stream"
	}
}

// audioFormatToMIME maps Responses API format strings to MIME types.
func audioFormatToMIME(format string) string {
	switch strings.ToLower(format) {
	case "mp3":
		return "audio/mp3"
	case "wav":
		return "audio/wav"
	case "ogg":
		return "audio/ogg"
	case "flac":
		return "audio/flac"
	case "m4a":
		return "audio/m4a"
	case "aac":
		return "audio/aac"
	default:
		return "audio/wav"
	}
}
