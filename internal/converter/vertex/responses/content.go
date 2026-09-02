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
		if fileID, _ := partMap["file_id"].(string); fileID != "" {
			return nil, converterutil.NewRequestValidationError("input_image.file_id", "file_id is not supported for this route")
		}
		return nil, converterutil.NewRequestValidationError("input_image", "missing supported image source")
	}

	// Try data: URL first (inline base64)
	part, err := parseDataURLToPart(imgURL)
	if err != nil {
		return nil, err
	}
	if part != nil {
		return []*genai.Part{part}, nil
	}

	// Regular https:// URL → fileData. Reject anything else — most notably a
	// data: URL that failed to parse above (unrecognized encoding, no comma,
	// non-base64 payload) — instead of forwarding it to Vertex as a literal,
	// potentially multi-megabyte FileURI.
	if !isRemoteFileScheme(imgURL) {
		return nil, converterutil.NewRequestValidationError("input_image.image_url", "unsupported image URL scheme")
	}
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
		if !isRemoteFileScheme(fileURL) {
			return nil, converterutil.NewRequestValidationError("input_file.file_url", "unsupported file URL scheme")
		}
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

// maxInlineBase64Size caps the base64-encoded (not decoded) payload of an
// inline data: URL. Above this, we reject the request with 413 instead of
// silently dropping the media or forwarding a request Vertex would reject
// anyway once decoded.
const maxInlineBase64Size = 100 * 1024 * 1024 // 100MB encoded

// isRemoteFileScheme reports whether rawURL uses a scheme Vertex can actually
// fetch as a FileData reference. Anything else — including a data: URL that
// failed to parse as inline data above — must never be forwarded as a
// FileURI: Vertex would receive it as a literal, potentially multi-megabyte
// string instead of the image/file it names.
func isRemoteFileScheme(rawURL string) bool {
	return strings.HasPrefix(rawURL, "http://") ||
		strings.HasPrefix(rawURL, "https://") ||
		strings.HasPrefix(rawURL, "gs://")
}

// parseDataURLToPart decodes a data: URL into an InlineData Part.
// Returns (nil, nil) if the string is not a data URL — callers fall back to
// treating it as a regular URL. Returns a non-nil error only when the string
// IS a data URL but is otherwise unusable (oversized or malformed base64).
//
// Splits on the first comma rather than requiring a literal ";base64," marker
// right after the MIME type: real-world data URLs commonly carry extra
// parameters first (e.g. "data:image/svg+xml;charset=utf-8;base64,...."),
// which the stricter form used to misdetect as "not a data URL" and forward
// to the caller's FileURI fallback (an mp3-sized string as a "URL").
func parseDataURLToPart(dataURL string) (*genai.Part, error) {
	if !strings.HasPrefix(dataURL, "data:") {
		return nil, nil
	}
	parts := strings.SplitN(dataURL, ",", 2) // SplitN to handle base64 with commas
	if len(parts) != 2 {
		return nil, nil
	}
	header := strings.TrimPrefix(parts[0], "data:")
	encoded := parts[1]

	mimeType := header
	if semi := strings.Index(header, ";"); semi >= 0 {
		mimeType = header[:semi]
	}
	if mimeType == "" {
		return nil, nil
	}

	if len(encoded) > maxInlineBase64Size {
		return nil, converterutil.NewRequestEntityTooLargeError("", fmt.Sprintf("inline data URL payload exceeds %dMB limit", maxInlineBase64Size/(1024*1024)))
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, nil
	}
	return &genai.Part{
		InlineData: &genai.Blob{MIMEType: mimeType, Data: decoded},
	}, nil
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
