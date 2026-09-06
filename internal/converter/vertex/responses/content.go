package vertexresponses

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
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
	if !isAllowedFileURL(imgURL) {
		return nil, converterutil.NewRequestValidationError("input_image.image_url", "unsupported or blocked image URL")
	}
	mimeType := detectMIMEFromURL(imgURL)
	if mimeType == "" {
		return nil, converterutil.NewRequestValidationError("input_image.image_url", "cannot determine image MIME type from URL")
	}
	return []*genai.Part{{
		FileData: &genai.FileData{
			MIMEType: mimeType,
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
	// Same limit and same explicit 413 as input_image: Vertex/Gemini's real
	// inline-data cap is a single, media-agnostic 100MB (base64-encoded), and
	// this path previously had no size check at all — any payload, however
	// large, was decoded in full before this fix.
	if len(data) > maxInlineBase64Size {
		return nil, converterutil.NewRequestEntityTooLargeError("input_audio.data", fmt.Sprintf("inline audio payload exceeds %dMB limit", maxInlineBase64Size/(1024*1024)))
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("input_audio: base64 decode: %w", err)
	}
	// Padding-only input (e.g. "====") decodes to zero bytes without erroring;
	// a blob with no bytes is rejected by Gemini with 400 INVALID_ARGUMENT.
	if len(decoded) == 0 {
		return nil, fmt.Errorf("input_audio: empty data")
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
		if !isAllowedFileURL(fileURL) {
			return nil, converterutil.NewRequestValidationError("input_file.file_url", "unsupported or blocked file URL")
		}
		mimeType := detectMIMEFromURL(fileURL)
		if mimeType == "" {
			return nil, converterutil.NewRequestValidationError("input_file.file_url", "cannot determine file MIME type from URL")
		}
		return []*genai.Part{{
			FileData: &genai.FileData{
				MIMEType: mimeType,
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
//
// A var, not a const: tests shrink it (see withMaxInlineBase64Size in
// request_test.go) so boundary cases don't have to allocate and base64-decode
// real ~100MB/~75MB strings just to exercise the ">" comparison.
var maxInlineBase64Size = 100 * 1024 * 1024 // 100MB encoded

// isAllowedFileURL reports whether rawURL is safe to forward to Vertex as a
// FileData reference: an allowed scheme (http/https/gs — anything else,
// including a data: URL that failed to parse as inline data above, must
// never be forwarded as a FileURI, since Vertex would receive it as a
// literal, potentially multi-megabyte string instead of the image/file it
// names), and — for http(s) — not a private/internal network address.
// Mirrors the chat-completions path's parseURLToPart guard (vertex package);
// duplicated here rather than shared because this package already imports
// vertex for other reasons and the check is a handful of lines.
func isAllowedFileURL(rawURL string) bool {
	allowedScheme := strings.HasPrefix(rawURL, "http://") ||
		strings.HasPrefix(rawURL, "https://") ||
		strings.HasPrefix(rawURL, "gs://")
	if !allowedScheme {
		return false
	}
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return !vertex.IsPrivateURL(rawURL)
	}
	return true
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
	// An empty payload decodes cleanly but yields a blob with no bytes, which
	// Gemini rejects with 400 INVALID_ARGUMENT ("required oneof field 'data' must
	// have one initialized field"). Treat it as not a usable data URL.
	if len(decoded) == 0 {
		return nil, nil
	}
	return &genai.Part{
		InlineData: &genai.Blob{MIMEType: mimeType, Data: decoded},
	}, nil
}

// detectMIMEFromURL guesses a MIME type from common file extensions in a URL.
// Returns "" when nothing matches: the bytes are not in hand here (Gemini fetches
// the URI itself), so there is nothing to sniff, and the previous
// "application/octet-stream" default was a guaranteed 400 INVALID_ARGUMENT
// ("Unsupported MIME type: application/octet-stream") that — being retryable —
// was then replayed across every credential. Callers turn "" into a client-visible
// validation error instead.
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
		return ""
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
