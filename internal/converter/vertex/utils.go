package vertex

import (
	"encoding/base64"
	"strings"

	"google.golang.org/genai"
)

// parseDataURLToPart converts a data URL string to a genai.Part with inline data
// Handles formats like: data:image/jpeg;base64,/9j/4AAQ...
func parseDataURLToPart(dataURL string) *genai.Part {
	if !strings.HasPrefix(dataURL, "data:") {
		return nil
	}

	// Split: data:image/jpeg;base64,<data>
	parts := strings.Split(dataURL, ",")
	if len(parts) != 2 {
		return nil
	}

	header := parts[0]  // data:image/jpeg;base64
	b64Data := parts[1] // base64 data

	// Extract mime type from header
	mimeType := extractMimeType(header)
	if mimeType == "" {
		return nil
	}

	// Decode base64 data to binary
	decodedData, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		return nil
	}

	return &genai.Part{
		InlineData: &genai.Blob{
			MIMEType: mimeType,
			Data:     decodedData,
		},
	}
}

// extractMimeType extracts MIME type from data URL header
// Example: "data:image/jpeg;base64" -> "image/jpeg"
func extractMimeType(header string) string {
	// Find start of mime type (after "data:")
	start := strings.Index(header, "data:")
	if start < 0 {
		return ""
	}
	start += 5 // len("data:")

	// Find end of mime type (at ";" or end of string)
	end := strings.Index(header[start:], ";")
	if end > 0 {
		return header[start : start+end]
	}

	// No semicolon, take from start to end
	return header[start:]
}

// mimeTypeMap maps file extensions to MIME types
var mimeTypeMap = map[string]string{
	"jpg":  "image/jpeg",
	"jpeg": "image/jpeg",
	"png":  "image/png",
	"gif":  "image/gif",
	"webp": "image/webp",
	"mp4":  "video/mp4",
	"mpeg": "video/mpeg",
	"mov":  "video/quicktime",
	"avi":  "video/x-msvideo",
	"mkv":  "video/x-matroska",
	"webm": "video/webm",
	"flv":  "video/x-flv",
	"pdf":  "application/pdf",
	"txt":  "text/plain",
}

// parseURLToPart converts a regular URL or file reference to a genai.Part.
// Supports http/https/gs/file URLs and determines MIME type from format or file extension.
func parseURLToPart(url string, fileObj map[string]interface{}) *genai.Part {
	if url == "" {
		return nil
	}

	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") &&
		!strings.HasPrefix(url, "gs://") && !strings.HasPrefix(url, "file://") {
		return nil
	}

	// Determine MIME type from explicit format field or URL extension
	mimeType := ""
	if format, ok := fileObj["format"].(string); ok && format != "" {
		mimeType = format
	} else {
		mimeType = getMimeTypeFromURL(url)
	}

	if mimeType == "" {
		mimeType = "application/octet-stream"
	}

	return &genai.Part{
		FileData: &genai.FileData{
			MIMEType: mimeType,
			FileURI:  url,
		},
	}
}

// getMimeTypeFromURL determines MIME type from URL extension
func getMimeTypeFromURL(url string) string {
	// Extract extension from URL (before query parameters)
	urlPath := url
	if idx := strings.Index(urlPath, "?"); idx > 0 {
		urlPath = urlPath[:idx]
	}

	// Get extension
	ext := ""
	if idx := strings.LastIndex(urlPath, "."); idx > 0 {
		ext = strings.ToLower(urlPath[idx+1:])
	}

	if mimeType, ok := mimeTypeMap[ext]; ok {
		return mimeType
	}

	return ""
}

// getAudioMimeType maps audio format to MIME type
func getAudioMimeType(format string) string {
	formatLower := strings.ToLower(format)
	mimeTypes := map[string]string{
		"wav":  "audio/wav",
		"mp3":  "audio/mpeg",
		"ogg":  "audio/ogg",
		"opus": "audio/opus",
		"aac":  "audio/aac",
		"flac": "audio/flac",
		"m4a":  "audio/mp4",
		"weba": "audio/webp",
	}

	if mimeType, ok := mimeTypes[formatLower]; ok {
		return mimeType
	}

	// Default to wav if format is not recognized
	return "audio/wav"
}
