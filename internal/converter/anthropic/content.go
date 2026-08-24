package anthropic

import (
	"encoding/base64"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

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
					cb := ContentBlock{Type: "text", Text: text}
					cb.CacheControl = blockMap["cache_control"]
					blocks = append(blocks, cb)
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
					cb.CacheControl = blockMap["cache_control"]
					blocks = append(blocks, *cb)
				}

			case "input_audio":
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: "[Audio input not supported by Anthropic API]",
				})

			case "video":
				// An Anthropic-shaped video block, preserved by MessagesToChat from a
				// client request. Anthropic itself has no video block, but some
				// Anthropic-compatible providers do, and the client picked that shape
				// deliberately — forward it and let the provider accept or reject it.
				source, ok := blockMap["source"].(map[string]interface{})
				if !ok {
					continue
				}
				mediaSource := mediaSourceFromMap(source)
				if mediaSource == nil {
					continue
				}
				cb := ContentBlock{Type: "video", Source: mediaSource}
				cb.CacheControl = blockMap["cache_control"]
				blocks = append(blocks, cb)

			case "video_url":
				// OpenAI's own video part has no Anthropic equivalent to map onto, so it
				// stays an explicit marker rather than a silently missing attachment.
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: "[Video input not supported by Anthropic API]",
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
						cb.CacheControl = blockMap["cache_control"]
						blocks = append(blocks, *cb)
					}
				} else {
					slog.Warn("unsupported file reference in Anthropic conversion, skipping",
						"file_id", fileID)
				}
			}
		}
		return blocks
	}
	return nil
}

// mediaSourceFromMap rebuilds a MediaSource from a decoded content-block source object.
// Returns nil when the source is neither a usable url nor base64 payload, so callers can
// skip the block instead of forwarding an empty attachment.
func mediaSourceFromMap(source map[string]interface{}) *MediaSource {
	mediaSource := &MediaSource{
		Type:      stringValue(source["type"]),
		MediaType: stringValue(source["media_type"]),
		Data:      stringValue(source["data"]),
		URL:       stringValue(source["url"]),
	}
	switch mediaSource.Type {
	case "url":
		if mediaSource.URL == "" {
			return nil
		}
	case "base64":
		if mediaSource.Data == "" {
			return nil
		}
	default:
		return nil
	}
	return mediaSource
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
		return DownloadAndEncodeImage(url)
	}

	return nil
}

var imageHTTPClient = &http.Client{Timeout: 15 * time.Second}

// allowedImageMediaTypes lists the media types accepted by the Anthropic API.
var allowedImageMediaTypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

// DownloadAndEncodeImage fetches an image from an HTTP/HTTPS URL and returns a base64
// ContentBlock. Anthropic does not support URL sources, so we must download and encode.
func DownloadAndEncodeImage(imgURL string) *ContentBlock {
	req, err := http.NewRequest(http.MethodGet, imgURL, nil)
	if err != nil {
		slog.Warn("Failed to create image download request for Anthropic", "url", imgURL, "error", err)
		return nil
	}
	// Use a browser-like User-Agent to avoid bot-blocking (e.g. Wikimedia returns 403 for Go-http-client).
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; auto-ai-router/1.0)")

	resp, err := imageHTTPClient.Do(req)
	if err != nil {
		slog.Warn("Failed to download image for Anthropic", "url", imgURL, "error", err)
		return nil
	}
	defer func() {
		err := resp.Body.Close()
		if err != nil {
			slog.Warn("Failed to close body", "error", err)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		slog.Warn("Image download returned non-200 status for Anthropic",
			"url", imgURL, "status", resp.StatusCode)
		return nil
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		slog.Warn("Failed to read image body for Anthropic", "url", imgURL, "error", err)
		return nil
	}

	mediaType := resp.Header.Get("Content-Type")
	if idx := strings.Index(mediaType, ";"); idx != -1 {
		mediaType = strings.TrimSpace(mediaType[:idx])
	}
	// If Content-Type is missing or not an accepted image type, detect from magic bytes.
	if !allowedImageMediaTypes[mediaType] {
		mediaType = detectImageMediaType(data, imgURL)
	}

	if !allowedImageMediaTypes[mediaType] {
		slog.Warn("Unsupported image media type for Anthropic, skipping",
			"url", imgURL, "media_type", mediaType)
		return nil
	}

	return &ContentBlock{
		Type: "image",
		Source: &MediaSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      base64.StdEncoding.EncodeToString(data),
		},
	}
}

// detectImageMediaType detects image format from magic bytes, falling back to URL extension.
func detectImageMediaType(data []byte, imgURL string) string {
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "image/jpeg"
	}
	if len(data) >= 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n" {
		return "image/png"
	}
	if len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a") {
		return "image/gif"
	}
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		return "image/webp"
	}
	// Fall back to URL extension.
	lower := strings.ToLower(imgURL)
	switch {
	case strings.Contains(lower, ".jpg") || strings.Contains(lower, ".jpeg"):
		return "image/jpeg"
	case strings.Contains(lower, ".png"):
		return "image/png"
	case strings.Contains(lower, ".gif"):
		return "image/gif"
	case strings.Contains(lower, ".webp"):
		return "image/webp"
	}
	return ""
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
