package anthropic

import (
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// convertOpenAIContentToAnthropic converts an OpenAI message content value (string or
// []interface{} of content blocks) into a slice of Anthropic ContentBlocks.
func convertOpenAIContentToAnthropic(content interface{}) ([]ContentBlock, error) {
	switch c := content.(type) {
	case string:
		if c == "" {
			return nil, nil
		}
		return []ContentBlock{{Type: "text", Text: c}}, nil

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
					return nil, converterutil.NewRequestValidationError("messages.content.image_url", "missing image_url object")
				}
				url, _ := imageURL["url"].(string)
				if url == "" {
					return nil, converterutil.NewRequestValidationError("messages.content.image_url.url", "missing image URL")
				}
				cb, err := convertImageURLToAnthropic(url)
				if err != nil {
					return nil, err
				}
				cb.CacheControl = blockMap["cache_control"]
				blocks = append(blocks, *cb)

			case "input_audio":
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: "[Audio input not supported by Anthropic API]",
				})

			case "video_url":
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: "[Video input not supported by Anthropic API]",
				})

			case "file":
				fileObj, ok := blockMap["file"].(map[string]interface{})
				if !ok {
					return nil, converterutil.NewRequestValidationError("messages.content.file", "missing file object")
				}
				cb, err := convertFileBlockToDocument(fileObj)
				if err != nil {
					return nil, err
				}
				cb.CacheControl = blockMap["cache_control"]
				blocks = append(blocks, *cb)

			case "document":
				cb, err := convertDocumentBlockToAnthropic(blockMap)
				if err != nil {
					return nil, err
				}
				cb.CacheControl = blockMap["cache_control"]
				blocks = append(blocks, *cb)
			}
		}
		return blocks, nil
	}
	return nil, nil
}

// convertImageURLToAnthropic converts an OpenAI image_url value to an Anthropic image block.
// Supports base64 data URLs and plain HTTPS/HTTP URLs.
func convertImageURLToAnthropic(url string) (*ContentBlock, error) {
	if strings.HasPrefix(url, "data:") {
		mediaType, data, err := ParseBase64DataURL(url, "messages.content.image_url.url")
		if err != nil {
			return nil, err
		}
		if mediaType == "application/pdf" {
			return nil, converterutil.NewRequestValidationError("messages.content.image_url.url", "PDF data URLs must be sent as file/document content, not image_url")
		}
		if !allowedImageMediaTypes[mediaType] {
			return nil, converterutil.NewRequestValidationError("messages.content.image_url.url", fmt.Sprintf("unsupported image media type %q for Anthropic", mediaType))
		}

		return &ContentBlock{
			Type: "image",
			Source: &MediaSource{
				Type:      "base64",
				MediaType: mediaType,
				Data:      data,
			},
		}, nil
	}

	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		block := DownloadAndEncodeImage(url)
		if block == nil {
			return nil, fmt.Errorf("image_url: failed to download image")
		}
		return block, nil
	}

	return nil, converterutil.NewRequestValidationError("messages.content.image_url.url", "unsupported image_url source")
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

// ParseBase64DataURL parses a data: URL and validates base64 encoding.
// It returns the MIME type and raw base64 payload without the data URL prefix.
func ParseBase64DataURL(dataURL, param string) (mimeType, b64data string, err error) {
	if !strings.HasPrefix(dataURL, "data:") {
		return "", "", converterutil.NewRequestValidationError(param, "expected data URL")
	}
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return "", "", converterutil.NewRequestValidationError(param, "invalid data URL: missing comma")
	}
	header := parts[0]
	data := parts[1]
	if data == "" {
		return "", "", converterutil.NewRequestValidationError(param, "invalid data URL: empty base64 payload")
	}

	mediaType := extractMediaType(header)
	if mediaType == "" {
		return "", "", converterutil.NewRequestValidationError(param, "invalid data URL: missing media type")
	}
	mediaType = strings.ToLower(mediaType)
	if !dataURLHeaderHasBase64(header) {
		return "", "", converterutil.NewRequestValidationError(param, "invalid data URL: expected base64 encoding")
	}
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return "", "", converterutil.NewRequestValidationError(param, "invalid data URL: base64 payload is malformed")
	}

	return mediaType, data, nil
}

func dataURLHeaderHasBase64(header string) bool {
	semi := strings.Split(header, ";")
	for _, part := range semi[1:] {
		if strings.EqualFold(strings.TrimSpace(part), "base64") {
			return true
		}
	}
	return false
}

func convertFileBlockToDocument(fileObj map[string]interface{}) (*ContentBlock, error) {
	if fileData, _ := fileObj["file_data"].(string); fileData != "" {
		return ConvertDataURLToDocument(fileData, "messages.content.file.file_data")
	}
	if fileID, _ := fileObj["file_id"].(string); fileID != "" {
		return nil, converterutil.NewRequestValidationError("messages.content.file.file_id", "file_id is not supported for this route")
	}
	return nil, converterutil.NewRequestValidationError("messages.content.file", "missing supported file source")
}

func convertDocumentBlockToAnthropic(blockMap map[string]interface{}) (*ContentBlock, error) {
	source, ok := blockMap["source"].(map[string]interface{})
	if !ok {
		return nil, converterutil.NewRequestValidationError("messages.content.document.source", "missing document source")
	}
	return ConvertDocumentSourceToBlock(source, "messages.content.document.source")
}

// ConvertDataURLToDocument converts a document data URL into an Anthropic document block.
func ConvertDataURLToDocument(dataURL, param string) (*ContentBlock, error) {
	mediaType, data, err := ParseBase64DataURL(dataURL, param)
	if err != nil {
		return nil, err
	}
	if !allowedDocumentMediaTypes[mediaType] {
		return nil, converterutil.NewRequestValidationError(param, fmt.Sprintf("unsupported document media type %q for Anthropic", mediaType))
	}

	return &ContentBlock{
		Type: "document",
		Source: &MediaSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		},
	}, nil
}

// ConvertDocumentSourceToBlock validates and converts an Anthropic document source map.
func ConvertDocumentSourceToBlock(source map[string]interface{}, param string) (*ContentBlock, error) {
	sourceType, _ := source["type"].(string)
	switch sourceType {
	case "base64":
		mediaType, _ := source["media_type"].(string)
		if mediaType == "" {
			return nil, converterutil.NewRequestValidationError(param+".media_type", "missing document media_type")
		}
		if !allowedDocumentMediaTypes[mediaType] {
			return nil, converterutil.NewRequestValidationError(param+".media_type", fmt.Sprintf("unsupported document media type %q for Anthropic", mediaType))
		}
		data, _ := source["data"].(string)
		if data == "" {
			return nil, converterutil.NewRequestValidationError(param+".data", "missing document base64 data")
		}
		if _, err := base64.StdEncoding.DecodeString(data); err != nil {
			return nil, converterutil.NewRequestValidationError(param+".data", "document base64 data is malformed")
		}
		return &ContentBlock{
			Type: "document",
			Source: &MediaSource{
				Type:      "base64",
				MediaType: mediaType,
				Data:      data,
			},
		}, nil
	case "url":
		url, _ := source["url"].(string)
		if url == "" {
			return nil, converterutil.NewRequestValidationError(param+".url", "missing document URL")
		}
		return &ContentBlock{
			Type: "document",
			Source: &MediaSource{
				Type: "url",
				URL:  url,
			},
		}, nil
	case "file", "file_id", "provider_file":
		return nil, converterutil.NewRequestValidationError(param+".file_id", "file_id is not supported for this route")
	default:
		return nil, converterutil.NewRequestValidationError(param+".type", "unsupported document source type")
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

var allowedDocumentMediaTypes = map[string]bool{
	"application/pdf": true,
}
