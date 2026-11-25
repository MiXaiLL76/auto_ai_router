package anthropic

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"net"
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
		block, err := DownloadAndEncodeImage(url, imageDownloadParam)
		if err != nil {
			return nil, err
		}
		return block, nil
	}

	return nil, converterutil.NewRequestValidationError("messages.content.image_url.url", "unsupported image_url source")
}

// imageDownloadParam is the request field reported to the client when an
// image URL cannot be resolved. It names the field to fix without disclosing
// anything about the target host.
const imageDownloadParam = "messages.content.image_url.url"

// maxImageDownloadBytes caps how much of a remote image the router will
// download. Anthropic's own limit for a base64 image is 5 MB per image;
// refusing anything much larger keeps a single client-supplied URL from
// pulling an unbounded payload into the router's memory and then into the
// base64-encoded request body.
var maxImageDownloadBytes = 25 * 1024 * 1024

// imageHTTPClient fetches remote images on the client's behalf.
//
// The dialer refuses to connect to private or link-local addresses, so a
// client-supplied URL cannot be used to reach the pod's own network, the
// cluster API, or a cloud metadata endpoint (SSRF). The timeouts are
// deliberately short: the router is in the request path, and a client that
// waited out a 15s image fetch would abandon the request long before the
// router gave up on it.
//
// No HTTP_PROXY-style proxy is used here on purpose. A forward proxy is
// reached by address, so the guard below would only vet the proxy's IP and
// never the target host — the SSRF check would be bypassed entirely, and
// the proxy could be used to reach link-local metadata endpoints on its own
// host. Image URLs are public CDN links by nature, so they are fetched
// directly.
var imageHTTPClient = &http.Client{
	Timeout: imageDownloadTimeout,
	Transport: &http.Transport{
		// The dialer is read at connect time rather than captured here, so
		// the SSRF guard can be swapped out in tests without rebuilding the
		// client.
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return imageDialContext(ctx, network, addr)
		},
		TLSHandshakeTimeout:   imageDialTimeout,
		ResponseHeaderTimeout: imageDownloadTimeout,
		ExpectContinueTimeout: time.Second,
		MaxIdleConnsPerHost:   4,
		DisableCompression:    true,
	},
}

// imageDialContext is the dialer used for image downloads. It is a variable
// so tests can substitute a permissive dialer for a loopback httptest
// server; production always uses the SSRF-guarded one.
var imageDialContext = converterutil.GuardedDialContext(&net.Dialer{Timeout: imageDialTimeout})

// imageURLGuard vets an image URL before it is fetched. It is a variable for
// the same reason as imageDialContext: tests need to reach a loopback
// httptest server, which the real guard rejects. The guard itself is covered
// by its own tests in converterutil.
var imageURLGuard = converterutil.IsPrivateURL

const (
	imageDialTimeout     = 3 * time.Second
	imageDownloadTimeout = 8 * time.Second
)

// allowedImageMediaTypes lists the media types accepted by the Anthropic API.
var allowedImageMediaTypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

// DownloadAndEncodeImage fetches an image from an HTTP/HTTPS URL and returns a base64
// ContentBlock. Anthropic does not support URL sources, so we must download and encode.
//
// The returned error is always a converterutil.ImageError, whose client-facing
// message is fixed per failure kind and never mentions the URL, the host, or
// the upstream status. The detailed reason (and the URL) goes to the router's
// internal log only.
//
// param names the request field the client must fix (e.g.
// "messages.content.image_url.url"); it is echoed to the client as the
// "param" field of the error and must not contain the image URL itself.
func DownloadAndEncodeImage(imgURL, param string) (*ContentBlock, error) {
	// Reject URLs pointing at private/internal address space before making
	// any request: the router fetches this URL itself, so an unvetted URL
	// would let a client reach the pod network, the cluster API, or a cloud
	// metadata endpoint.
	if imageURLGuard(imgURL) {
		slog.Warn("Refusing to download image from a non-public address",
			"url", imgURL, "reason", "private_address")
		return nil, converterutil.NewImageError(converterutil.ImageErrorURLNotAllowed, param)
	}

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, imgURL, nil)
	if err != nil {
		slog.Warn("Failed to create image download request", "url", imgURL, "error", err)
		return nil, converterutil.NewImageError(converterutil.ImageErrorFetchFailed, param)
	}
	// A browser-like User-Agent: some CDNs reject unknown clients outright.
	// Note this only helps against naive bot filters — Wikimedia, for
	// instance, rejects browser-like UAs from datacenter IPs as well, so a
	// blocked host stays blocked regardless of the User-Agent.
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; auto-ai-router/1.0)")
	req.Header.Set("Accept", "image/*,*/*;q=0.8")

	resp, err := imageHTTPClient.Do(req)
	if err != nil {
		slog.Warn("Failed to download image", "url", imgURL, "error", err)
		return nil, converterutil.NewImageError(converterutil.ImageErrorFetchFailed, param)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			slog.Warn("Failed to close image body", "error", err)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		slog.Warn("Image download returned non-200 status",
			"url", imgURL, "status", resp.StatusCode)
		return nil, converterutil.NewImageError(converterutil.ImageErrorFetchFailed, param)
	}

	// Read one byte past the limit so an oversized body is detected without
	// buffering the whole thing.
	data, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxImageDownloadBytes)+1))
	if err != nil {
		slog.Warn("Failed to read image body", "url", imgURL, "error", err)
		return nil, converterutil.NewImageError(converterutil.ImageErrorFetchFailed, param)
	}
	if len(data) > maxImageDownloadBytes {
		slog.Warn("Image exceeds the download limit",
			"url", imgURL, "limit_bytes", maxImageDownloadBytes)
		return nil, converterutil.NewImageError(converterutil.ImageErrorTooLarge, param)
	}
	if len(data) == 0 {
		slog.Warn("Image response body is empty", "url", imgURL)
		return nil, converterutil.NewImageError(converterutil.ImageErrorInvalidData, param)
	}

	mediaType := resp.Header.Get("Content-Type")
	if idx := strings.Index(mediaType, ";"); idx != -1 {
		mediaType = strings.TrimSpace(mediaType[:idx])
	}
	mediaType = strings.ToLower(mediaType)

	// A content type the provider accepts is authoritative — the bytes may
	// still be junk, but the server claims to be sending an image.
	if allowedImageMediaTypes[mediaType] {
		return &ContentBlock{
			Type: "image",
			Source: &MediaSource{
				Type:      "base64",
				MediaType: mediaType,
				Data:      base64.StdEncoding.EncodeToString(data),
			},
		}, nil
	}

	// The content type is missing or not an image type. Sniff the actual
	// format from the bytes: plenty of hosts serve images as
	// application/octet-stream or with no type at all.
	detected := detectImageMediaType(data, imgURL)
	if detected == "" {
		slog.Warn("Image URL did not return a recognizable image",
			"url", imgURL, "content_type", mediaType)
		return nil, converterutil.NewImageError(converterutil.ImageErrorInvalidData, param)
	}
	if !allowedImageMediaTypes[detected] {
		slog.Warn("Unsupported image media type", "url", imgURL, "media_type", detected)
		return nil, converterutil.NewImageError(converterutil.ImageErrorUnsupportedMediaType, param)
	}

	return &ContentBlock{
		Type: "image",
		Source: &MediaSource{
			Type:      "base64",
			MediaType: detected,
			Data:      base64.StdEncoding.EncodeToString(data),
		},
	}, nil
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
