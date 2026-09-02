package vertex

import (
	"context"
	"encoding/base64"
	"fmt"
	"math"
	"net"
	"net/url"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"google.golang.org/genai"
)

// maxBase64Size caps the base64-encoded (not decoded) payload of an inline
// data: URL. Above this, the request is rejected with 413 instead of either
// silently dropping the media or (the previous behaviour) falling through to
// parseURLToPart, which would treat the same giant base64 string as if it
// were a plain http(s) URL and forward a malformed request to Vertex.
const maxBase64Size = 100 * 1024 * 1024 // 100MB encoded

// ClampInt32 narrows a caller-supplied integer to int32, saturating instead of
// wrapping on out-of-range values. The genai SDK takes int32 for token counts,
// candidate counts and similar fields; a client sending an absurd value should
// be capped, never silently turned negative.
func ClampInt32[T int | int64](v T) int32 {
	if v > T(math.MaxInt32) {
		return math.MaxInt32
	}
	if v < T(math.MinInt32) {
		return math.MinInt32
	}
	return int32(v)
}

// parseDataURLToPart converts a data URL string to a genai.Part with inline data
// Handles formats like: data:image/jpeg;base64,/9j/4AAQ...
// Returns (nil, nil) if dataURL is not a data: URL at all — callers fall back
// to treating it as a regular URL. Returns a non-nil error only when the
// string IS a data URL but is otherwise unusable (oversized payload).
func parseDataURLToPart(dataURL string) (*genai.Part, error) {
	if !strings.HasPrefix(dataURL, "data:") {
		return nil, nil
	}

	// Split: data:image/jpeg;base64,<data>
	parts := strings.SplitN(dataURL, ",", 2) // SplitN to handle base64 with commas
	if len(parts) != 2 {
		return nil, nil
	}

	header := parts[0]  // data:image/jpeg;base64
	b64Data := parts[1] // base64 data

	// Extract mime type from header
	mimeType := extractMimeType(header)
	if mimeType == "" {
		return nil, nil
	}

	// check base64 payload size before decoding
	if len(b64Data) > maxBase64Size {
		return nil, converterutil.NewRequestEntityTooLargeError("", fmt.Sprintf("inline data URL payload exceeds %dMB limit", maxBase64Size/(1024*1024)))
	}

	// Decode base64 data to binary
	decodedData, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		return nil, nil
	}

	return &genai.Part{
		InlineData: &genai.Blob{
			MIMEType: mimeType,
			Data:     decodedData,
		},
	}, nil
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

// IsPrivateURL checks if a URL points to a private/internal network address.
// Exported so the Responses API converter package (vertexresponses) can apply
// the same SSRF guard to its own FileData URL fallback.
//
// When the hostname is not a literal IP, it is resolved once via net.LookupIP
// and the result is checked against private ranges.  This is vulnerable to DNS
// rebinding: an attacker can serve a public IP during this check and then flip
// the DNS record to a private address before Vertex AI makes its own request.
// The risk is accepted because (a) Vertex AI fetches the URL server-side and
// may apply its own protections, and (b) exploitation requires the attacker to
// control DNS with a very short TTL.  Do not rely on this function as the sole
// SSRF defence in higher-trust environments.
func IsPrivateURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return true // block unparseable URLs
	}

	hostname := parsed.Hostname()

	// Block well-known metadata endpoints
	blockedHosts := []string{"metadata.google.internal", "metadata.aws.internal"}
	for _, blocked := range blockedHosts {
		if strings.EqualFold(hostname, blocked) {
			return true
		}
	}

	// Block localhost
	if strings.EqualFold(hostname, "localhost") {
		return true
	}

	// Resolve hostname and check IP ranges
	ip := net.ParseIP(hostname)
	if ip == nil {
		// Could be a hostname — resolve it
		ips, err := net.DefaultResolver.LookupIPAddr(context.Background(), hostname)
		if err != nil || len(ips) == 0 {
			return false // can't resolve, let Vertex handle it
		}
		ip = ips[0].IP
	}

	// Private IP ranges
	privateRanges := []string{
		"127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
		"192.168.0.0/16", "169.254.0.0/16", "::1/128", "fc00::/7",
	}
	for _, cidr := range privateRanges {
		_, network, err := net.ParseCIDR(cidr)
		if err != nil {
			continue
		}
		if network.Contains(ip) {
			return true
		}
	}

	return false
}

// parseURLToPart converts a regular URL or file reference to a genai.Part.
// Supports https and gs:// URLs. Blocks file:// and private network URLs (SSRF protection).
func parseURLToPart(rawURL string, fileObj map[string]interface{}) *genai.Part {
	if rawURL == "" {
		return nil
	}

	// block file:// URLs completely (SSRF vector)
	if strings.HasPrefix(rawURL, "file://") {
		return nil
	}

	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") &&
		!strings.HasPrefix(rawURL, "gs://") {
		return nil
	}

	//  block private/internal network URLs
	if (strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://")) && IsPrivateURL(rawURL) {
		return nil
	}

	// Determine MIME type from explicit format field or URL extension
	mimeType := ""
	if format, ok := fileObj["format"].(string); ok && format != "" {
		mimeType = format
	} else {
		mimeType = getMimeTypeFromURL(rawURL)
	}

	if mimeType == "" {
		mimeType = "application/octet-stream"
	}

	return &genai.Part{
		FileData: &genai.FileData{
			MIMEType: mimeType,
			FileURI:  rawURL,
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
		"weba": "audio/webm",
	}

	if mimeType, ok := mimeTypes[formatLower]; ok {
		return mimeType
	}

	// Default to wav if format is not recognized
	return "audio/wav"
}
