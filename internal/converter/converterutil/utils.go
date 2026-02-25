package converterutil

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// GenerateID generates a unique chat completion ID.
// Used by multiple transformers to generate response IDs in a consistent format.
func GenerateID() string {
	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)
	return "chatcmpl-" + hex.EncodeToString(bytes)[:20]
}

// GetCurrentTimestamp returns the current Unix timestamp (UTC).
// Used by multiple transformers for response created timestamp.
func GetCurrentTimestamp() int64 {
	return utils.NowUTC().Unix()
}

// GetString safely retrieves a string value from a map.
// Returns empty string if key not found or value is not a string.
func GetString(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

// ExtractTextBlocks returns all text content blocks found in the OpenAI content payload.
// For plain string content, it returns a single-element slice with that string.
func ExtractTextBlocks(content interface{}) []string {
	switch c := content.(type) {
	case string:
		if c == "" {
			return nil
		}
		return []string{c}
	case []interface{}:
		var parts []string
		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if blockMap["type"] != "text" {
				continue
			}
			if text, ok := blockMap["text"].(string); ok && text != "" {
				parts = append(parts, text)
			}
		}
		return parts
	default:
		return nil
	}
}

// EncodeBase64 encodes a byte slice to base64 string.
// Used for preserving binary data like Gemini 3 thoughtSignature in JSON responses.
func EncodeBase64(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	return base64.StdEncoding.EncodeToString(data)
}

// DecodeBase64 decodes a base64 string to byte slice.
// Used for restoring binary data like Gemini 3 thoughtSignature from JSON requests.
// Returns nil if input is empty or invalid base64.
func DecodeBase64(encoded string) []byte {
	if encoded == "" {
		return nil
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil
	}
	return decoded
}
