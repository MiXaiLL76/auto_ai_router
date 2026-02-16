package converter

import "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"

// GenerateID generates a unique chat completion ID. Delegates to converterutil.
func GenerateID() string { return converterutil.GenerateID() }

// GetCurrentTimestamp returns the current Unix timestamp. Delegates to converterutil.
func GetCurrentTimestamp() int64 { return converterutil.GetCurrentTimestamp() }

// GetString safely retrieves a string value from a map. Delegates to converterutil.
func GetString(m map[string]interface{}, key string) string {
	return converterutil.GetString(m, key)
}

// ExtractTextBlocks returns all text content blocks from OpenAI content payload. Delegates to converterutil.
func ExtractTextBlocks(content interface{}) []string {
	return converterutil.ExtractTextBlocks(content)
}
