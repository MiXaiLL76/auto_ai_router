package common

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
