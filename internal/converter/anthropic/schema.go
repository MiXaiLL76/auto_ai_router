package anthropic

import "strings"

// convertOpenAISchemaToAnthropic converts an OpenAI JSON Schema map to the Anthropic
// input_schema format.  Anthropic uses standard JSON Schema but does not support the
// OpenAI-specific "strict" and "additionalProperties" extensions, so those are stripped.
// Nested schemas are cleaned recursively.
func convertOpenAISchemaToAnthropic(schema map[string]interface{}) map[string]interface{} {
	if schema == nil {
		return map[string]interface{}{"type": "object"}
	}

	result := make(map[string]interface{}, len(schema))
	for k, v := range schema {
		// Drop OpenAI-specific fields unsupported by Anthropic
		if k == "additionalProperties" || k == "strict" {
			continue
		}

		switch val := v.(type) {
		case map[string]interface{}:
			result[k] = convertOpenAISchemaToAnthropic(val)
		case []interface{}:
			cleaned := make([]interface{}, len(val))
			for i, item := range val {
				if itemMap, ok := item.(map[string]interface{}); ok {
					cleaned[i] = convertOpenAISchemaToAnthropic(itemMap)
				} else {
					cleaned[i] = item
				}
			}
			result[k] = cleaned
		default:
			result[k] = v
		}
	}

	// Normalise type value to lowercase (Anthropic requires lowercase types)
	if typeVal, ok := result["type"].(string); ok {
		result["type"] = strings.ToLower(typeVal)
	}

	return result
}
