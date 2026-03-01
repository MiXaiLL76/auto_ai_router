package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvertOpenAISchemaToAnthropic(t *testing.T) {
	t.Run("nil_schema", func(t *testing.T) {
		result := convertOpenAISchemaToAnthropic(nil)
		assert.Equal(t, map[string]interface{}{"type": "object"}, result)
	})

	t.Run("removes_strict_and_additionalProperties", func(t *testing.T) {
		input := map[string]interface{}{
			"type":                 "Object",
			"strict":               true,
			"additionalProperties": false,
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "String",
				},
			},
		}
		result := convertOpenAISchemaToAnthropic(input)

		// strict and additionalProperties should be removed
		_, hasStrict := result["strict"]
		_, hasAdditional := result["additionalProperties"]
		assert.False(t, hasStrict)
		assert.False(t, hasAdditional)

		// type should be lowercased
		assert.Equal(t, "object", result["type"])

		// Nested properties should also be cleaned
		props := result["properties"].(map[string]interface{})
		nameSchema := props["name"].(map[string]interface{})
		assert.Equal(t, "string", nameSchema["type"])
	})
}

func TestConvertOpenAISchemaToAnthropic_Nested(t *testing.T) {
	input := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"items": map[string]interface{}{
				"type":                 "array",
				"strict":               true,
				"additionalProperties": false,
			},
		},
		"required": []interface{}{"items"},
	}

	result := convertOpenAISchemaToAnthropic(input)

	// Top-level
	assert.Equal(t, "object", result["type"])

	// Nested schema should have strict/additionalProperties removed
	props := result["properties"].(map[string]interface{})
	items := props["items"].(map[string]interface{})
	_, hasStrict := items["strict"]
	_, hasAdditional := items["additionalProperties"]
	assert.False(t, hasStrict)
	assert.False(t, hasAdditional)
	assert.Equal(t, "array", items["type"])

	// required should be preserved as-is (slice of interface{})
	req := result["required"].([]interface{})
	assert.Len(t, req, 1)
	assert.Equal(t, "items", req[0])
}
