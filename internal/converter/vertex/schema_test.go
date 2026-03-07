package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertMapToGenaiSchema(t *testing.T) {
	t.Run("nil map returns nil", func(t *testing.T) {
		result := convertMapToGenaiSchema(nil)
		assert.Nil(t, result)
	})

	t.Run("simple object with string and int properties", func(t *testing.T) {
		data := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "The name",
				},
				"age": map[string]interface{}{
					"type": "integer",
				},
			},
			"required": []interface{}{"name"},
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		assert.Equal(t, "OBJECT", string(result.Type))
		assert.Contains(t, result.Properties, "name")
		assert.Contains(t, result.Properties, "age")
		assert.Equal(t, "STRING", string(result.Properties["name"].Type))
		assert.Equal(t, "The name", result.Properties["name"].Description)
		assert.Equal(t, "INTEGER", string(result.Properties["age"].Type))
		assert.Equal(t, []string{"name"}, result.Required)
	})

	t.Run("array type with items", func(t *testing.T) {
		data := map[string]interface{}{
			"type": "array",
			"items": map[string]interface{}{
				"type": "string",
			},
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		assert.Equal(t, "ARRAY", string(result.Type))
		require.NotNil(t, result.Items)
		assert.Equal(t, "STRING", string(result.Items.Type))
	})

	t.Run("nested objects", func(t *testing.T) {
		data := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"address": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{
							"type": "string",
						},
					},
				},
			},
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		require.Contains(t, result.Properties, "address")
		addressSchema := result.Properties["address"]
		assert.Equal(t, "OBJECT", string(addressSchema.Type))
		require.Contains(t, addressSchema.Properties, "city")
		assert.Equal(t, "STRING", string(addressSchema.Properties["city"].Type))
	})

	t.Run("enum values", func(t *testing.T) {
		data := map[string]interface{}{
			"type": "string",
			"enum": []interface{}{"red", "green", "blue"},
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		assert.Equal(t, []string{"red", "green", "blue"}, result.Enum)
	})

	t.Run("numeric constraints", func(t *testing.T) {
		data := map[string]interface{}{
			"type":    "number",
			"minimum": float64(0),
			"maximum": float64(100),
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		require.NotNil(t, result.Minimum)
		assert.Equal(t, float64(0), *result.Minimum)
		require.NotNil(t, result.Maximum)
		assert.Equal(t, float64(100), *result.Maximum)
	})

	t.Run("title and format", func(t *testing.T) {
		data := map[string]interface{}{
			"type":   "string",
			"title":  "Email Address",
			"format": "email",
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		assert.Equal(t, "Email Address", result.Title)
		assert.Equal(t, "email", result.Format)
	})

	t.Run("anyOf schemas", func(t *testing.T) {
		data := map[string]interface{}{
			"anyOf": []interface{}{
				map[string]interface{}{"type": "string"},
				map[string]interface{}{"type": "integer"},
			},
		}

		result := convertMapToGenaiSchema(data)
		require.NotNil(t, result)
		require.Len(t, result.AnyOf, 2)
		assert.Equal(t, "STRING", string(result.AnyOf[0].Type))
		assert.Equal(t, "INTEGER", string(result.AnyOf[1].Type))
	})
}

func TestConvertOpenAIParamsToGenaiSchema(t *testing.T) {
	t.Run("nil params returns nil", func(t *testing.T) {
		result := convertOpenAIParamsToGenaiSchema(nil)
		assert.Nil(t, result)
	})

	t.Run("valid params with properties", func(t *testing.T) {
		params := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query",
				},
			},
			"required": []interface{}{"query"},
		}

		result := convertOpenAIParamsToGenaiSchema(params)
		require.NotNil(t, result)
		assert.Equal(t, "OBJECT", string(result.Type))
		assert.Contains(t, result.Properties, "query")
		assert.Equal(t, []string{"query"}, result.Required)
	})
}

func TestConvertOpenAIResponseFormatToGenaiSchema(t *testing.T) {
	t.Run("nil returns nil", func(t *testing.T) {
		result := convertOpenAIResponseFormatToGenaiSchema(nil)
		assert.Nil(t, result)
	})

	t.Run("json_object type returns nil", func(t *testing.T) {
		rf := map[string]interface{}{
			"type": "json_object",
		}
		result := convertOpenAIResponseFormatToGenaiSchema(rf)
		assert.Nil(t, result)
	})

	t.Run("text type returns nil", func(t *testing.T) {
		rf := map[string]interface{}{
			"type": "text",
		}
		result := convertOpenAIResponseFormatToGenaiSchema(rf)
		assert.Nil(t, result)
	})

	t.Run("json_schema with schema", func(t *testing.T) {
		rf := map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name": "MyResponse",
				"schema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"result": map[string]interface{}{
							"type": "string",
						},
					},
				},
			},
		}

		result := convertOpenAIResponseFormatToGenaiSchema(rf)
		require.NotNil(t, result)
		assert.Equal(t, "MyResponse", result.Title)
		assert.Equal(t, "OBJECT", string(result.Type))
		assert.Contains(t, result.Properties, "result")
	})

	t.Run("json_schema without nested schema falls back to json_schema map", func(t *testing.T) {
		rf := map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"type": "object",
			},
		}

		result := convertOpenAIResponseFormatToGenaiSchema(rf)
		require.NotNil(t, result)
		assert.Equal(t, "OBJECT", string(result.Type))
	})

	t.Run("non-map type returns nil", func(t *testing.T) {
		result := convertOpenAIResponseFormatToGenaiSchema("invalid")
		assert.Nil(t, result)
	})
}
