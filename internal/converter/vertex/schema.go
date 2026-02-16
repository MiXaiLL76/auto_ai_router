package vertex

import (
	"strings"

	"google.golang.org/genai"
)

// convertMapToGenaiSchema recursively converts a map[string]interface{} (JSON Schema) to *genai.Schema.
// This is used for response schemas to maintain structured format instead of raw JSON.
func convertMapToGenaiSchema(data map[string]interface{}) *genai.Schema {
	if data == nil {
		return nil
	}

	schema := &genai.Schema{}

	// Convert type field
	if typeVal, ok := data["type"].(string); ok {
		schema.Type = genai.Type(strings.ToUpper(typeVal))
	}

	// Convert title
	if title, ok := data["title"].(string); ok {
		schema.Title = title
	}

	// Convert description
	if desc, ok := data["description"].(string); ok {
		schema.Description = desc
	}

	// Convert enum
	if enumVals, ok := data["enum"].([]interface{}); ok {
		enumStrs := make([]string, 0, len(enumVals))
		for _, e := range enumVals {
			if str, ok := e.(string); ok {
				enumStrs = append(enumStrs, str)
			}
		}
		if len(enumStrs) > 0 {
			schema.Enum = enumStrs
		}
	}

	// Convert required array
	if required, ok := data["required"].([]interface{}); ok {
		requiredFields := make([]string, 0, len(required))
		for _, req := range required {
			if field, ok := req.(string); ok {
				requiredFields = append(requiredFields, field)
			}
		}
		if len(requiredFields) > 0 {
			schema.Required = requiredFields
		}
	}

	// Convert properties (recursive)
	if properties, ok := data["properties"].(map[string]interface{}); ok {
		schema.Properties = make(map[string]*genai.Schema)
		for propName, propVal := range properties {
			if propMap, ok := propVal.(map[string]interface{}); ok {
				schema.Properties[propName] = convertMapToGenaiSchema(propMap)
			}
		}
	}

	// Convert items for array types
	if items, ok := data["items"].(map[string]interface{}); ok {
		schema.Items = convertMapToGenaiSchema(items)
	}

	// Convert anyOf
	if anyOf, ok := data["anyOf"].([]interface{}); ok {
		schemas := make([]*genai.Schema, 0, len(anyOf))
		for _, item := range anyOf {
			if itemMap, ok := item.(map[string]interface{}); ok {
				schemas = append(schemas, convertMapToGenaiSchema(itemMap))
			}
		}
		if len(schemas) > 0 {
			schema.AnyOf = schemas
		}
	}

	// Convert format (e.g., "email", "date", etc.)
	if format, ok := data["format"].(string); ok {
		schema.Format = format
	}

	// Convert pattern (regex for string validation)
	if pattern, ok := data["pattern"].(string); ok {
		schema.Pattern = pattern
	}

	// Convert numeric constraints
	if minimum, ok := data["minimum"].(float64); ok {
		schema.Minimum = &minimum
	}
	if maximum, ok := data["maximum"].(float64); ok {
		schema.Maximum = &maximum
	}
	if minLength, ok := data["minLength"].(float64); ok {
		minLengthInt := int64(minLength)
		schema.MinLength = &minLengthInt
	}
	if maxLength, ok := data["maxLength"].(float64); ok {
		maxLengthInt := int64(maxLength)
		schema.MaxLength = &maxLengthInt
	}

	// Convert array constraints
	if minItems, ok := data["minItems"].(float64); ok {
		minItemsInt := int64(minItems)
		schema.MinItems = &minItemsInt
	}
	if maxItems, ok := data["maxItems"].(float64); ok {
		maxItemsInt := int64(maxItems)
		schema.MaxItems = &maxItemsInt
	}

	// Convert property ordering
	if propOrdering, ok := data["propertyOrdering"].([]interface{}); ok {
		propOrderingStrs := make([]string, 0, len(propOrdering))
		for _, prop := range propOrdering {
			if str, ok := prop.(string); ok {
				propOrderingStrs = append(propOrderingStrs, str)
			}
		}
		if len(propOrderingStrs) > 0 {
			schema.PropertyOrdering = propOrderingStrs
		}
	}

	// Convert default value
	if def, ok := data["default"]; ok {
		schema.Default = def
	}

	// Convert example
	if example, ok := data["example"]; ok {
		schema.Example = example
	}

	return schema
}

// convertOpenAIParamsToGenaiSchema converts OpenAI parameter schema to genai.Schema.
// Delegates to convertMapToGenaiSchema for full recursive conversion.
func convertOpenAIParamsToGenaiSchema(params map[string]interface{}) *genai.Schema {
	if params == nil {
		return nil
	}
	return convertMapToGenaiSchema(params)
}

// convertOpenAIResponseFormatToGenaiSchema converts OpenAI response_format to Vertex AI structured schema.
// Using ResponseSchema (structured) instead of ResponseJsonSchema (raw JSON) may produce
// different output formatting (compact vs pretty-printed JSON).
func convertOpenAIResponseFormatToGenaiSchema(responseFormat interface{}) *genai.Schema {
	// response_format can be:
	// 1. {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}
	// 2. {"type": "text"}
	// 3. nil

	if responseFormat == nil {
		return nil
	}

	switch rf := responseFormat.(type) {
	case map[string]interface{}:
		// Check if it's json_schema type
		if rfType, ok := rf["type"].(string); ok {
			switch rfType {
			case "json_schema":
				// Extract the json_schema field
				if jsonSchema, ok := rf["json_schema"].(map[string]interface{}); ok {
					if schema, ok := jsonSchema["schema"].(map[string]interface{}); ok {
						// Include schema name from OpenAI format if present
						if schemaName, ok := jsonSchema["name"].(string); ok && schemaName != "" {
							// Add title field to preserve the schema name for Vertex
							schema["title"] = schemaName
						}
						// Convert map to structured *genai.Schema
						return convertMapToGenaiSchema(schema)
					}
					// If no nested schema, convert the whole json_schema object
					return convertMapToGenaiSchema(jsonSchema)
				}
			case "json_object":
				// For simple json_object type, Vertex doesn't need additional schema
				return nil
			}
		}
	}

	return nil
}
