package vertex

import (
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// resolveRef resolves a $ref pointer within a root schema's $defs or definitions.
func resolveRef(ref string, rootSchema map[string]interface{}) map[string]interface{} {
	// Handle "#/$defs/Name" or "#/definitions/Name"
	parts := strings.Split(ref, "/")
	if len(parts) != 3 || parts[0] != "#" {
		return nil
	}
	section := parts[1] // "$defs" or "definitions"
	name := parts[2]

	if defs, ok := rootSchema[section].(map[string]interface{}); ok {
		if def, ok := defs[name].(map[string]interface{}); ok {
			return def
		}
	}
	return nil
}

// mergeAllOf merges multiple schemas from allOf into one combined schema.
func mergeAllOf(schemas []interface{}, rootSchema map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{})
	mergedProps := make(map[string]interface{})
	var mergedRequired []interface{}

	for _, item := range schemas {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		// Resolve $ref if present
		if ref, ok := itemMap["$ref"].(string); ok {
			if resolved := resolveRef(ref, rootSchema); resolved != nil {
				itemMap = resolved
			}
		}
		// Merge properties
		if props, ok := itemMap["properties"].(map[string]interface{}); ok {
			for k, v := range props {
				mergedProps[k] = v
			}
		}
		// Merge required
		if req, ok := itemMap["required"].([]interface{}); ok {
			mergedRequired = append(mergedRequired, req...)
		}
		// Copy type if present
		if t, ok := itemMap["type"]; ok {
			merged["type"] = t
		}
	}

	if len(mergedProps) > 0 {
		merged["properties"] = mergedProps
	}
	if len(mergedRequired) > 0 {
		merged["required"] = mergedRequired
	}
	return merged
}

// convertMapToGenaiSchemaWithRoot recursively converts a JSON Schema to *genai.Schema,
// with access to the root schema for $ref resolution.
func convertMapToGenaiSchemaWithRoot(data map[string]interface{}, rootSchema map[string]interface{}) *genai.Schema {
	if data == nil {
		return nil
	}

	// resolve $ref before processing
	if ref, ok := data["$ref"].(string); ok {
		if resolved := resolveRef(ref, rootSchema); resolved != nil {
			return convertMapToGenaiSchemaWithRoot(resolved, rootSchema)
		}
	}

	//  handle allOf by merging sub-schemas
	if allOf, ok := data["allOf"].([]interface{}); ok {
		merged := mergeAllOf(allOf, rootSchema)
		// Copy other fields from data that aren't allOf
		for k, v := range data {
			if k != "allOf" {
				if _, exists := merged[k]; !exists {
					merged[k] = v
				}
			}
		}
		return convertMapToGenaiSchemaWithRoot(merged, rootSchema)
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
			} else {
				enumStrs = append(enumStrs, fmt.Sprintf("%v", e))
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
				schema.Properties[propName] = convertMapToGenaiSchemaWithRoot(propMap, rootSchema)
			}
		}
	}

	// Convert items for array types
	if items, ok := data["items"].(map[string]interface{}); ok {
		schema.Items = convertMapToGenaiSchemaWithRoot(items, rootSchema)
	}

	// Convert anyOf
	if anyOf, ok := data["anyOf"].([]interface{}); ok {
		schemas := make([]*genai.Schema, 0, len(anyOf))
		for _, item := range anyOf {
			if itemMap, ok := item.(map[string]interface{}); ok {
				schemas = append(schemas, convertMapToGenaiSchemaWithRoot(itemMap, rootSchema))
			}
		}
		if len(schemas) > 0 {
			schema.AnyOf = schemas
		}
	}

	// handle oneOf (same as anyOf for Vertex)
	if oneOf, ok := data["oneOf"].([]interface{}); ok {
		schemas := make([]*genai.Schema, 0, len(oneOf))
		for _, item := range oneOf {
			if itemMap, ok := item.(map[string]interface{}); ok {
				schemas = append(schemas, convertMapToGenaiSchemaWithRoot(itemMap, rootSchema))
			}
		}
		if len(schemas) > 0 {
			// Vertex doesn't have OneOf, use AnyOf as closest equivalent
			if schema.AnyOf == nil {
				schema.AnyOf = schemas
			} else {
				schema.AnyOf = append(schema.AnyOf, schemas...)
			}
		}
	}

	// Convert format
	if format, ok := data["format"].(string); ok {
		schema.Format = format
	}

	// Convert pattern
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

// convertMapToGenaiSchema recursively converts a map[string]interface{} (JSON Schema) to *genai.Schema.
// This is used for response schemas to maintain structured format instead of raw JSON.
func convertMapToGenaiSchema(data map[string]interface{}) *genai.Schema {
	if data == nil {
		return nil
	}
	return convertMapToGenaiSchemaWithRoot(data, data) // use data itself as root for $ref resolution
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

	if rf, ok := responseFormat.(map[string]interface{}); ok {
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
