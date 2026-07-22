package vertexresponses

import (
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"google.golang.org/genai"
)

// responsesToolsToVertex converts Responses API tools to Vertex AI genai.Tool slice.
// Function tools are grouped into one Tool with FunctionDeclarations.
// Built-in tools (web_search, code_interpreter) become separate Tool entries.
func responsesToolsToVertex(tools []responses.Tool) []*genai.Tool {
	if len(tools) == 0 {
		return nil
	}

	var funcDecls []*genai.FunctionDeclaration
	var builtinTools []*genai.Tool

	for _, t := range tools {
		switch t.Type {
		case "function":
			decl := &genai.FunctionDeclaration{
				Name:        t.Name,
				Description: t.Description,
			}
			if t.Parameters != nil {
				if schema := interfaceToSchema(t.Parameters); schema != nil {
					decl.Parameters = schema
				}
			}
			funcDecls = append(funcDecls, decl)

		case "web_search_preview", "web_search_preview_2025_03_11", "web_search":
			builtinTools = append(builtinTools, &genai.Tool{
				GoogleSearch: &genai.GoogleSearch{},
			})

		case "code_interpreter":
			builtinTools = append(builtinTools, &genai.Tool{
				CodeExecution: &genai.ToolCodeExecution{},
			})

			// Other tool types (file_search, computer_use, mcp) are not supported by
			// Vertex and are silently skipped. image_generation is intentionally not
			// emitted as a tool either: buildGenConfig translates it to the IMAGE
			// response modality.
		}
	}

	// Gemini API does NOT allow combining built-in tools with function declarations.
	// When both are present, prefer built-in tools and silently drop functions.
	if len(builtinTools) > 0 {
		return builtinTools
	}
	if len(funcDecls) > 0 {
		return []*genai.Tool{{FunctionDeclarations: funcDecls}}
	}
	return nil
}

// responsesToolChoiceToVertex maps Responses API tool_choice to Vertex ToolConfig.
func responsesToolChoiceToVertex(toolChoice interface{}) *genai.ToolConfig {
	if toolChoice == nil {
		return nil
	}
	fcc := &genai.FunctionCallingConfig{}
	switch tc := toolChoice.(type) {
	case string:
		switch tc {
		case "none":
			fcc.Mode = genai.FunctionCallingConfigModeNone
		case "required":
			fcc.Mode = genai.FunctionCallingConfigModeAny
		case "auto":
			fcc.Mode = genai.FunctionCallingConfigModeAuto
		default:
			return nil
		}
	case map[string]interface{}:
		tcType, _ := tc["type"].(string)
		if tcType == "function" {
			name, _ := tc["name"].(string)
			if name != "" {
				fcc.Mode = genai.FunctionCallingConfigModeAny
				fcc.AllowedFunctionNames = []string{name}
			}
		}
		if fcc.Mode == "" {
			return nil
		}
	default:
		return nil
	}
	return &genai.ToolConfig{FunctionCallingConfig: fcc}
}

// interfaceToSchema converts an interface{} (JSON-compatible) to *genai.Schema.
func interfaceToSchema(params interface{}) *genai.Schema {
	paramsMap, ok := params.(map[string]interface{})
	if !ok {
		return nil
	}

	schema := &genai.Schema{}
	if typeName, ok := paramsMap["type"].(string); ok {
		schema.Type = schemaTypeFromString(typeName)
	}
	if description, ok := paramsMap["description"].(string); ok {
		schema.Description = description
	}
	if format, ok := paramsMap["format"].(string); ok {
		schema.Format = format
	}
	if title, ok := paramsMap["title"].(string); ok {
		schema.Title = title
	}
	if enumValues, ok := paramsMap["enum"].([]interface{}); ok {
		schema.Enum = make([]string, 0, len(enumValues))
		for _, value := range enumValues {
			if s, ok := value.(string); ok {
				schema.Enum = append(schema.Enum, s)
			}
		}
	}
	if requiredValues, ok := paramsMap["required"].([]interface{}); ok {
		schema.Required = make([]string, 0, len(requiredValues))
		for _, value := range requiredValues {
			if s, ok := value.(string); ok {
				schema.Required = append(schema.Required, s)
			}
		}
	}
	if items, ok := paramsMap["items"]; ok {
		schema.Items = interfaceToSchema(items)
	}
	if properties, ok := paramsMap["properties"].(map[string]interface{}); ok {
		schema.Properties = make(map[string]*genai.Schema, len(properties))
		for name, value := range properties {
			if propertySchema := interfaceToSchema(value); propertySchema != nil {
				schema.Properties[name] = propertySchema
			}
		}
	}
	if anyOf, ok := paramsMap["anyOf"].([]interface{}); ok {
		schema.AnyOf = make([]*genai.Schema, 0, len(anyOf))
		for _, value := range anyOf {
			if anyOfSchema := interfaceToSchema(value); anyOfSchema != nil {
				schema.AnyOf = append(schema.AnyOf, anyOfSchema)
			}
		}
	}

	return schema
}

func schemaTypeFromString(typeName string) genai.Type {
	switch strings.ToUpper(typeName) {
	case "ARRAY":
		return genai.TypeArray
	case "BOOLEAN":
		return genai.TypeBoolean
	case "INTEGER":
		return genai.TypeInteger
	case "NUMBER":
		return genai.TypeNumber
	case "OBJECT":
		return genai.TypeObject
	case "STRING":
		return genai.TypeString
	default:
		return genai.Type("")
	}
}
