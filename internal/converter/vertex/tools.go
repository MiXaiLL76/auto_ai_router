package vertex

import (
	"encoding/json"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"google.golang.org/genai"
)

// mapToolChoice maps OpenAI tool_choice to Vertex ToolConfig.
// OpenAI formats:
//   - "none"     → FunctionCallingMode NONE
//   - "required" → FunctionCallingMode ANY
//   - "auto"     → FunctionCallingMode AUTO
//   - {function: {name: "fn"}} → FunctionCallingMode ANY + allowedFunctionNames
func mapToolChoice(toolChoice interface{}) *genai.ToolConfig {
	if toolChoice == nil {
		return nil
	}

	fcc := &genai.FunctionCallingConfig{}

	switch choice := toolChoice.(type) {
	case string:
		switch choice {
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
		// {"type": "function", "function": {"name": "func_name"}}
		if funcObj, ok := choice["function"].(map[string]interface{}); ok {
			if name, ok := funcObj["name"].(string); ok && name != "" {
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

// convertOpenAIToolsToVertex converts OpenAI tools to genai.Tool slice.
// Functions are grouped in one Tool; special tools each get their own Tool object.
func convertOpenAIToolsToVertex(openAITools []interface{}) []*genai.Tool {
	if len(openAITools) == 0 {
		return nil
	}

	var tools []*genai.Tool
	var functionDecls []*genai.FunctionDeclaration

	for _, toolInterface := range openAITools {
		toolMap, ok := toolInterface.(map[string]interface{})
		if !ok {
			continue
		}
		toolType, _ := toolMap["type"].(string)

		switch toolType {
		case "function":
			if funcObj, ok := toolMap["function"].(map[string]interface{}); ok {
				if decl := convertToFunctionDecl(funcObj); decl != nil {
					functionDecls = append(functionDecls, decl)
				}
			}
		case "computer_use":
			tools = append(tools, &genai.Tool{
				ComputerUse: &genai.ComputerUse{},
			})
		case "web_search", "web_search_preview":
			tools = append(tools, &genai.Tool{
				GoogleSearch: &genai.GoogleSearch{},
			})
		case "google_search_retrieval":
			retrieval := convertGoogleSearchRetrieval(toolMap)
			tools = append(tools, &genai.Tool{
				GoogleSearchRetrieval: retrieval,
			})
		case "google_maps":
			tools = append(tools, &genai.Tool{
				GoogleMaps: &genai.GoogleMaps{},
			})
		case "code_execution":
			tools = append(tools, &genai.Tool{
				CodeExecution: &genai.ToolCodeExecution{},
			})
		}
	}

	// Functions go first, grouped in a single Tool
	if len(functionDecls) > 0 {
		tools = append([]*genai.Tool{{FunctionDeclarations: functionDecls}}, tools...)
	}

	if len(tools) == 0 {
		return nil
	}
	return tools
}

// convertToFunctionDecl converts OpenAI function definition to genai.FunctionDeclaration
func convertToFunctionDecl(funcObj map[string]interface{}) *genai.FunctionDeclaration {
	name := converterutil.GetString(funcObj, "name")
	if name == "" {
		return nil
	}
	decl := &genai.FunctionDeclaration{
		Name:        name,
		Description: converterutil.GetString(funcObj, "description"),
	}
	if params, ok := funcObj["parameters"].(map[string]interface{}); ok {
		decl.Parameters = convertOpenAIParamsToGenaiSchema(params)
	}
	return decl
}

// convertGoogleSearchRetrieval converts OpenAI google_search_retrieval tool to genai.GoogleSearchRetrieval
func convertGoogleSearchRetrieval(toolMap map[string]interface{}) *genai.GoogleSearchRetrieval {
	retrieval := &genai.GoogleSearchRetrieval{}
	// Extract dynamic retrieval config if present
	if config, ok := toolMap["dynamic_retrieval_config"].(map[string]interface{}); ok {
		dynConfig := &genai.DynamicRetrievalConfig{}
		if threshold, ok := config["dynamic_threshold"].(float64); ok {
			t := float32(threshold)
			dynConfig.DynamicThreshold = &t
		}
		retrieval.DynamicRetrievalConfig = dynConfig
	}
	return retrieval
}

// convertToolCallsToGenaiParts converts OpenAI tool_calls to genai.Part with FunctionCall.
// Restores thoughtSignature from provider_specific_fields for Gemini 3 multi-turn conversations.
func convertToolCallsToGenaiParts(toolCalls []interface{}) []*genai.Part {
	if len(toolCalls) == 0 {
		return nil
	}

	var parts []*genai.Part

	for _, toolCallInterface := range toolCalls {
		toolCallMap, ok := toolCallInterface.(map[string]interface{})
		if !ok {
			continue
		}

		// Extract function information from either flat or nested structure
		var funcName string
		var argsStr string

		if funcObj, ok := toolCallMap["function"].(map[string]interface{}); ok {
			// Nested structure: {"function": {"name": "...", "arguments": "..."}}
			funcName = converterutil.GetString(funcObj, "name")
			argsStr = converterutil.GetString(funcObj, "arguments")
		}

		// Flat structure: {"name": "...", "arguments": "..."} — overrides if present
		if topName := converterutil.GetString(toolCallMap, "name"); topName != "" {
			funcName = topName
			if topArgs := converterutil.GetString(toolCallMap, "arguments"); topArgs != "" {
				argsStr = topArgs
			}
		}

		if funcName == "" {
			continue
		}

		// Parse arguments
		var args map[string]interface{}
		if argsStr != "" {
			if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
				args = map[string]interface{}{"_error": "failed to parse arguments"}
			}
		}

		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: funcName,
				Args: args,
			},
		}

		// Restore thoughtSignature from provider_specific_fields if present.
		// Required for Gemini 3 models to maintain context across multi-turn conversations.
		foundThoughtSignature := false
		if providerFields, ok := toolCallMap["provider_specific_fields"].(map[string]interface{}); ok {
			if thoughtSigStr, ok := providerFields["thought_signature"].(string); ok && thoughtSigStr != "" {
				// Decode from base64 back to binary
				if decoded := converterutil.DecodeBase64(thoughtSigStr); decoded != nil {
					part.ThoughtSignature = decoded
					foundThoughtSignature = true
				}
			}
		}

		// Fallback: If no thoughtSignature provided, add dummy value.
		// Per litellm and Google docs, clients (like LangChain) may not preserve provider_specific_fields.
		// The dummy validator allows Gemini 3 to accept the request without validation errors.
		if !foundThoughtSignature {
			part.ThoughtSignature = []byte("skip_thought_signature_validator")
		}

		parts = append(parts, part)
	}

	return parts
}
