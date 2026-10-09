package vertex

import (
	"encoding/json"
	"fmt"
	"strings"

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

// vertexToolsResult holds the converted tools and metadata about what was found.
type vertexToolsResult struct {
	Tools            []*genai.Tool
	HasFunctionDecls bool // true when at least one function declaration is present
	HasBuiltinTools  bool // true when GoogleSearch, CodeExecution, etc. are present
}

// convertOpenAIToolsToVertex converts OpenAI tools to genai.Tool slice.
// Functions are grouped in one Tool; special tools each get their own Tool object,
// except search tools, which merge into one GoogleSearch tool.
// Gemini API does NOT allow combining built-in tools (GoogleSearch, CodeExecution, etc.)
// with function declarations in the same request. When both are present, built-in tools
// take priority and function declarations are dropped.
func convertOpenAIToolsToVertex(openAITools []interface{}) (vertexToolsResult, error) {
	result := vertexToolsResult{}
	if len(openAITools) == 0 {
		return result, nil
	}

	var builtinTools []*genai.Tool
	var functionDecls []*genai.FunctionDeclaration
	var search GoogleSearchTypes

	for i, toolInterface := range openAITools {
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
			builtinTools = append(builtinTools, &genai.Tool{
				ComputerUse: &genai.ComputerUse{},
			})
		case "web_search", "web_search_preview", "google_search":
			tool, err := search.Add(toolMap["search_types"], fmt.Sprintf("tools[%d].search_types", i))
			if err != nil {
				return vertexToolsResult{}, err
			}
			if tool != nil {
				builtinTools = append(builtinTools, tool)
			}
		case "google_search_retrieval":
			retrieval := convertGoogleSearchRetrieval(toolMap)
			builtinTools = append(builtinTools, &genai.Tool{
				GoogleSearchRetrieval: retrieval,
			})
		case "google_maps":
			builtinTools = append(builtinTools, &genai.Tool{
				GoogleMaps: &genai.GoogleMaps{},
			})
		case "code_execution":
			builtinTools = append(builtinTools, &genai.Tool{
				CodeExecution: &genai.ToolCodeExecution{},
			})
		case "url_context":
			builtinTools = append(builtinTools, &genai.Tool{
				URLContext: &genai.URLContext{},
			})
		}
	}

	result.HasBuiltinTools = len(builtinTools) > 0
	result.HasFunctionDecls = len(functionDecls) > 0

	// Gemini API constraint: built-in tools and function calling cannot be combined.
	// When both are present, prefer built-in tools (GoogleSearch, etc.) and drop functions.
	if result.HasBuiltinTools {
		result.Tools = builtinTools
	} else if result.HasFunctionDecls {
		result.Tools = []*genai.Tool{{FunctionDeclarations: functionDecls}}
	}

	return result, nil
}

// GoogleSearchTypes merges a request's search tools into one GoogleSearch tool
// enabling the union of their search types. Exported for vertex/responses.
type GoogleSearchTypes struct {
	tool       *genai.Tool
	web, image bool
}

// Add reads one tool's search_types: ["web_search", "image_search"] or Gemini's
// {"webSearch": {}, "imageSearch": {}}; absent means web search. The first call
// returns the tool to add to the request, later ones widen it and return nil.
// Web-only search is sent without searchTypes, as for models that predate them.
func (s *GoogleSearchTypes) Add(searchTypes interface{}, param string) (*genai.Tool, error) {
	web, image, err := parseGoogleSearchTypes(searchTypes, param)
	if err != nil {
		return nil, err
	}
	s.web, s.image = s.web || web, s.image || image
	var added *genai.Tool
	if s.tool == nil {
		s.tool = &genai.Tool{GoogleSearch: &genai.GoogleSearch{}}
		added = s.tool
	}
	s.tool.GoogleSearch.SearchTypes = nil
	if s.image {
		s.tool.GoogleSearch.SearchTypes = &genai.SearchTypes{ImageSearch: &genai.ImageSearch{}}
		if s.web {
			s.tool.GoogleSearch.SearchTypes.WebSearch = &genai.WebSearch{}
		}
	}
	return added, nil
}

// parseGoogleSearchTypes reads search_types; an unknown type is an error, not a
// silent web search.
func parseGoogleSearchTypes(raw interface{}, param string) (web, image bool, err error) {
	enable := func(name string) error {
		switch strings.ToLower(name) {
		case "web_search", "websearch":
			web = true
		case "image_search", "imagesearch":
			image = true
		default:
			return unsupportedSearchTypeError(param, name)
		}
		return nil
	}
	switch value := raw.(type) {
	case nil:
	case []interface{}:
		for _, item := range value {
			name, ok := item.(string)
			if !ok {
				return false, false, converterutil.NewInvalidTypeError(param)
			}
			if err := enable(name); err != nil {
				return false, false, err
			}
		}
	case map[string]interface{}:
		for name, config := range value {
			switch config := config.(type) {
			case nil:
				// {"imageSearch": null}: listed, not enabled.
				continue
			case bool:
				if !config {
					continue
				}
			case map[string]interface{}:
			default:
				return false, false, converterutil.NewInvalidTypeError(param + "." + name)
			}
			if err := enable(name); err != nil {
				return false, false, err
			}
		}
	default:
		return false, false, converterutil.NewInvalidTypeError(param)
	}
	if !web && !image {
		web = true
	}
	return web, image, nil
}

func unsupportedSearchTypeError(param, value string) error {
	return &converterutil.RequestValidationError{
		Param:   param,
		Code:    "invalid_value",
		Message: fmt.Sprintf("Unsupported search type %q; supported values: web_search, image_search", value),
	}
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
