package vertex

import (
	"testing"
)

// TestConvertToolCallsToGenaiParts_NestedFormat verifies conversion of
// OpenAI tool_calls with nested structure: {"function": {"name": "fn", "arguments": "{}"}}.
func TestConvertToolCallsToGenaiParts_NestedFormat(t *testing.T) {
	toolCalls := []interface{}{
		map[string]interface{}{
			"id":   "call_001",
			"type": "function",
			"function": map[string]interface{}{
				"name":      "get_weather",
				"arguments": `{"city": "Tokyo"}`,
			},
		},
	}

	parts := convertToolCallsToGenaiParts(toolCalls)

	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fc := parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall, got nil")
	}
	if fc.Name != "get_weather" {
		t.Fatalf("expected name = %q, got %q", "get_weather", fc.Name)
	}
	if city, ok := fc.Args["city"]; !ok || city != "Tokyo" {
		t.Fatalf("expected args city = Tokyo, got %v", fc.Args)
	}
}

// TestConvertToolCallsToGenaiParts_FlatFormat verifies conversion of
// tool_calls with flat structure: {"name": "fn", "arguments": "{}"}.
func TestConvertToolCallsToGenaiParts_FlatFormat(t *testing.T) {
	toolCalls := []interface{}{
		map[string]interface{}{
			"name":      "calculate",
			"arguments": `{"x": 10, "y": 20}`,
		},
	}

	parts := convertToolCallsToGenaiParts(toolCalls)

	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fc := parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall, got nil")
	}
	if fc.Name != "calculate" {
		t.Fatalf("expected name = %q, got %q", "calculate", fc.Name)
	}
	if x, ok := fc.Args["x"]; !ok || x != float64(10) {
		t.Fatalf("expected args x = 10, got %v", fc.Args)
	}
	if y, ok := fc.Args["y"]; !ok || y != float64(20) {
		t.Fatalf("expected args y = 20, got %v", fc.Args)
	}
}

// TestConvertToolCallsToGenaiParts_FlatOverridesNested verifies that when both
// nested and flat name are present, the flat (top-level) name takes precedence.
func TestConvertToolCallsToGenaiParts_FlatOverridesNested(t *testing.T) {
	toolCalls := []interface{}{
		map[string]interface{}{
			"name":      "flat_name",
			"arguments": `{"a": 1}`,
			"function": map[string]interface{}{
				"name":      "nested_name",
				"arguments": `{"b": 2}`,
			},
		},
	}

	parts := convertToolCallsToGenaiParts(toolCalls)

	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fc := parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall, got nil")
	}
	// Flat name should override nested name
	if fc.Name != "flat_name" {
		t.Fatalf("expected name = %q (flat override), got %q", "flat_name", fc.Name)
	}
	// Flat arguments should override nested arguments
	if _, ok := fc.Args["a"]; !ok {
		t.Fatalf("expected flat args to override nested, got %v", fc.Args)
	}
}

// TestConvertToolCallsToGenaiParts_EmptyFuncName verifies that a tool call
// with no function name (empty in both nested and flat) is skipped entirely.
func TestConvertToolCallsToGenaiParts_EmptyFuncName(t *testing.T) {
	toolCalls := []interface{}{
		map[string]interface{}{
			"id":   "call_empty",
			"type": "function",
			"function": map[string]interface{}{
				"name":      "",
				"arguments": `{"x": 1}`,
			},
		},
	}

	parts := convertToolCallsToGenaiParts(toolCalls)

	if len(parts) != 0 {
		t.Fatalf("expected 0 parts for empty funcName, got %d", len(parts))
	}
}

// TestConvertToolCallsToGenaiParts_EmptySlice verifies nil return for empty input.
func TestConvertToolCallsToGenaiParts_EmptySlice(t *testing.T) {
	parts := convertToolCallsToGenaiParts(nil)
	if parts != nil {
		t.Fatalf("expected nil for nil input, got %v", parts)
	}

	parts = convertToolCallsToGenaiParts([]interface{}{})
	if parts != nil {
		t.Fatalf("expected nil for empty input, got %v", parts)
	}
}

// TestConvertToolCallsToGenaiParts_InvalidArguments verifies that malformed
// JSON arguments result in an error marker in args, not a crash.
func TestConvertToolCallsToGenaiParts_InvalidArguments(t *testing.T) {
	toolCalls := []interface{}{
		map[string]interface{}{
			"name":      "broken_fn",
			"arguments": `not valid json`,
		},
	}

	parts := convertToolCallsToGenaiParts(toolCalls)

	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fc := parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall, got nil")
	}
	if fc.Name != "broken_fn" {
		t.Fatalf("expected name = %q, got %q", "broken_fn", fc.Name)
	}
	// Should have error marker in args
	if errMsg, ok := fc.Args["_error"]; !ok || errMsg != "failed to parse arguments" {
		t.Fatalf("expected _error in args for invalid JSON, got %v", fc.Args)
	}
}

// TestConvertToolCallsToGenaiParts_ThoughtSignatureFallback verifies that when
// no provider_specific_fields are present, a dummy ThoughtSignature is set.
func TestConvertToolCallsToGenaiParts_ThoughtSignatureFallback(t *testing.T) {
	toolCalls := []interface{}{
		map[string]interface{}{
			"name":      "some_fn",
			"arguments": `{}`,
		},
	}

	parts := convertToolCallsToGenaiParts(toolCalls)

	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	// Should have fallback ThoughtSignature
	if parts[0].ThoughtSignature == nil {
		t.Fatalf("expected ThoughtSignature fallback, got nil")
	}
	if string(parts[0].ThoughtSignature) != "skip_thought_signature_validator" {
		t.Fatalf("expected dummy ThoughtSignature, got %q", string(parts[0].ThoughtSignature))
	}
}

// TestConvertOpenAIToolsToVertex_FunctionGrouping verifies that multiple function
// tools are grouped into a single Tool with multiple FunctionDeclarations.
func TestConvertOpenAIToolsToVertex_FunctionGrouping(t *testing.T) {
	openAITools := []interface{}{
		map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "get_weather",
				"description": "Get weather data",
			},
		},
		map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "search_web",
				"description": "Search the web",
			},
		},
	}

	tools := convertOpenAIToolsToVertex(openAITools)

	if len(tools) != 1 {
		t.Fatalf("expected 1 tool (grouped functions), got %d", len(tools))
	}

	if len(tools[0].FunctionDeclarations) != 2 {
		t.Fatalf("expected 2 function declarations, got %d", len(tools[0].FunctionDeclarations))
	}
	if tools[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Fatalf("expected first function = get_weather, got %q", tools[0].FunctionDeclarations[0].Name)
	}
	if tools[0].FunctionDeclarations[1].Name != "search_web" {
		t.Fatalf("expected second function = search_web, got %q", tools[0].FunctionDeclarations[1].Name)
	}
}

// TestConvertOpenAIToolsToVertex_SpecialToolsSeparate verifies that special tools
// (web_search, code_execution, etc.) each get their own Tool object.
func TestConvertOpenAIToolsToVertex_SpecialToolsSeparate(t *testing.T) {
	openAITools := []interface{}{
		map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "my_func",
			},
		},
		map[string]interface{}{
			"type": "web_search",
		},
		map[string]interface{}{
			"type": "code_execution",
		},
	}

	tools := convertOpenAIToolsToVertex(openAITools)

	// 1 grouped function tool + 1 web_search + 1 code_execution = 3
	if len(tools) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(tools))
	}

	// First tool should be the grouped functions
	if tools[0].FunctionDeclarations == nil || len(tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected first tool to be function declarations, got %+v", tools[0])
	}

	// One of the remaining should be GoogleSearch, one CodeExecution
	hasSearch := false
	hasCode := false
	for _, tool := range tools[1:] {
		if tool.GoogleSearch != nil {
			hasSearch = true
		}
		if tool.CodeExecution != nil {
			hasCode = true
		}
	}
	if !hasSearch {
		t.Fatalf("expected GoogleSearch tool")
	}
	if !hasCode {
		t.Fatalf("expected CodeExecution tool")
	}
}

// TestMapToolChoice verifies mapping of OpenAI tool_choice to Vertex ToolConfig.
func TestMapToolChoice(t *testing.T) {
	// nil returns nil
	if got := mapToolChoice(nil); got != nil {
		t.Fatalf("expected nil for nil input, got %+v", got)
	}

	// "none" -> NONE
	tc := mapToolChoice("none")
	if tc == nil || tc.FunctionCallingConfig == nil {
		t.Fatalf("expected ToolConfig for 'none'")
	}
	if tc.FunctionCallingConfig.Mode != "NONE" {
		t.Fatalf("expected mode NONE, got %q", tc.FunctionCallingConfig.Mode)
	}

	// "auto" -> AUTO
	tc = mapToolChoice("auto")
	if tc == nil || tc.FunctionCallingConfig.Mode != "AUTO" {
		t.Fatalf("expected mode AUTO for 'auto'")
	}

	// "required" -> ANY
	tc = mapToolChoice("required")
	if tc == nil || tc.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("expected mode ANY for 'required'")
	}

	// dict with function name -> ANY + allowedFunctionNames
	tc = mapToolChoice(map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": "specific_func",
		},
	})
	if tc == nil || tc.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("expected mode ANY for dict tool_choice")
	}
	if len(tc.FunctionCallingConfig.AllowedFunctionNames) != 1 || tc.FunctionCallingConfig.AllowedFunctionNames[0] != "specific_func" {
		t.Fatalf("expected allowedFunctionNames = [specific_func], got %v", tc.FunctionCallingConfig.AllowedFunctionNames)
	}

	// unknown string returns nil
	if got := mapToolChoice("unknown_value"); got != nil {
		t.Fatalf("expected nil for unknown string, got %+v", got)
	}
}
