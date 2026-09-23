package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
		return
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
		return
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
		return
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
		return
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
		return
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

	result := convertOpenAIToolsToVertex(openAITools)

	assert.True(t, result.HasFunctionDecls, "should have function declarations")
	assert.False(t, result.HasBuiltinTools, "should not have builtin tools")

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool (grouped functions), got %d", len(result.Tools))
	}

	if len(result.Tools[0].FunctionDeclarations) != 2 {
		t.Fatalf("expected 2 function declarations, got %d", len(result.Tools[0].FunctionDeclarations))
	}
	if result.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Fatalf("expected first function = get_weather, got %q", result.Tools[0].FunctionDeclarations[0].Name)
	}
	if result.Tools[0].FunctionDeclarations[1].Name != "search_web" {
		t.Fatalf("expected second function = search_web, got %q", result.Tools[0].FunctionDeclarations[1].Name)
	}
}

// TestConvertOpenAIToolsToVertex_BuiltinToolsOnly verifies that when only
// built-in tools are present (no functions), only built-in tools are returned.
func TestConvertOpenAIToolsToVertex_BuiltinToolsOnly(t *testing.T) {
	openAITools := []interface{}{
		map[string]interface{}{"type": "web_search"},
	}

	result := convertOpenAIToolsToVertex(openAITools)

	assert.False(t, result.HasFunctionDecls, "should not have function declarations")
	assert.True(t, result.HasBuiltinTools, "should have builtin tools")
	require.Len(t, result.Tools, 1)
	assert.NotNil(t, result.Tools[0].GoogleSearch)
}

func TestConvertOpenAIToolsToVertex_URLContext(t *testing.T) {
	openAITools := []interface{}{
		map[string]interface{}{"type": "url_context"},
	}

	result := convertOpenAIToolsToVertex(openAITools)

	assert.False(t, result.HasFunctionDecls, "should not have function declarations")
	assert.True(t, result.HasBuiltinTools, "url_context is a builtin tool")
	require.Len(t, result.Tools, 1)
	assert.NotNil(t, result.Tools[0].URLContext)
}

// TestConvertOpenAIToolsToVertex_MixedToolsDropsFunctions verifies that when
// built-in tools and functions are mixed, functions are dropped (Gemini API limitation).
func TestConvertOpenAIToolsToVertex_MixedToolsDropsFunctions(t *testing.T) {
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

	result := convertOpenAIToolsToVertex(openAITools)

	assert.True(t, result.HasFunctionDecls, "should detect function declarations")
	assert.True(t, result.HasBuiltinTools, "should detect builtin tools")

	// Built-in tools take priority; functions are dropped
	require.Len(t, result.Tools, 2)

	hasSearch := false
	hasCode := false
	for _, tool := range result.Tools {
		if tool.GoogleSearch != nil {
			hasSearch = true
		}
		if tool.CodeExecution != nil {
			hasCode = true
		}
		assert.Nil(t, tool.FunctionDeclarations, "function declarations should be dropped when built-in tools present")
	}
	assert.True(t, hasSearch, "expected GoogleSearch tool")
	assert.True(t, hasCode, "expected CodeExecution tool")
}

// TestMapToolChoice verifies mapping of OpenAI tool_choice to Vertex ToolConfig.
func TestMapToolChoice(t *testing.T) {
	// nil returns nil
	if got := mapToolChoice(nil); got != nil {
		t.Fatalf("expected nil for nil input, got %+v", got)
	}

	// "none" -> NONE
	tc := mapToolChoice("none")
	if tc == nil {
		t.Fatalf("expected ToolConfig for 'none'")
		return
	}
	if tc.FunctionCallingConfig == nil {
		t.Fatalf("expected FunctionCallingConfig for 'none'")
		return
	}
	if tc.FunctionCallingConfig.Mode != "NONE" {
		t.Fatalf("expected mode NONE, got %q", tc.FunctionCallingConfig.Mode)
	}

	// "auto" -> AUTO
	tc = mapToolChoice("auto")
	if tc == nil {
		t.Fatalf("expected ToolConfig for 'auto'")
		return
	}
	if tc.FunctionCallingConfig == nil {
		t.Fatalf("expected FunctionCallingConfig for 'auto'")
		return
	}
	if tc.FunctionCallingConfig.Mode != "AUTO" {
		t.Fatalf("expected mode AUTO for 'auto', got %q", tc.FunctionCallingConfig.Mode)
	}

	// "required" -> ANY
	tc = mapToolChoice("required")
	if tc == nil {
		t.Fatalf("expected ToolConfig for 'required'")
		return
	}
	if tc.FunctionCallingConfig == nil {
		t.Fatalf("expected FunctionCallingConfig for 'required'")
		return
	}
	if tc.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("expected mode ANY for 'required', got %q", tc.FunctionCallingConfig.Mode)
	}

	// dict with function name -> ANY + allowedFunctionNames
	tc = mapToolChoice(map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": "specific_func",
		},
	})
	if tc == nil {
		t.Fatalf("expected ToolConfig for dict tool_choice")
		return
	}
	if tc.FunctionCallingConfig == nil {
		t.Fatalf("expected FunctionCallingConfig for dict tool_choice")
		return
	}
	if tc.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("expected mode ANY for dict tool_choice, got %q", tc.FunctionCallingConfig.Mode)
	}
	if len(tc.FunctionCallingConfig.AllowedFunctionNames) != 1 || tc.FunctionCallingConfig.AllowedFunctionNames[0] != "specific_func" {
		t.Fatalf("expected allowedFunctionNames = [specific_func], got %v", tc.FunctionCallingConfig.AllowedFunctionNames)
	}

	// unknown string returns nil
	if got := mapToolChoice("unknown_value"); got != nil {
		t.Fatalf("expected nil for unknown string, got %+v", got)
	}
}

// TestConvertGoogleSearchRetrieval verifies conversion of google_search_retrieval tool.
func TestConvertGoogleSearchRetrieval(t *testing.T) {
	tests := []struct {
		name    string
		toolMap map[string]interface{}
		hasDyn  bool
		thresh  *float32
	}{
		{
			name:    "nil dynamic_retrieval_config returns empty retrieval",
			toolMap: map[string]interface{}{"type": "google_search_retrieval"},
			hasDyn:  false,
		},
		{
			name: "with dynamic_retrieval_config and threshold",
			toolMap: map[string]interface{}{
				"type": "google_search_retrieval",
				"dynamic_retrieval_config": map[string]interface{}{
					"dynamic_threshold": float64(0.5),
				},
			},
			hasDyn: true,
			thresh: func() *float32 { v := float32(0.5); return &v }(),
		},
		{
			name: "with dynamic_retrieval_config without threshold",
			toolMap: map[string]interface{}{
				"type":                     "google_search_retrieval",
				"dynamic_retrieval_config": map[string]interface{}{},
			},
			hasDyn: true,
			thresh: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertGoogleSearchRetrieval(tt.toolMap)
			require.NotNil(t, result)

			if tt.hasDyn {
				require.NotNil(t, result.DynamicRetrievalConfig)
				if tt.thresh != nil {
					require.NotNil(t, result.DynamicRetrievalConfig.DynamicThreshold)
					assert.Equal(t, *tt.thresh, *result.DynamicRetrievalConfig.DynamicThreshold)
				} else {
					assert.Nil(t, result.DynamicRetrievalConfig.DynamicThreshold)
				}
			} else {
				assert.Nil(t, result.DynamicRetrievalConfig)
			}
		})
	}
}
