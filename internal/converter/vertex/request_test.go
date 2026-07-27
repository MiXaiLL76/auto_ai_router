package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/require"
)

// TestOpenAIToVertex_ToolRoleMessage_UsesNameField verifies that when a tool-role
// message has Name set (e.g. "get_weather"), the resulting FunctionResponse.Name
// uses that Name, NOT the ToolCallID.
func TestOpenAIToVertex_ToolRoleMessage_UsesNameField(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "What is the weather?"},
			{
				Role:       "tool",
				Name:       "get_weather",
				ToolCallID: "call_abc123",
				Content:    `{"temperature": 22}`,
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Contents[0] = user message, Contents[1] = tool result
	if len(vertexReq.Contents) < 2 {
		t.Fatalf("expected at least 2 contents, got %d", len(vertexReq.Contents))
	}

	toolContent := vertexReq.Contents[1]
	if len(toolContent.Parts) != 1 {
		t.Fatalf("expected 1 part in tool content, got %d", len(toolContent.Parts))
	}

	fr := toolContent.Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("expected FunctionResponse, got nil")
		return
	}

	// The critical check: Name must be "get_weather" (from msg.Name), not "call_abc123" (from msg.ToolCallID)
	if fr.Name != "get_weather" {
		t.Fatalf("expected FunctionResponse.Name = %q, got %q", "get_weather", fr.Name)
	}

	// Verify JSON content was parsed into response map
	if temp, ok := fr.Response["temperature"]; !ok {
		t.Fatalf("expected 'temperature' key in response map")
	} else if temp != float64(22) {
		t.Fatalf("expected temperature = 22, got %v", temp)
	}
}

func TestOpenAIToVertex_TopLevelImageConfig(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-3.1-flash-image",
		Messages: []openai.OpenAIMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "edit"},
				},
			},
		},
		ImageConfigSnake: map[string]interface{}{
			"aspect_ratio": "3:4",
			"image_size":   "2K",
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-3.1-flash-image", "application/json")
	require.NoError(t, err)

	var vertexReq VertexRequest
	require.NoError(t, json.Unmarshal(resultBytes, &vertexReq))
	require.NotNil(t, vertexReq.GenerationConfig)
	require.NotNil(t, vertexReq.GenerationConfig.ImageConfig)
	require.Equal(t, "3:4", vertexReq.GenerationConfig.ImageConfig.AspectRatio)
	require.Equal(t, "2K", vertexReq.GenerationConfig.ImageConfig.ImageSize)
}

// TestOpenAIToVertex_ToolRoleMessage_EmptyName_FallbackToToolResult verifies
// that when a tool-role message has an empty Name field, the FunctionResponse.Name
// falls back to "tool_result".
func TestOpenAIToVertex_ToolRoleMessage_EmptyName_FallbackToToolResult(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Do something"},
			{
				Role:       "tool",
				Name:       "", // empty name
				ToolCallID: "call_xyz789",
				Content:    "some result",
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	if len(vertexReq.Contents) < 2 {
		t.Fatalf("expected at least 2 contents, got %d", len(vertexReq.Contents))
	}

	toolContent := vertexReq.Contents[1]
	fr := toolContent.Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("expected FunctionResponse, got nil")
		return
	}

	if fr.Name != "tool_result" {
		t.Fatalf("expected FunctionResponse.Name = %q (fallback), got %q", "tool_result", fr.Name)
	}

	// Non-JSON content should be wrapped as {"output": "some result"}
	if output, ok := fr.Response["output"]; !ok || output != "some result" {
		t.Fatalf("expected response output = %q, got %v", "some result", fr.Response)
	}
}

// TestOpenAIToVertex_ToolRoleMessage_EmptyName_ResolvesFromToolCalls verifies that
// when a tool-role message has no Name but has a ToolCallID matching a preceding
// assistant message's tool_calls, the function name is resolved from the tool_calls.
// This is the most common real-world case: clients like Google's OpenAI-compatible
// endpoint don't include "name" in tool result messages.
func TestOpenAIToVertex_ToolRoleMessage_EmptyName_ResolvesFromToolCalls(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-3-flash-preview",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Search for something"},
			{
				Role: "assistant",
				ToolCalls: []interface{}{
					map[string]interface{}{
						"id":   "call_abc123",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "multisearch",
							"arguments": `{"query": "test"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				Name:       "", // empty — should be resolved from tool_calls above
				ToolCallID: "call_abc123",
				Content:    `{"results": []}`,
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-3-flash-preview", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Contents: [0] user, [1] assistant (model), [2] tool result
	if len(vertexReq.Contents) < 3 {
		t.Fatalf("expected at least 3 contents, got %d", len(vertexReq.Contents))
	}

	toolContent := vertexReq.Contents[2]
	fr := toolContent.Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("expected FunctionResponse, got nil")
		return
	}

	// Must be "multisearch" (resolved from tool_calls), NOT "tool_result"
	if fr.Name != "multisearch" {
		t.Fatalf("expected FunctionResponse.Name = %q (resolved from tool_calls), got %q", "multisearch", fr.Name)
	}
}

// TestOpenAIToVertex_ToolRoleMessage_MultipleToolCalls_ResolvesCorrectName verifies
// that with multiple tool_calls, each tool result resolves to the correct function name.
func TestOpenAIToVertex_ToolRoleMessage_MultipleToolCalls_ResolvesCorrectName(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-3-flash-preview",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Do two things"},
			{
				Role: "assistant",
				ToolCalls: []interface{}{
					map[string]interface{}{
						"id":   "call_1",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city": "Moscow"}`,
						},
					},
					map[string]interface{}{
						"id":   "call_2",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_time",
							"arguments": `{"timezone": "MSK"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				ToolCallID: "call_2", // second tool call
				Content:    `{"time": "15:00"}`,
			},
			{
				Role:       "tool",
				ToolCallID: "call_1", // first tool call (order doesn't matter)
				Content:    `{"temp": 20}`,
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-3-flash-preview", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Contents: [0] user, [1] model, [2] grouped tool results
	if len(vertexReq.Contents) < 3 {
		t.Fatalf("expected at least 3 contents, got %d", len(vertexReq.Contents))
	}

	toolResponses := vertexReq.Contents[2]
	if len(toolResponses.Parts) != 2 {
		t.Fatalf("expected 2 grouped tool response parts, got %d", len(toolResponses.Parts))
	}

	// First tool result should resolve to "get_time" (call_2)
	fr1 := toolResponses.Parts[0].FunctionResponse
	if fr1 == nil || fr1.Name != "get_time" {
		t.Fatalf("expected first tool result Name = %q, got %q", "get_time", fr1.Name)
	}

	// Second tool result should resolve to "get_weather" (call_1)
	fr2 := toolResponses.Parts[1].FunctionResponse
	if fr2 == nil || fr2.Name != "get_weather" {
		t.Fatalf("expected second tool result Name = %q, got %q", "get_weather", fr2.Name)
	}
}

// TestOpenAIToVertex_ToolRoleMessage_MultipleToolCalls_AreGroupedIntoSingleTurn verifies
// that consecutive tool messages are converted into a single user content with multiple
// functionResponse parts, matching Gemini's requirement for multi-tool response turns.
func TestOpenAIToVertex_ToolRoleMessage_MultipleToolCalls_AreGroupedIntoSingleTurn(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-3-pro-preview",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Get weather for NY and MSC"},
			{
				Role: "assistant",
				ToolCalls: []interface{}{
					map[string]interface{}{
						"id":   "call_ny",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city":"NY"}`,
						},
					},
					map[string]interface{}{
						"id":   "call_msc",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city":"MSC"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				ToolCallID: "call_ny",
				Content:    "17",
			},
			{
				Role:       "tool",
				ToolCallID: "call_msc",
				Content:    "15.3",
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-3-pro-preview", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Contents: [0] user, [1] model with 2 function calls, [2] user with 2 function responses
	if len(vertexReq.Contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(vertexReq.Contents))
	}

	toolResponses := vertexReq.Contents[2]
	if toolResponses.Role != "user" {
		t.Fatalf("expected grouped tool responses role = %q, got %q", "user", toolResponses.Role)
	}
	if len(toolResponses.Parts) != 2 {
		t.Fatalf("expected 2 grouped function response parts, got %d", len(toolResponses.Parts))
	}

	fr1 := toolResponses.Parts[0].FunctionResponse
	if fr1 == nil || fr1.Name != "get_weather" {
		t.Fatalf("expected first grouped function response name = %q, got %#v", "get_weather", fr1)
	}
	if output, ok := fr1.Response["output"]; !ok || output != "17" {
		t.Fatalf("expected first grouped response output = %q, got %v", "17", fr1.Response)
	}

	fr2 := toolResponses.Parts[1].FunctionResponse
	if fr2 == nil || fr2.Name != "get_weather" {
		t.Fatalf("expected second grouped function response name = %q, got %#v", "get_weather", fr2)
	}
	if output, ok := fr2.Response["output"]; !ok || output != "15.3" {
		t.Fatalf("expected second grouped response output = %q, got %v", "15.3", fr2.Response)
	}
}

// TestOpenAIToVertex_BasicMessageConversion verifies that user, assistant, and
// system messages are correctly converted to Vertex AI format.
func TestOpenAIToVertex_BasicMessageConversion(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
			{Role: "user", Content: "How are you?"},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// System message should go to SystemInstruction, not Contents
	if vertexReq.SystemInstruction == nil {
		t.Fatalf("expected SystemInstruction, got nil")
		return
	}
	if len(vertexReq.SystemInstruction.Parts) != 1 || vertexReq.SystemInstruction.Parts[0].Text != "You are helpful." {
		t.Fatalf("unexpected SystemInstruction: %+v", vertexReq.SystemInstruction)
	}

	// Remaining 3 messages (user, assistant, user) should be in Contents
	if len(vertexReq.Contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(vertexReq.Contents))
	}

	// First content: user "Hello"
	if vertexReq.Contents[0].Role != "user" {
		t.Fatalf("expected role 'user', got %q", vertexReq.Contents[0].Role)
	}
	if vertexReq.Contents[0].Parts[0].Text != "Hello" {
		t.Fatalf("expected text 'Hello', got %q", vertexReq.Contents[0].Parts[0].Text)
	}

	// Second content: assistant -> model
	if vertexReq.Contents[1].Role != "model" {
		t.Fatalf("expected role 'model' (mapped from assistant), got %q", vertexReq.Contents[1].Role)
	}
	if vertexReq.Contents[1].Parts[0].Text != "Hi there!" {
		t.Fatalf("expected text 'Hi there!', got %q", vertexReq.Contents[1].Parts[0].Text)
	}

	// Third content: user "How are you?"
	if vertexReq.Contents[2].Role != "user" {
		t.Fatalf("expected role 'user', got %q", vertexReq.Contents[2].Role)
	}
	if vertexReq.Contents[2].Parts[0].Text != "How are you?" {
		t.Fatalf("expected text 'How are you?', got %q", vertexReq.Contents[2].Parts[0].Text)
	}
}

// TestOpenAIToVertex_DeveloperRole verifies that "developer" role is treated
// the same as "system" (mapped to SystemInstruction).
func TestOpenAIToVertex_DeveloperRole(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "developer", Content: "Be concise."},
			{Role: "user", Content: "Hi"},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	if vertexReq.SystemInstruction == nil {
		t.Fatalf("expected SystemInstruction for developer role, got nil")
		return
	}
	if vertexReq.SystemInstruction.Parts[0].Text != "Be concise." {
		t.Fatalf("unexpected SystemInstruction text: %q", vertexReq.SystemInstruction.Parts[0].Text)
	}
}

// TestOpenAIToVertex_ToolRoleMessage_JSONContent verifies that JSON content
// in a tool result is parsed as a map, not wrapped in {"output": ...}.
func TestOpenAIToVertex_ToolRoleMessage_JSONContent(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Check weather"},
			{
				Role:    "tool",
				Name:    "get_weather",
				Content: `{"city": "Moscow", "temp": 15}`,
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	fr := vertexReq.Contents[1].Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("expected FunctionResponse, got nil")
		return
	}

	// JSON content should be parsed directly, not wrapped
	if city, ok := fr.Response["city"]; !ok || city != "Moscow" {
		t.Fatalf("expected city = Moscow in response, got %v", fr.Response)
	}
	if temp, ok := fr.Response["temp"]; !ok || temp != float64(15) {
		t.Fatalf("expected temp = 15 in response, got %v", fr.Response)
	}
}

// TestOpenAIToVertex_ToolRoleMessage_EmptyContent verifies that empty tool
// result content produces {"output": ""}.
func TestOpenAIToVertex_ToolRoleMessage_EmptyContent(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Do it"},
			{
				Role:    "tool",
				Name:    "delete_item",
				Content: "",
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	fr := vertexReq.Contents[1].Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("expected FunctionResponse, got nil")
		return
	}

	if output, ok := fr.Response["output"]; !ok || output != "" {
		t.Fatalf("expected response = {output: ''}, got %v", fr.Response)
	}
}

// TestOpenAIToVertex_ReasoningContent verifies that assistant messages with
// reasoning_content are converted to a Thought=true Part prepended before the
// regular text Part. This is required for Gemini 2.5/3 multi-turn reasoning.
func TestOpenAIToVertex_ReasoningContent(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Solve 2+2"},
			{
				Role:             "assistant",
				Content:          "The answer is 4.",
				ReasoningContent: "Let me think: 2+2=4",
			},
			{Role: "user", Content: "And 3+3?"},
		},
	}

	body, _ := json.Marshal(req)
	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Contents: [0]=user, [1]=model(thought+text), [2]=user
	if len(vertexReq.Contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(vertexReq.Contents))
	}
	modelContent := vertexReq.Contents[1]
	if modelContent.Role != "model" {
		t.Fatalf("expected role model, got %s", modelContent.Role)
	}
	if len(modelContent.Parts) < 2 {
		t.Fatalf("expected at least 2 parts (thought + text), got %d", len(modelContent.Parts))
	}
	// First part must be Thought=true
	if !modelContent.Parts[0].Thought {
		t.Fatalf("first part must have Thought=true (reasoning content)")
	}
	if modelContent.Parts[0].Text != "Let me think: 2+2=4" {
		t.Fatalf("thought text mismatch: %q", modelContent.Parts[0].Text)
	}
	// Second part is regular text
	if modelContent.Parts[1].Thought {
		t.Fatalf("second part must not have Thought=true")
	}
	if modelContent.Parts[1].Text != "The answer is 4." {
		t.Fatalf("text part mismatch: %q", modelContent.Parts[1].Text)
	}
}

// TestOpenAIToVertex_UserMediaOnlyBlankText verifies that a user message with only
// images (no text) gets a blank text Part appended. Gemini rejects user content
// without at least one text Part ("must have a text parameter" error).
// Uses a gs:// URI so no DNS resolution is attempted (avoids network in unit tests).
func TestOpenAIToVertex_UserMediaOnlyBlankText(t *testing.T) {
	content := []interface{}{
		map[string]interface{}{
			"type":      "image_url",
			"image_url": map[string]interface{}{"url": "gs://my-bucket/image.jpg"},
		},
	}
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: content},
		},
	}

	body, _ := json.Marshal(req)
	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	if len(vertexReq.Contents) == 0 {
		t.Fatal("expected at least one content block")
	}

	userParts := vertexReq.Contents[0].Parts
	hasText := false
	for _, p := range userParts {
		if p != nil && p.Text != "" {
			hasText = true
			break
		}
	}
	if !hasText {
		t.Fatalf("user content with only image should have a blank text Part appended, parts=%+v", userParts)
	}
}

// TestOpenAIToVertex_SystemOnlyFallback verifies that a request with only system
// messages generates a minimal user turn so Vertex doesn't reject empty Contents.
func TestOpenAIToVertex_SystemOnlyFallback(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "system", Content: "You are a helpful assistant."},
		},
	}

	body, _ := json.Marshal(req)
	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	if vertexReq.SystemInstruction == nil {
		t.Fatal("expected SystemInstruction to be set")
		return
	}
	if len(vertexReq.Contents) == 0 {
		t.Fatal("expected a fallback user content when only system messages present")
	}
	if vertexReq.Contents[0].Role != "user" {
		t.Fatalf("fallback content must have role=user, got %q", vertexReq.Contents[0].Role)
	}
}

// TestOpenAIToVertex_FunctionRole verifies that the deprecated role:"function"
// (old OpenAI function-calling API) is treated the same as role:"tool".
func TestOpenAIToVertex_FunctionRole(t *testing.T) {
	req := openai.OpenAIRequest{
		Model: "gemini-2.5-flash",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "What is the weather?"},
			{
				Role:    "function",
				Name:    "get_weather",
				Content: `{"temperature": 25}`,
			},
		},
	}

	body, _ := json.Marshal(req)
	resultBytes, err := OpenAIToVertex(body, false, false, "gemini-2.5-flash", "application/json")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Should produce: [0]=user, [1]=user(functionResponse) - tool parts flushed as user role
	found := false
	for _, c := range vertexReq.Contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "get_weather" {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("expected FunctionResponse with name get_weather from function-role message")
	}
}
