package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-2.5-flash")
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-2.5-flash")
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-3-flash-preview")
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-3-flash-preview")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	// Contents: [0] user, [1] model, [2] tool result (call_2), [3] tool result (call_1)
	if len(vertexReq.Contents) < 4 {
		t.Fatalf("expected at least 4 contents, got %d", len(vertexReq.Contents))
	}

	// First tool result should resolve to "get_time" (call_2)
	fr1 := vertexReq.Contents[2].Parts[0].FunctionResponse
	if fr1 == nil || fr1.Name != "get_time" {
		t.Fatalf("expected first tool result Name = %q, got %q", "get_time", fr1.Name)
	}

	// Second tool result should resolve to "get_weather" (call_1)
	fr2 := vertexReq.Contents[3].Parts[0].FunctionResponse
	if fr2 == nil || fr2.Name != "get_weather" {
		t.Fatalf("expected second tool result Name = %q, got %q", "get_weather", fr2.Name)
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-2.5-flash")
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("OpenAIToVertex error: %v", err)
	}

	var vertexReq VertexRequest
	if err := json.Unmarshal(resultBytes, &vertexReq); err != nil {
		t.Fatalf("unmarshal vertex request: %v", err)
	}

	if vertexReq.SystemInstruction == nil {
		t.Fatalf("expected SystemInstruction for developer role, got nil")
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-2.5-flash")
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

	resultBytes, err := OpenAIToVertex(body, false, "gemini-2.5-flash")
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
	}

	if output, ok := fr.Response["output"]; !ok || output != "" {
		t.Fatalf("expected response = {output: ''}, got %v", fr.Response)
	}
}
