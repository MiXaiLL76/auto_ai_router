package vertex

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// OpenAIToVertex converts OpenAI format request to Vertex AI format.
func OpenAIToVertex(openAIBody []byte, isImageGeneration bool, model string) ([]byte, error) {
	var req openai.OpenAIRequest

	if isImageGeneration {
		if strings.Contains(model, "gemini") {
			chatBody, err := ImageRequestToOpenAIChatRequest(openAIBody)
			if err != nil {
				return nil, err
			}
			openAIBody = chatBody
		} else {
			return OpenAIImageToVertex(openAIBody)
		}
	}

	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	vertexReq := VertexRequest{
		Contents: make([]*genai.Content, 0),
	}

	// Generation config
	vertexReq.GenerationConfig = buildGenerationConfig(&req, model)

	// Messages → Contents + SystemInstruction
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "developer":
			content := extractTextContent(msg.Content)
			vertexReq.SystemInstruction = &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: content}},
			}
		case "tool":
			// OpenAI tool result: {role: "tool", tool_call_id: "call_xyz", name: "func_name", content: "..."}
			// Vertex expects: Part.FunctionResponse{Name: funcName, Response: {output: content}}
			funcName := msg.Name
			if funcName == "" && msg.ToolCallID != "" {
				// Look up function name from preceding assistant message's tool_calls
				funcName = findFunctionNameByToolCallID(req.Messages, msg.ToolCallID)
			}
			if funcName == "" {
				funcName = "tool_result" // last resort default if Name not set
			}
			content := extractTextContent(msg.Content)
			// Build response map: try to parse JSON first, fallback to string
			var responseData map[string]interface{}
			if content != "" {
				// Try to unmarshal as JSON object
				if err := json.Unmarshal([]byte(content), &responseData); err != nil {
					// Not JSON or not object - wrap as string output
					responseData = map[string]interface{}{"output": content}
				}
			} else {
				responseData = map[string]interface{}{"output": ""}
			}
			vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     funcName,
						Response: responseData,
					},
				}},
			})
		default:
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}
			parts := convertContentToParts(msg.Content)
			if len(msg.ToolCalls) > 0 && role == "model" {
				parts = append(parts, convertToolCallsToGenaiParts(msg.ToolCalls)...)
			}
			vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
				Role:  role,
				Parts: parts,
			})
		}
	}

	// Tools
	if len(req.Tools) > 0 {
		if tools := convertOpenAIToolsToVertex(req.Tools); len(tools) > 0 {
			vertexReq.Tools = tools
		}
	}

	// Tool choice (Phase 1)
	if req.ToolChoice != nil {
		vertexReq.ToolConfig = mapToolChoice(req.ToolChoice)
	}

	return json.Marshal(vertexReq)
}

// findFunctionNameByToolCallID searches assistant messages' tool_calls for a matching
// tool_call_id and returns the function name. This is needed because many OpenAI clients
// (including Google's own OpenAI-compatible endpoint) don't include the "name" field
// in tool result messages. Without the correct name, Vertex API receives "tool_result"
// as FunctionResponse.Name which doesn't match any FunctionDeclaration, causing the model
// to waste reasoning tokens trying to figure out the context.
func findFunctionNameByToolCallID(messages []openai.OpenAIMessage, toolCallID string) string {
	for _, msg := range messages {
		if msg.Role != "assistant" || len(msg.ToolCalls) == 0 {
			continue
		}
		for _, tc := range msg.ToolCalls {
			tcMap, ok := tc.(map[string]interface{})
			if !ok {
				continue
			}
			id, _ := tcMap["id"].(string)
			if id != toolCallID {
				continue
			}
			// Extract function name from nested or flat structure
			if funcObj, ok := tcMap["function"].(map[string]interface{}); ok {
				if name, ok := funcObj["name"].(string); ok && name != "" {
					return name
				}
			}
			if name, ok := tcMap["name"].(string); ok && name != "" {
				return name
			}
		}
	}
	return ""
}
