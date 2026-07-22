package vertex

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// OpenAIToVertex converts OpenAI format request to Vertex AI format.
func OpenAIToVertex(openAIBody []byte, isImageGeneration bool, isImageEdit bool, model, contentType string) ([]byte, error) {
	var req openai.OpenAIRequest

	if isImageGeneration || isImageEdit {
		if strings.Contains(model, "gemini") {
			var (
				chatBody []byte
				err      error
			)
			if isImageEdit {
				chatBody, err = imageEditRequestToOpenAIChatRequest(openAIBody, contentType, model)
			} else {
				chatBody, err = imageRequestToOpenAIChatRequest(openAIBody, model)
			}
			if err != nil {
				return nil, err
			}
			openAIBody = chatBody
		} else {
			if isImageEdit {
				return nil, fmt.Errorf("image edits are not supported for model %s", model)
			}
			return OpenAIImageToVertex(openAIBody)
		}
	}

	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	vertexReq := VertexRequest{
		Contents: make([]*genai.Content, 0),
	}
	var pendingToolParts []*genai.Part

	flushPendingToolParts := func() {
		if len(pendingToolParts) == 0 {
			return
		}
		vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
			Role:  "user",
			Parts: pendingToolParts,
		})
		pendingToolParts = nil
	}

	// Generation config
	vertexReq.GenerationConfig = buildGenerationConfig(&req, model)

	// Messages → Contents + SystemInstruction
	for _, msg := range req.Messages {
		if msg.Role != "tool" && msg.Role != "function" {
			flushPendingToolParts()
		}
		switch msg.Role {
		case "system", "developer":
			// concatenate multiple system messages instead of overwriting
			content := extractTextContent(msg.Content)
			if vertexReq.SystemInstruction != nil && len(vertexReq.SystemInstruction.Parts) > 0 {
				vertexReq.SystemInstruction.Parts[0].Text += "\n" + content
			} else {
				vertexReq.SystemInstruction = &genai.Content{
					Role:  "user",
					Parts: []*genai.Part{{Text: content}},
				}
			}
		case "tool", "function":
			// OpenAI tool result: {role: "tool", tool_call_id: "call_xyz", name: "func_name", content: "..."}
			// role: "function" is the deprecated OpenAI function-calling API; treat the same way.
			// Vertex expects all responses for a single model tool-call turn to be grouped
			// in one user content with multiple functionResponse parts.
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
			pendingToolParts = append(pendingToolParts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					Name:     funcName,
					Response: responseData,
				},
			})
		default:
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}
			var parts []*genai.Part
			// For assistant/model turns: prepend reasoning_content as a Thought=true Part.
			// Gemini 2.5/3 multi-turn conversations return reasoning_content in assistant
			// messages; sending it back as Thought=true preserves the reasoning context.
			// (Matches LiteLLM: assistant_msg.get("reasoning_content") → PartType(thought=True, text=...))
			if role == "model" && msg.ReasoningContent != "" {
				parts = append(parts, &genai.Part{
					Thought: true,
					Text:    msg.ReasoningContent,
				})
			}
			// Only add text parts if content is non-nil and non-empty.
			// Assistant messages with tool_calls often have content: null or content: ""
			// which would produce a "<nil>" text part via fmt.Sprintf.
			if msg.Content != nil {
				if s, ok := msg.Content.(string); !ok || s != "" {
					parts = append(parts, convertContentToParts(msg.Content)...)
				}
			}
			if len(msg.ToolCalls) > 0 && role == "model" {
				parts = append(parts, convertToolCallsToGenaiParts(msg.ToolCalls)...)
			}
			// Gemini requires at least one text Part in user-role content.
			// Multimodal messages with only images/audio/files will fail with
			// "Unable to submit request because it must have a text parameter".
			// Add a blank space Part as fallback (matches LiteLLM _check_text_in_content).
			// Note: check even when parts is empty — a media-only block that failed to
			// parse still needs a text anchor so the resulting content is not rejected.
			if role == "user" && !partsHaveText(parts) {
				parts = append(parts, &genai.Part{Text: " "})
			}
			if parts == nil {
				parts = []*genai.Part{} // avoid nil parts
			}
			vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
				Role:  role,
				Parts: parts,
			})
		}
	}
	flushPendingToolParts()

	// If all messages were system instructions and no user/assistant messages remain,
	// Vertex API will reject an empty Contents array. Add a minimal user turn as fallback.
	// (Matches LiteLLM _default_user_message_when_system_message_passed logic.)
	if len(vertexReq.Contents) == 0 && vertexReq.SystemInstruction != nil {
		vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{Text: "."}},
		})
	}

	// Tools
	var hasUserFunctions bool
	if len(req.Tools) > 0 {
		toolsResult := convertOpenAIToolsToVertex(req.Tools)
		if len(toolsResult.Tools) > 0 {
			vertexReq.Tools = toolsResult.Tools
		}
		hasUserFunctions = toolsResult.HasFunctionDecls && !toolsResult.HasBuiltinTools
	}

	// Tool choice — only set FunctionCallingConfig when function declarations are present.
	// Gemini rejects FunctionCallingConfig without function_declarations.
	if req.ToolChoice != nil && hasUserFunctions {
		vertexReq.ToolConfig = mapToolChoice(req.ToolChoice)
	}

	return json.Marshal(vertexReq)
}

// partsHaveText reports whether any Part in the slice contains a non-empty Text field.
// Used to detect user messages that consist only of media (images, audio, files) with no
// accompanying text — Gemini rejects such messages unless a text Part is present.
func partsHaveText(parts []*genai.Part) bool {
	for _, p := range parts {
		if p != nil && p.Text != "" {
			return true
		}
	}
	return false
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
