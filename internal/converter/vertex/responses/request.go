package vertexresponses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"google.golang.org/genai"
)

// ResponsesRequestToVertex converts a Responses API request body to the Vertex AI
// GenerateContent request format (as JSON).
func ResponsesRequestToVertex(body []byte, model string) ([]byte, error) {
	var req responses.Request
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("ResponsesRequestToVertex: parse request: %w", err)
	}

	vertexReq, err := buildVertexRequest(&req, model)
	if err != nil {
		return nil, fmt.Errorf("ResponsesRequestToVertex: build request: %w", err)
	}

	result, err := json.Marshal(vertexReq)
	if err != nil {
		return nil, fmt.Errorf("ResponsesRequestToVertex: marshal: %w", err)
	}
	return result, nil
}

func buildVertexRequest(req *responses.Request, model string) (*vertex.VertexRequest, error) {
	vr := &vertex.VertexRequest{
		Contents: make([]*genai.Content, 0),
	}

	// System instruction from instructions field.
	if inst := responses.ExtractInstructionsText(req.Instructions); inst != "" {
		vr.SystemInstruction = &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{Text: inst}},
		}
	}

	// Convert input items to contents.
	contents, err := inputToContents(req.Input)
	if err != nil {
		return nil, err
	}
	vr.Contents = contents

	// Tools.
	if len(req.Tools) > 0 {
		vr.Tools = responsesToolsToVertex(req.Tools)
	}

	// Tool choice — only set FunctionCallingConfig when function tools are present AND
	// no built-in tools (built-ins take priority and function decls are silently dropped).
	if req.ToolChoice != nil {
		hasBuiltinTools := false
		hasFunctionTools := false
		for _, t := range req.Tools {
			switch t.Type {
			case "function":
				hasFunctionTools = true
			case "web_search_preview", "web_search_preview_2025_03_11", "web_search", "code_interpreter":
				hasBuiltinTools = true
			}
		}
		if hasFunctionTools && !hasBuiltinTools {
			vr.ToolConfig = responsesToolChoiceToVertex(req.ToolChoice)
		}
	}

	// Generation config.
	if generationConfig := buildGenConfig(req, model); generationConfig != nil {
		vr.GenerationConfig = &vertex.VertexGenerationConfig{GenerationConfig: generationConfig}
	}

	return vr, nil
}

// inputToContents converts the Responses API input (string | []InputItem) to genai.Content slice.
func inputToContents(input interface{}) ([]*genai.Content, error) {
	if input == nil {
		return nil, nil
	}

	// Plain string input → single user content.
	if s, ok := input.(string); ok {
		return []*genai.Content{{
			Role:  "user",
			Parts: []*genai.Part{{Text: s}},
		}}, nil
	}

	// Unmarshal through JSON to get []interface{} reliably.
	raw, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("inputToContents: marshal input: %w", err)
	}
	var items []interface{}
	if err := json.Unmarshal(raw, &items); err != nil {
		// Single object — wrap in array.
		var single interface{}
		if err2 := json.Unmarshal(raw, &single); err2 != nil {
			return nil, fmt.Errorf("inputToContents: parse input: %w", err)
		}
		items = []interface{}{single}
	}

	var contents []*genai.Content
	var pendingToolParts []*genai.Part // function responses waiting to be flushed as a user content

	flushToolParts := func() {
		if len(pendingToolParts) == 0 {
			return
		}
		contents = append(contents, &genai.Content{
			Role:  "user",
			Parts: pendingToolParts,
		})
		pendingToolParts = nil
	}

	for _, item := range items {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		itemType, _ := itemMap["type"].(string)
		role, _ := itemMap["role"].(string)

		// Normalize: items without explicit type but with a role are messages.
		if itemType == "" && role != "" {
			itemType = "message"
		}

		switch itemType {
		case "message", "":
			flushToolParts()
			content, err := messageItemToContent(itemMap, role)
			if err != nil {
				return nil, err
			}
			if content != nil {
				contents = append(contents, content)
			}

		case "function_call":
			// Model-generated function call → model content with FunctionCall part.
			flushToolParts()
			callID, _ := itemMap["call_id"].(string)
			name, _ := itemMap["name"].(string)
			arguments, _ := itemMap["arguments"].(string)
			var argsMap map[string]interface{}
			_ = json.Unmarshal([]byte(arguments), &argsMap)
			contents = append(contents, &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: name,
						Args: argsMap,
					},
				}},
			})

		case "function_call_output":
			// Tool result → accumulate as FunctionResponse parts (flushed as user content).
			callID, _ := itemMap["call_id"].(string)
			name, _ := itemMap["name"].(string)
			if name == "" {
				name = callID
			}
			var responseData map[string]interface{}
			switch o := itemMap["output"].(type) {
			case string:
				responseData = map[string]interface{}{"output": o}
			case map[string]interface{}:
				responseData = o
			default:
				responseData = map[string]interface{}{"output": fmt.Sprintf("%v", itemMap["output"])}
			}
			pendingToolParts = append(pendingToolParts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       callID,
					Name:     name,
					Response: responseData,
				},
			})

		case "reasoning":
			// Reasoning items carry model summary text; include as model content.
			flushToolParts()
			if summary, ok := itemMap["summary"].([]interface{}); ok {
				var text strings.Builder
				for _, s := range summary {
					if sm, ok := s.(map[string]interface{}); ok {
						if t, ok := sm["text"].(string); ok {
							text.WriteString(t)
						}
					}
				}
				if text.Len() > 0 {
					contents = append(contents, &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{{Text: "[Reasoning]: " + text.String()}},
					})
				}
			}

		default:
			// Unknown item types are skipped to avoid corrupting the conversation.
			flushToolParts()
		}
	}

	flushToolParts()
	return contents, nil
}

// messageItemToContent converts an input message item to a genai.Content.
func messageItemToContent(itemMap map[string]interface{}, role string) (*genai.Content, error) {
	vertexRole := "user"
	switch role {
	case "assistant", "model":
		vertexRole = "model"
	}

	rawContent := itemMap["content"]

	var parts []*genai.Part
	switch c := rawContent.(type) {
	case string:
		parts = []*genai.Part{{Text: c}}

	case []interface{}:
		for _, part := range c {
			pm, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			converted, err := contentPartToVertexParts(pm)
			if err != nil {
				return nil, err
			}
			parts = append(parts, converted...)
		}

	default:
		parts = []*genai.Part{{Text: ""}}
	}

	if len(parts) == 0 {
		return nil, nil
	}

	return &genai.Content{Role: vertexRole, Parts: parts}, nil
}

// buildGenConfig constructs Vertex GenerationConfig from a Responses API request.
func buildGenConfig(req *responses.Request, model string) *genai.GenerationConfig {
	cfg := &genai.GenerationConfig{}
	hasParams := false

	// OpenAI exposes image generation as a built-in Responses tool. Vertex does
	// not have an equivalent tool object; Gemini image models select generated
	// image output through generationConfig.responseModalities instead.
	for _, tool := range req.Tools {
		if tool.Type == "image_generation" {
			cfg.ResponseModalities = []genai.Modality{genai.Modality("IMAGE")}
			hasParams = true
			break
		}
	}

	if req.MaxOutputTokens != nil {
		cfg.MaxOutputTokens = int32(*req.MaxOutputTokens)
		hasParams = true
	}
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
		hasParams = true
	}
	if req.TopP != nil {
		v := float32(*req.TopP)
		cfg.TopP = &v
		hasParams = true
	}

	// Stop sequences.
	if req.Stop != nil {
		switch s := req.Stop.(type) {
		case string:
			cfg.StopSequences = []string{s}
			hasParams = true
		case []interface{}:
			for _, sv := range s {
				if str, ok := sv.(string); ok {
					cfg.StopSequences = append(cfg.StopSequences, str)
				}
			}
			if len(cfg.StopSequences) > 0 {
				hasParams = true
			}
		}
	}

	// Text format → response MIME type and schema.
	if req.Text != nil {
		if formatMap, ok := req.Text.Format.(map[string]interface{}); ok {
			fmtType, _ := formatMap["type"].(string)
			switch fmtType {
			case "json_object":
				cfg.ResponseMIMEType = "application/json"
				hasParams = true
			case "json_schema":
				cfg.ResponseMIMEType = "application/json"
				schemaSource := formatMap["schema"]
				if schemaSource == nil {
					schemaSource = formatMap
				}
				if schema := interfaceToSchema(schemaSource); schema != nil {
					cfg.ResponseSchema = schema
				}
				hasParams = true
			}
		}
	}

	// Reasoning → ThinkingConfig.
	if req.Reasoning != nil && req.Reasoning.Effort != "" && req.Reasoning.Effort != "none" {
		cfg.ThinkingConfig = vertex.MapReasoningEffortToThinkingConfig(req.Reasoning.Effort, model)
		hasParams = true
	} else if def := vertex.DefaultThinkingConfig(model); def != nil {
		cfg.ThinkingConfig = def
		hasParams = true
	}

	if !hasParams {
		return nil
	}
	return cfg
}
