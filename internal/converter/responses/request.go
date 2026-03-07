package responses

import (
	"encoding/json"
	"fmt"
)

// IsResponsesAPI checks if the body is a Responses API request.
// Returns true if body has "input" field and does NOT have "messages" field.
func IsResponsesAPI(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return false
	}
	_, hasInput := raw["input"]
	_, hasMessages := raw["messages"]
	return hasInput && !hasMessages
}

// RequestToChat converts a Responses API request body to Chat Completions format.
// Returns the converted body ready for orchestrateRequest + provider converters.
func RequestToChat(body []byte) ([]byte, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	messages, err := convertInput(raw)
	if err != nil {
		return nil, fmt.Errorf("failed to convert input: %w", err)
	}

	// Prepend system message from instructions
	if instructions, ok := raw["instructions"].(string); ok && instructions != "" {
		systemMsg := map[string]interface{}{
			"role":    "system",
			"content": instructions,
		}
		messages = append([]interface{}{systemMsg}, messages...)
	}

	// Set messages
	raw["messages"] = messages

	// max_output_tokens -> max_completion_tokens
	if maxOut, ok := raw["max_output_tokens"]; ok {
		raw["max_completion_tokens"] = maxOut
	}

	// Convert tools from flat to nested format
	convertTools(raw)

	// Convert tool_choice
	convertToolChoice(raw)

	// reasoning.effort -> reasoning_effort
	convertReasoning(raw)

	// text.format -> response_format
	convertTextFormat(raw)

	// Remove Responses-API-only fields
	deleteResponsesFields(raw)

	result, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal converted request: %w", err)
	}
	return result, nil
}

// convertInput converts the "input" field to Chat Completions "messages".
func convertInput(raw map[string]interface{}) ([]interface{}, error) {
	input, ok := raw["input"]
	if !ok {
		return nil, fmt.Errorf("missing input field")
	}

	// String input -> single user message
	if inputStr, ok := input.(string); ok {
		return []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": inputStr,
			},
		}, nil
	}

	// Array input -> iterate items
	inputArr, ok := input.([]interface{})
	if !ok {
		return nil, fmt.Errorf("input must be string or array")
	}

	var messages []interface{}
	// pendingToolCalls accumulates consecutive function_call items
	// to merge them into a single assistant message with multiple tool_calls.
	var pendingToolCalls []interface{}

	flushToolCalls := func() {
		if len(pendingToolCalls) == 0 {
			return
		}
		messages = append(messages, map[string]interface{}{
			"role":       "assistant",
			"content":    nil,
			"tool_calls": pendingToolCalls,
		})
		pendingToolCalls = nil
	}

	for _, item := range inputArr {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		itemType, _ := itemMap["type"].(string)

		switch itemType {
		case "function_call":
			// Accumulate tool calls; they'll be flushed as a single assistant message
			callID, _ := itemMap["call_id"].(string)
			name, _ := itemMap["name"].(string)
			arguments, _ := itemMap["arguments"].(string)
			pendingToolCalls = append(pendingToolCalls, map[string]interface{}{
				"id":   callID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": arguments,
				},
			})

		case "function_call_output":
			// Flush any pending tool calls before the output
			flushToolCalls()
			msg := convertFunctionCallOutput(itemMap)
			messages = append(messages, msg)

		default:
			// Flush any pending tool calls before a regular message
			flushToolCalls()
			msg := convertMessage(itemMap)
			messages = append(messages, msg)
		}
	}

	// Flush any remaining tool calls
	flushToolCalls()

	return messages, nil
}

// convertMessage converts an InputMessage to a Chat Completions message.
func convertMessage(item map[string]interface{}) map[string]interface{} {
	msg := map[string]interface{}{
		"role": item["role"],
	}

	content := item["content"]
	switch c := content.(type) {
	case string:
		msg["content"] = c
	case []interface{}:
		msg["content"] = convertContentParts(c)
	default:
		msg["content"] = content
	}

	return msg
}

// convertContentParts converts Responses API content parts to Chat Completions format.
func convertContentParts(parts []interface{}) []interface{} {
	var result []interface{}
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)
		switch partType {
		case "input_text":
			result = append(result, map[string]interface{}{
				"type": "text",
				"text": partMap["text"],
			})

		case "input_image":
			imgURL := ""
			if u, ok := partMap["image_url"].(string); ok {
				imgURL = u
			}
			if imgURL == "" {
				// file_id and other unsupported sources — skip with no silent corruption
				continue
			}
			entry := map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": imgURL,
				},
			}
			if detail, ok := partMap["detail"].(string); ok && detail != "" {
				entry["image_url"].(map[string]interface{})["detail"] = detail
			}
			result = append(result, entry)

		case "input_audio":
			entry := map[string]interface{}{
				"type": "input_audio",
				"input_audio": map[string]interface{}{
					"data":   partMap["data"],
					"format": partMap["format"],
				},
			}
			result = append(result, entry)

		default:
			// Pass through unknown types as-is
			result = append(result, part)
		}
	}
	return result
}

// convertFunctionCallOutput converts a function_call_output input item to a tool message.
func convertFunctionCallOutput(item map[string]interface{}) map[string]interface{} {
	callID, _ := item["call_id"].(string)
	output, _ := item["output"].(string)

	return map[string]interface{}{
		"role":         "tool",
		"tool_call_id": callID,
		"content":      output,
	}
}

// convertTools converts Responses API flat tools to Chat Completions nested format.
func convertTools(raw map[string]interface{}) {
	toolsRaw, ok := raw["tools"]
	if !ok {
		return
	}
	toolsArr, ok := toolsRaw.([]interface{})
	if !ok {
		return
	}

	var converted []interface{}
	for _, t := range toolsArr {
		toolMap, ok := t.(map[string]interface{})
		if !ok {
			continue
		}

		toolType, _ := toolMap["type"].(string)
		if toolType != "function" {
			// Skip non-function tools (web_search, etc.)
			continue
		}

		// Flat: {type: "function", name: "x", description: "y", parameters: {...}, strict: bool}
		// Nested: {type: "function", function: {name: "x", description: "y", parameters: {...}, strict: bool}}
		funcDef := map[string]interface{}{}
		if name, ok := toolMap["name"]; ok {
			funcDef["name"] = name
		}
		if desc, ok := toolMap["description"]; ok {
			funcDef["description"] = desc
		}
		if params, ok := toolMap["parameters"]; ok {
			funcDef["parameters"] = params
		}
		if strict, ok := toolMap["strict"]; ok {
			funcDef["strict"] = strict
		}

		converted = append(converted, map[string]interface{}{
			"type":     "function",
			"function": funcDef,
		})
	}

	if len(converted) > 0 {
		raw["tools"] = converted
	} else {
		delete(raw, "tools")
	}
}

// convertToolChoice converts Responses API tool_choice to Chat Completions format.
func convertToolChoice(raw map[string]interface{}) {
	tc, ok := raw["tool_choice"]
	if !ok {
		return
	}

	tcMap, ok := tc.(map[string]interface{})
	if !ok {
		// string values like "auto", "none", "required" pass through unchanged
		return
	}

	tcType, _ := tcMap["type"].(string)
	if tcType == "function" {
		name, _ := tcMap["name"].(string)
		raw["tool_choice"] = map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": name,
			},
		}
	}
	// Other types pass through unchanged
}

// convertReasoning extracts reasoning.effort and sets it as top-level reasoning_effort.
func convertReasoning(raw map[string]interface{}) {
	reasoning, ok := raw["reasoning"]
	if !ok {
		return
	}
	reasoningMap, ok := reasoning.(map[string]interface{})
	if !ok {
		return
	}
	if effort, ok := reasoningMap["effort"].(string); ok && effort != "" {
		raw["reasoning_effort"] = effort
	}
}

// convertTextFormat converts text.format to response_format.
// Responses API json_schema: {type: "json_schema", name: "...", schema: {...}, strict: bool}
// Chat Completions:          {type: "json_schema", json_schema: {name: "...", schema: {...}, strict: bool}}
func convertTextFormat(raw map[string]interface{}) {
	text, ok := raw["text"]
	if !ok {
		return
	}
	textMap, ok := text.(map[string]interface{})
	if !ok {
		return
	}
	format, ok := textMap["format"]
	if !ok {
		return
	}
	formatMap, ok := format.(map[string]interface{})
	if !ok {
		raw["response_format"] = format
		return
	}

	formatType, _ := formatMap["type"].(string)
	if formatType == "json_schema" {
		// Wrap Responses API flat format into Chat Completions nested format
		jsonSchema := map[string]interface{}{}
		for k, v := range formatMap {
			if k != "type" {
				jsonSchema[k] = v
			}
		}
		raw["response_format"] = map[string]interface{}{
			"type":        "json_schema",
			"json_schema": jsonSchema,
		}
	} else {
		// "text", "json_object" — pass through as-is
		raw["response_format"] = format
	}
}

// deleteResponsesFields removes Responses-API-only fields from the request.
func deleteResponsesFields(raw map[string]interface{}) {
	delete(raw, "input")
	delete(raw, "instructions")
	delete(raw, "max_output_tokens")
	delete(raw, "previous_response_id")
	delete(raw, "store")
	delete(raw, "reasoning")
	delete(raw, "text")
}
