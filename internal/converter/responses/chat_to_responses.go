package responses

import (
	"encoding/json"
	"fmt"
)

// ChatRequestToResponses converts a Chat Completions request body into a
// Responses API request body -- the mirror of RequestToChat, used when a
// model is marked responses_only (config.ModelRPMConfig.ResponsesOnly): the
// client called /v1/chat/completions, but the upstream only accepts
// /v1/responses, so AIR must translate the request on the way in and
// translate the response back on the way out (see ResponseToChat and
// TransformResponsesStreamToChat).
func ChatRequestToResponses(body []byte) ([]byte, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	rawMessages, ok := raw["messages"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("messages is required")
	}

	input, err := chatMessagesToInput(rawMessages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}
	raw["input"] = input

	// max_completion_tokens (reasoning models) / max_tokens -> max_output_tokens,
	// the single universal Responses API parameter.
	if maxTokens, ok := raw["max_completion_tokens"]; ok {
		raw["max_output_tokens"] = maxTokens
	} else if maxTokens, ok := raw["max_tokens"]; ok {
		raw["max_output_tokens"] = maxTokens
	}

	if err := chatToolsToResponses(raw); err != nil {
		return nil, err
	}
	chatToolChoiceToResponses(raw)
	chatReasoningToResponses(raw)
	chatResponseFormatToText(raw)

	deleteChatOnlyFields(raw)

	result, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal converted request: %w", err)
	}
	return result, nil
}

// chatMessagesToInput converts Chat Completions "messages" to Responses API
// "input" items. A plain system/developer/user message with string or
// content-part-array content becomes one input message item. An assistant
// message carrying tool_calls is split into an optional message item for
// any text content plus one function_call item per tool call -- Responses
// API has no "tool_calls" field on a message, tool invocations are always
// separate input items. A "tool" role message (the client's tool result)
// becomes a function_call_output item keyed by tool_call_id -> call_id, the
// same correlation ID space ChatRequestToResponses/ResponseToChat share so a
// multi-turn tool-use conversation round-trips correctly.
func chatMessagesToInput(messages []interface{}) ([]interface{}, error) {
	input := make([]interface{}, 0, len(messages))
	for _, raw := range messages {
		msg, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid message")
		}
		role, _ := msg["role"].(string)
		switch role {
		case "system", "developer", "user":
			item, err := chatMessageToInputItem(role, msg["content"])
			if err != nil {
				return nil, err
			}
			if item != nil {
				input = append(input, item)
			}
		case "assistant":
			input = append(input, chatAssistantMessageToInputItems(msg)...)
		case "tool":
			callID, _ := msg["tool_call_id"].(string)
			input = append(input, map[string]interface{}{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  chatToolContentToString(msg["content"]),
			})
		default:
			return nil, fmt.Errorf("unsupported message role %q", role)
		}
	}
	return input, nil
}

// chatMessageToInputItem converts one system/developer/user message's
// content into a Responses API input message item.
func chatMessageToInputItem(role string, content interface{}) (map[string]interface{}, error) {
	switch c := content.(type) {
	case nil:
		return nil, nil
	case string:
		if c == "" {
			return nil, nil
		}
		return map[string]interface{}{"role": role, "content": c}, nil
	case []interface{}:
		parts, err := chatContentPartsToInput(c)
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"role": role, "content": parts}, nil
	default:
		return map[string]interface{}{"role": role, "content": content}, nil
	}
}

// chatContentPartsToInput converts Chat Completions message content parts
// (text / image_url / input_audio) to Responses API input content parts
// (input_text / input_image / input_audio). Mirrors convertContentParts
// (request.go) in reverse.
func chatContentPartsToInput(parts []interface{}) ([]interface{}, error) {
	result := make([]interface{}, 0, len(parts))
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		switch partType, _ := partMap["type"].(string); partType {
		case "text":
			result = append(result, map[string]interface{}{
				"type": "input_text",
				"text": partMap["text"],
			})
		case "image_url":
			imgURL, detail := "", ""
			switch v := partMap["image_url"].(type) {
			case string:
				imgURL = v
			case map[string]interface{}:
				imgURL, _ = v["url"].(string)
				detail, _ = v["detail"].(string)
			}
			entry := map[string]interface{}{
				"type":      "input_image",
				"image_url": imgURL,
			}
			if detail != "" {
				entry["detail"] = detail
			}
			result = append(result, entry)
		case "input_audio":
			entry := map[string]interface{}{"type": "input_audio"}
			if audio, ok := partMap["input_audio"].(map[string]interface{}); ok {
				entry["data"] = audio["data"]
				entry["format"] = audio["format"]
			}
			result = append(result, entry)
		default:
			// Unknown content part type -- skip rather than forward something
			// the Responses API would reject outright (mirrors
			// convertContentParts' handling of unknown Responses-side types).
			continue
		}
	}
	return result, nil
}

// chatAssistantMessageToInputItems converts one assistant Chat Completions
// message into zero or more Responses API input items: any reasoning items
// carried in "thinking_blocks", a message item for any text content, and one
// function_call item per tool call.
func chatAssistantMessageToInputItems(msg map[string]interface{}) []interface{} {
	var items []interface{}

	// Reasoning items must come first, in their original relative order --
	// see ResponseToChat's injectThinkingBlocks, which packs them into
	// "thinking_blocks" specifically so a reasoning model's tool-calling
	// loop keeps its chain of thought across Chat Completions turns (this
	// feature always runs the Responses API in stateless/store:false mode,
	// rebuilding the full input from message history on every turn -- see
	// ChatRequestToResponses -- so nothing else re-supplies this context).
	items = append(items, chatThinkingBlocksToReasoningItems(msg["thinking_blocks"])...)

	if content := chatAssistantContentToOutputParts(msg["content"]); len(content) > 0 {
		items = append(items, map[string]interface{}{
			"role":    "assistant",
			"content": content,
		})
	}

	toolCalls, _ := msg["tool_calls"].([]interface{})
	for _, tc := range toolCalls {
		tcMap, ok := tc.(map[string]interface{})
		if !ok {
			continue
		}
		fn, ok := tcMap["function"].(map[string]interface{})
		if !ok {
			continue
		}
		callID, _ := tcMap["id"].(string)
		name, _ := fn["name"].(string)
		args, _ := fn["arguments"].(string)
		items = append(items, map[string]interface{}{
			"type":      "function_call",
			"call_id":   callID,
			"name":      name,
			"arguments": args,
		})
	}

	return items
}

// chatThinkingBlocksToReasoningItems reconstructs Responses API "reasoning"
// input items from an assistant message's "thinking_blocks" field -- the
// inverse of ResponseToChat's injectThinkingBlocks. Only entries this
// package itself wrote (type == responsesReasoningBlockType) are consumed;
// anything else (e.g. an Anthropic-flavored openai.OpenAIThinkingBlock
// entry, should one somehow arrive on this route) is ignored rather than
// forwarded as a malformed reasoning item.
func chatThinkingBlocksToReasoningItems(raw interface{}) []interface{} {
	blocks, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	items := make([]interface{}, 0, len(blocks))
	for _, b := range blocks {
		blockMap, ok := b.(map[string]interface{})
		if !ok || blockMap["type"] != responsesReasoningBlockType {
			continue
		}
		item := map[string]interface{}{"type": "reasoning"}
		if id, ok := blockMap["id"].(string); ok && id != "" {
			item["id"] = id
		}
		if enc, ok := blockMap["encrypted_content"].(string); ok && enc != "" {
			item["encrypted_content"] = enc
		}
		if summary, ok := blockMap["summary"]; ok {
			item["summary"] = summary
		}
		items = append(items, item)
	}
	return items
}

// chatAssistantContentToOutputParts converts an assistant message's content
// (string or content-part array) into Responses API "output_text" parts --
// the part type Responses API expects assistant-authored input content to
// carry (see outputToInputItems in request.go, which does the same when
// replaying conversation history).
func chatAssistantContentToOutputParts(content interface{}) []interface{} {
	switch c := content.(type) {
	case string:
		if c == "" {
			return nil
		}
		return []interface{}{map[string]interface{}{"type": "output_text", "text": c}}
	case []interface{}:
		var out []interface{}
		for _, part := range c {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := partMap["text"].(string); ok && text != "" {
				out = append(out, map[string]interface{}{"type": "output_text", "text": text})
			}
		}
		return out
	default:
		return nil
	}
}

// chatToolContentToString flattens a "tool" message's content (a Chat
// Completions tool result is conventionally a string, but some clients send
// a JSON value) into the plain string the Responses API's
// function_call_output.output field expects.
func chatToolContentToString(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case nil:
		return ""
	default:
		if b, err := json.Marshal(c); err == nil {
			return string(b)
		}
		return ""
	}
}

// chatToolsToResponses converts Chat Completions' nested tool definitions
// ({type:"function", function:{name,...}}) to the Responses API's flat shape
// ({type:"function", name,...}). Mirrors convertTools (request.go) in
// reverse; also accepts an already-flat tool object unchanged, matching
// convertTools' own tolerance for both shapes on its side.
func chatToolsToResponses(raw map[string]interface{}) error {
	toolsRaw, ok := raw["tools"]
	if !ok {
		return nil
	}
	toolsArr, ok := toolsRaw.([]interface{})
	if !ok {
		return fmt.Errorf("tools must be an array")
	}
	converted := make([]interface{}, 0, len(toolsArr))
	for _, t := range toolsArr {
		toolMap, ok := t.(map[string]interface{})
		if !ok {
			continue
		}
		if toolType, _ := toolMap["type"].(string); toolType != "function" {
			// Non-function tools have no portable Chat Completions shape to
			// have arrived as in the first place; nothing to convert.
			converted = append(converted, toolMap)
			continue
		}
		fn, ok := toolMap["function"].(map[string]interface{})
		if !ok {
			// Already flat.
			converted = append(converted, toolMap)
			continue
		}
		flat := map[string]interface{}{"type": "function"}
		for k, v := range fn {
			flat[k] = v
		}
		converted = append(converted, flat)
	}
	if len(converted) > 0 {
		raw["tools"] = converted
	} else {
		delete(raw, "tools")
		delete(raw, "tool_choice")
	}
	return nil
}

// chatToolChoiceToResponses converts a nested Chat Completions tool_choice
// object ({type:"function", function:{name}}) to the Responses API's flat
// shape ({type:"function", name}). String values ("auto"/"none"/"required")
// are valid in both APIs and pass through unchanged.
func chatToolChoiceToResponses(raw map[string]interface{}) {
	tcMap, ok := raw["tool_choice"].(map[string]interface{})
	if !ok || tcMap["type"] != "function" {
		return
	}
	fn, ok := tcMap["function"].(map[string]interface{})
	if !ok {
		return
	}
	raw["tool_choice"] = map[string]interface{}{"type": "function", "name": fn["name"]}
}

// chatReasoningToResponses converts Chat Completions' top-level
// reasoning_effort into the Responses API's nested reasoning.effort.
func chatReasoningToResponses(raw map[string]interface{}) {
	effort, ok := raw["reasoning_effort"].(string)
	if !ok || effort == "" {
		return
	}
	if existing, ok := raw["reasoning"].(map[string]interface{}); ok {
		existing["effort"] = effort
		return
	}
	raw["reasoning"] = map[string]interface{}{"effort": effort}
}

// chatResponseFormatToText converts Chat Completions' response_format into
// the Responses API's text.format. Mirrors convertTextFormat (request.go)
// in reverse.
func chatResponseFormatToText(raw map[string]interface{}) {
	format, ok := raw["response_format"]
	if !ok {
		return
	}
	formatMap, ok := format.(map[string]interface{})
	if !ok {
		raw["text"] = map[string]interface{}{"format": format}
		return
	}
	if formatType, _ := formatMap["type"].(string); formatType == "json_schema" {
		jsonSchema, _ := formatMap["json_schema"].(map[string]interface{})
		flat := map[string]interface{}{"type": "json_schema"}
		for k, v := range jsonSchema {
			flat[k] = v
		}
		raw["text"] = map[string]interface{}{"format": flat}
		return
	}
	raw["text"] = map[string]interface{}{"format": format}
}

// deleteChatOnlyFields removes Chat-Completions-only fields the Responses
// API does not accept, mirroring deleteResponsesFields' role (request.go)
// for the opposite direction.
func deleteChatOnlyFields(raw map[string]interface{}) {
	delete(raw, "messages")
	delete(raw, "max_tokens")
	delete(raw, "max_completion_tokens")
	delete(raw, "reasoning_effort")
	delete(raw, "response_format")
	delete(raw, "stop")
	delete(raw, "frequency_penalty")
	delete(raw, "presence_penalty")
	delete(raw, "logit_bias")
	delete(raw, "n")
	delete(raw, "stream_options")
	delete(raw, "functions")
	delete(raw, "function_call")
}
