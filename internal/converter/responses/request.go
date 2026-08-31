package responses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// ResponsesMetadata holds Responses-API-only fields that are extracted from
// the request body before it is converted to Chat Completions format.
type ResponsesMetadata struct {
	Store              bool
	PreviousResponseID string
	Metadata           map[string]string
	TTL                int             // seconds; 0 = no expiry
	AccumulatedInput   json.RawMessage // full input array after history prepending; used for multi-turn storage
}

// ExtractResponsesMetadata extracts store / previous_response_id / metadata / ttl
// from a Responses API request body without modifying it.
func ExtractResponsesMetadata(body []byte) ResponsesMetadata {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return ResponsesMetadata{}
	}
	var meta ResponsesMetadata
	if v, ok := raw["store"].(bool); ok {
		meta.Store = v
	}
	if v, ok := raw["previous_response_id"].(string); ok {
		meta.PreviousResponseID = v
	}
	if v, ok := raw["metadata"].(map[string]interface{}); ok {
		meta.Metadata = make(map[string]string, len(v))
		for k, val := range v {
			if s, ok := val.(string); ok {
				meta.Metadata[k] = s
			}
		}
	}
	if v, ok := raw["ttl"].(float64); ok {
		meta.TTL = int(v)
	}
	return meta
}

// PrependOutputToInput prepends the output items from a previous response to
// the "input" field of a Responses API request body.
// The body is returned unmodified on any parse error.
func PrependOutputToInput(body []byte, output []OutputItem) ([]byte, error) {
	if len(output) == 0 {
		return body, nil
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, err
	}

	// Build input items from previous output
	prevItems := outputToInputItems(output)

	// Ensure current input is an array
	currentItems, err := inputToArray(raw["input"])
	if err != nil {
		return body, fmt.Errorf("PrependOutputToInput: %w", err)
	}

	raw["input"] = append(prevItems, currentItems...)
	result, err := json.Marshal(raw)
	if err != nil {
		return body, err
	}
	return result, nil
}

// ExtractInputArray returns the "input" field of a Responses API request body as a
// normalised JSON array (even if the original value was a plain string).
// Returns nil on any parse error or if the field is absent.
func ExtractInputArray(body []byte) json.RawMessage {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil
	}
	arr, err := inputToArray(raw["input"])
	if err != nil {
		return nil
	}
	result, err := json.Marshal(arr)
	if err != nil {
		return nil
	}
	return json.RawMessage(result)
}

// PrependHistoryToInput reconstructs the full conversation context for a multi-turn
// request.  It takes the current request body plus the stored accumulated input (all
// input items from the previous turn, including its own history) and the output items
// from the previous response, and produces:
//
//	accumulatedInput  (previous turns' full context as input items)
//	+ outputToInputItems(output)   (the previous assistant turn in input form)
//	+ current "input" items
//
// The body is returned unmodified on any parse error.
func PrependHistoryToInput(body []byte, accumulatedInput json.RawMessage, output []OutputItem) ([]byte, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, err
	}

	var historyItems []interface{}

	// Previous turns' accumulated context.
	if len(accumulatedInput) > 0 {
		var prevItems []interface{}
		if err := json.Unmarshal(accumulatedInput, &prevItems); err == nil {
			historyItems = append(historyItems, prevItems...)
		}
	}

	// Previous response output converted to input format.
	historyItems = append(historyItems, outputToInputItems(output)...)

	if len(historyItems) == 0 {
		return body, nil
	}

	// Ensure current input is an array.
	currentItems, err := inputToArray(raw["input"])
	if err != nil {
		return body, fmt.Errorf("PrependHistoryToInput: %w", err)
	}

	raw["input"] = append(historyItems, currentItems...)
	result, err := json.Marshal(raw)
	if err != nil {
		return body, err
	}
	return result, nil
}

// inputToArray coerces any valid "input" value to []interface{}.
func inputToArray(input interface{}) ([]interface{}, error) {
	if input == nil {
		return nil, fmt.Errorf("missing input field")
	}
	if s, ok := input.(string); ok {
		return []interface{}{
			map[string]interface{}{"role": "user", "content": s},
		}, nil
	}
	switch v := input.(type) {
	case []interface{}:
		return v, nil
	case map[string]interface{}:
		return []interface{}{v}, nil
	default:
		return nil, fmt.Errorf("unsupported input type")
	}
}

// outputToInputItems converts Responses API output items to input array items
// suitable for use as the "input" of the next request (multi-turn history).
func outputToInputItems(output []OutputItem) []interface{} {
	items := make([]interface{}, 0, len(output))
	for _, item := range output {
		switch item.Type {
		case "message":
			content := make([]interface{}, 0, len(item.Content))
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					// Keep output_text type: valid for assistant-role messages in both
					// the native Responses API (codex passthrough) and is handled by
					// convertContentParts for the Chat Completions conversion path.
					content = append(content, map[string]interface{}{
						"type": "output_text",
						"text": c.Text,
					})
				case "output_refusal":
					content = append(content, map[string]interface{}{
						"type":    "output_refusal",
						"refusal": c.Refusal,
					})
				}
			}
			items = append(items, map[string]interface{}{
				"type":    "message",
				"role":    item.Role,
				"content": content,
			})
		case "function_call":
			items = append(items, map[string]interface{}{
				"type":      "function_call",
				"call_id":   item.CallID,
				"name":      item.Name,
				"arguments": item.Arguments,
			})

		case "reasoning":
			// Skip reasoning items that have neither summary nor encrypted_content —
			// there's nothing useful to pass to the next turn.
			if len(item.Summary) == 0 && item.EncryptedContent == "" {
				continue
			}
			reasoningItem := map[string]interface{}{
				"type": "reasoning",
			}
			if item.ID != "" {
				reasoningItem["id"] = item.ID
			}
			// Build summary array (provider-agnostic; Anthropic/Vertex will each
			// extract what they need from it).
			if len(item.Summary) > 0 {
				summary := make([]interface{}, 0, len(item.Summary))
				for _, s := range item.Summary {
					if s.Text != "" {
						summary = append(summary, map[string]interface{}{
							"type": s.Type,
							"text": s.Text,
						})
					}
				}
				if len(summary) > 0 {
					reasoningItem["summary"] = summary
				}
			}
			// Preserve encrypted_content for same-provider round-trips (Anthropic).
			// Vertex ignores this field; OpenAI passthrough uses it verbatim.
			if item.EncryptedContent != "" {
				reasoningItem["encrypted_content"] = item.EncryptedContent
			}
			items = append(items, reasoningItem)

		case "computer_call":
			callItem := map[string]interface{}{
				"type":    "computer_call",
				"call_id": item.CallID,
				"name":    item.Name,
				"action":  item.Action,
			}
			if item.ID != "" {
				callItem["id"] = item.ID
			}
			items = append(items, callItem)

		case "computer_call_output":
			outputItem := map[string]interface{}{
				"type":    "computer_call_output",
				"call_id": item.CallID,
			}
			if item.ID != "" {
				outputItem["id"] = item.ID
			}
			if item.Output != nil {
				outputItem["output"] = item.Output
			}
			items = append(items, outputItem)

		case "web_search_call":
			wsItem := map[string]interface{}{
				"type": "web_search_call",
			}
			if item.ID != "" {
				wsItem["id"] = item.ID
			}
			if len(item.Queries) > 0 {
				wsItem["queries"] = item.Queries
			}
			items = append(items, wsItem)

		case "code_interpreter_call":
			ciItem := map[string]interface{}{
				"type": "code_interpreter_call",
				"code": item.Code,
			}
			if item.ID != "" {
				ciItem["id"] = item.ID
			}
			if item.Outputs != nil {
				ciItem["outputs"] = item.Outputs
			}
			items = append(items, ciItem)

		case "file_search_call":
			fsItem := map[string]interface{}{
				"type": "file_search_call",
			}
			if item.ID != "" {
				fsItem["id"] = item.ID
			}
			if item.Status != "" {
				fsItem["status"] = item.Status
			}
			if len(item.Queries) > 0 {
				fsItem["queries"] = item.Queries
			}
			if item.Results != nil {
				fsItem["results"] = item.Results
			}
			items = append(items, fsItem)

		case "image_generation_call":
			igItem := map[string]interface{}{
				"type": "image_generation_call",
			}
			if item.ID != "" {
				igItem["id"] = item.ID
			}
			if item.Status != "" {
				igItem["status"] = item.Status
			}
			if item.Result != "" {
				igItem["result"] = item.Result
			}
			if item.RevisedPrompt != "" {
				igItem["revised_prompt"] = item.RevisedPrompt
			}
			items = append(items, igItem)

		case "mcp_tool_call":
			mcpItem := map[string]interface{}{
				"type":         "mcp_tool_call",
				"call_id":      item.CallID,
				"name":         item.Name,
				"arguments":    item.Arguments,
				"server_label": item.ServerLabel,
			}
			if item.ID != "" {
				mcpItem["id"] = item.ID
			}
			if item.Error != nil {
				mcpItem["error"] = item.Error
			}
			items = append(items, mcpItem)

		case "local_shell_call":
			lsItem := map[string]interface{}{
				"type":    "local_shell_call",
				"call_id": item.CallID,
				"action":  item.Action,
			}
			if item.ID != "" {
				lsItem["id"] = item.ID
			}
			if item.Status != "" {
				lsItem["status"] = item.Status
			}
			items = append(items, lsItem)

		case "compaction":
			compItem := map[string]interface{}{
				"type": "compaction",
			}
			if item.ID != "" {
				compItem["id"] = item.ID
			}
			if item.EncryptedContent != "" {
				compItem["encrypted_content"] = item.EncryptedContent
			}
			items = append(items, compItem)
		}
	}
	return items
}

// PrepareCodexPassthrough strips proxy-internal fields and normalises the
// request body so it is accepted by OpenAI's native /v1/responses endpoint.
//
// Normalizations applied (in one JSON round-trip):
//  1. Strip store / metadata / ttl (handled by the proxy, not forwarded).
//  2. Strip previous_response_id when prevEntryHandled=true (history already
//     injected into input by PrependHistoryToInput).
//  3. input: single message object → wrapped in a one-element array.
//  4. instructions: array of messages → content joined as a plain string.
//  5. tools: nested Chat Completions format ({function:{name:...}}) →
//     flat Responses API format ({name:...}) for function-type tools.
func PrepareCodexPassthrough(body []byte, prevEntryHandled bool) []byte {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}

	// 1 & 2. Strip proxy-internal and conditionally previous_response_id.
	delete(raw, "store")
	delete(raw, "metadata")
	delete(raw, "ttl")
	if prevEntryHandled {
		delete(raw, "previous_response_id")
	}

	// 3. Normalize input: single dict → one-element array.
	if inputVal, ok := raw["input"]; ok {
		if inputMap, ok := inputVal.(map[string]interface{}); ok {
			raw["input"] = []interface{}{inputMap}
		}
	}

	// 4. Normalize instructions: array → plain string (native API requires string).
	if instVal, ok := raw["instructions"]; ok {
		if instArr, ok := instVal.([]interface{}); ok {
			var parts []string
			for _, item := range instArr {
				if m, ok := item.(map[string]interface{}); ok {
					if content, ok := m["content"].(string); ok && content != "" {
						parts = append(parts, content)
					}
				}
			}
			if len(parts) > 0 {
				raw["instructions"] = strings.Join(parts, "\n")
			} else {
				delete(raw, "instructions")
			}
		}
	}

	// 4.5. Drop reasoning.effort="none" for native passthrough.
	// OpenAI Responses API rejects reasoning.effort on non-reasoning models such as
	// gpt-4o-mini, while "none" is semantically equivalent to omitting reasoning.
	if reasoningVal, ok := raw["reasoning"]; ok {
		if reasoningMap, ok := reasoningVal.(map[string]interface{}); ok {
			if effort, ok := reasoningMap["effort"].(string); ok && effort == "none" {
				delete(raw, "reasoning")
			}
		}
	}

	// 5. Normalize tools: nested Chat Completions function format → flat Responses API format.
	// Input:  {type:"function", function:{name:"...", description:"...", parameters:{...}}}
	// Output: {type:"function", name:"...", description:"...", parameters:{...}}
	// Empty tools array is removed — some providers reject tools:[].
	if toolsVal, ok := raw["tools"]; ok {
		if toolsArr, ok := toolsVal.([]interface{}); ok {
			if len(toolsArr) == 0 {
				delete(raw, "tools")
				delete(raw, "tool_choice")
			} else {
				normalized := make([]interface{}, len(toolsArr))
				for i, t := range toolsArr {
					toolMap, ok := t.(map[string]interface{})
					if !ok {
						normalized[i] = t
						continue
					}
					if toolMap["type"] == "function" {
						if funcDef, ok := toolMap["function"].(map[string]interface{}); ok {
							flat := map[string]interface{}{"type": "function"}
							for k, v := range funcDef {
								flat[k] = v
							}
							normalized[i] = flat
							continue
						}
					}
					normalized[i] = t
				}
				raw["tools"] = normalized
			}
		}
	}

	// 6. Convert compaction items to user messages.
	// OpenAI's native Responses API does not support type:"compaction" items.
	// Convert them to standard user messages carrying the compacted context summary.
	if inputVal, ok := raw["input"]; ok {
		if inputArr, ok := inputVal.([]interface{}); ok {
			out := make([]interface{}, 0, len(inputArr))
			changed := false
			for _, item := range inputArr {
				itemMap, ok := item.(map[string]interface{})
				if !ok || itemMap["type"] != "compaction" {
					out = append(out, item)
					continue
				}
				changed = true
				ec, _ := itemMap["encrypted_content"].(string)
				if ec != "" {
					out = append(out, map[string]interface{}{
						"type": "message",
						"role": "user",
						"content": []interface{}{
							map[string]interface{}{
								"type": "input_text",
								"text": "[Conversation context summary]: " + ec,
							},
						},
					})
				}
				// Skip empty compaction items
			}
			if changed {
				raw["input"] = out
			}
		}
	}

	result, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return result
}

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

	messages, err := convertInputValue(raw["input"])
	if err != nil {
		return nil, fmt.Errorf("failed to convert input: %w", err)
	}

	// Prepend system/developer messages from instructions
	if instructions, ok := raw["instructions"]; ok {
		instMsgs, err := convertInstructions(instructions)
		if err != nil {
			return nil, fmt.Errorf("failed to convert instructions: %w", err)
		}
		if len(instMsgs) > 0 {
			messages = append(instMsgs, messages...)
		}
	}

	// Set messages
	raw["messages"] = messages

	// max_output_tokens -> max_tokens (universal Chat Completions parameter).
	// Reasoning models (o1/o3/o4/gpt-5) will have this renamed to
	// max_completion_tokens by openai.ReplaceBodyParam applied after conversion.
	if maxOut, ok := raw["max_output_tokens"]; ok {
		raw["max_tokens"] = maxOut
	}

	// Convert tools from flat to nested format
	if err := convertTools(raw); err != nil {
		return nil, err
	}

	// Convert tool_choice
	if err := convertToolChoice(raw); err != nil {
		return nil, err
	}

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

// convertInputValue converts the "input" value to Chat Completions "messages".
func convertInputValue(input interface{}) ([]interface{}, error) {
	if input == nil {
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

	var inputArr []interface{}
	switch v := input.(type) {
	case []interface{}:
		inputArr = v
	case map[string]interface{}:
		inputArr = []interface{}{v}
	default:
		return nil, fmt.Errorf("input must be string, object, or array")
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

		case "reasoning":
			// Serialize reasoning as a synthetic assistant message using summary text.
			// encrypted_content is dropped — it is only meaningful to the original provider.
			flushToolCalls()
			var textParts []interface{}
			if summary, ok := itemMap["summary"].([]interface{}); ok {
				for _, s := range summary {
					sm, ok := s.(map[string]interface{})
					if !ok {
						continue
					}
					text, _ := sm["text"].(string)
					if text != "" {
						textParts = append(textParts, map[string]interface{}{
							"type": "text",
							"text": "[Reasoning]: " + text,
						})
					}
				}
			}
			if len(textParts) > 0 {
				messages = append(messages, map[string]interface{}{
					"role":    "assistant",
					"content": textParts,
				})
			}

		case "web_search_call":
			// Serialize as a synthetic function_call + tool result pair.
			flushToolCalls()
			callID, _ := itemMap["call_id"].(string)
			if callID == "" {
				callID, _ = itemMap["id"].(string)
			}
			if callID != "" {
				name, _ := itemMap["name"].(string)
				if name == "" {
					name = "web_search"
				}
				pendingToolCalls = append(pendingToolCalls, map[string]interface{}{
					"id":   callID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": "{}",
					},
				})
				flushToolCalls()
				var outputStr string
				if results := itemMap["results"]; results != nil {
					if b, err := json.Marshal(results); err == nil {
						outputStr = string(b)
					}
				}
				messages = append(messages, map[string]interface{}{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      outputStr,
				})
			}

		case "computer_call_output":
			// Convert screenshot/output to a user message with image content.
			flushToolCalls()
			var content []interface{}
			if output, ok := itemMap["output"].(map[string]interface{}); ok {
				outputType, _ := output["type"].(string)
				switch outputType {
				case "computer_screenshot":
					// Schema format: image_url is a nested {url: "...", detail: "..."} object.
					if imgURLMap, ok := output["image_url"].(map[string]interface{}); ok {
						if imgURL, ok := imgURLMap["url"].(string); ok && imgURL != "" {
							content = append(content, map[string]interface{}{
								"type": "image_url",
								"image_url": map[string]interface{}{
									"url": imgURL,
								},
							})
						}
					}
				default:
					// Legacy/proxy format: image_url is a plain string.
					if imgURL, ok := output["image_url"].(string); ok && imgURL != "" {
						content = append(content, map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": imgURL,
							},
						})
					}
				}
			}
			if len(content) == 0 {
				content = append(content, map[string]interface{}{
					"type": "text",
					"text": "[computer_call_output]",
				})
			}
			messages = append(messages, map[string]interface{}{
				"role":    "user",
				"content": content,
			})

		case "code_interpreter_call":
			// Serialize as a synthetic function_call + tool result pair.
			flushToolCalls()
			callID, _ := itemMap["call_id"].(string)
			if callID == "" {
				callID, _ = itemMap["id"].(string)
			}
			if callID != "" {
				code, _ := itemMap["code"].(string)
				argsJSON, _ := json.Marshal(map[string]interface{}{"code": code})
				pendingToolCalls = append(pendingToolCalls, map[string]interface{}{
					"id":   callID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      "code_interpreter",
						"arguments": string(argsJSON),
					},
				})
				flushToolCalls()
				var outputStr string
				if outputs := itemMap["outputs"]; outputs != nil {
					if b, err := json.Marshal(outputs); err == nil {
						outputStr = string(b)
					}
				}
				messages = append(messages, map[string]interface{}{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      outputStr,
				})
			}

		case "file_search_call":
			// Serialize search results as a synthetic function_call + tool result pair.
			flushToolCalls()
			callID, _ := itemMap["call_id"].(string)
			if callID == "" {
				callID, _ = itemMap["id"].(string)
			}
			if callID != "" {
				pendingToolCalls = append(pendingToolCalls, map[string]interface{}{
					"id":   callID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      "file_search",
						"arguments": "{}",
					},
				})
				flushToolCalls()
				var outputStr string
				if results := itemMap["results"]; results != nil {
					if b, err := json.Marshal(results); err == nil {
						outputStr = string(b)
					}
				}
				messages = append(messages, map[string]interface{}{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      outputStr,
				})
			}

		case "image_generation_call":
			// Emit the generated image as an assistant message with an image_url part.
			flushToolCalls()
			result, _ := itemMap["result"].(string)
			if result != "" {
				messages = append(messages, map[string]interface{}{
					"role": "assistant",
					"content": []interface{}{
						map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": result,
							},
						},
					},
				})
			}

		case "mcp_tool_call":
			// Serialize as a synthetic function_call + tool result pair.
			flushToolCalls()
			callID, _ := itemMap["call_id"].(string)
			if callID == "" {
				callID, _ = itemMap["id"].(string)
			}
			if callID != "" {
				name, _ := itemMap["name"].(string)
				serverLabel, _ := itemMap["server_label"].(string)
				if name == "" {
					name = "mcp_tool"
				}
				if serverLabel != "" {
					name = serverLabel + "_" + name
				}
				var argsStr string
				if input := itemMap["input"]; input != nil {
					if b, err := json.Marshal(input); err == nil {
						argsStr = string(b)
					}
				}
				if argsStr == "" {
					argsStr = "{}"
				}
				pendingToolCalls = append(pendingToolCalls, map[string]interface{}{
					"id":   callID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": argsStr,
					},
				})
				flushToolCalls()
				var outputStr string
				if output := itemMap["output"]; output != nil {
					if b, err := json.Marshal(output); err == nil {
						outputStr = string(b)
					}
				}
				messages = append(messages, map[string]interface{}{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      outputStr,
				})
			}

		case "compaction":
			// A compaction item carries an encrypted_content summary of prior context.
			// Inject it as a user message so the model has the compacted context.
			flushToolCalls()
			ec, _ := itemMap["encrypted_content"].(string)
			if ec != "" {
				messages = append(messages, map[string]interface{}{
					"role":    "user",
					"content": "[Conversation context summary]: " + ec,
				})
			}

		default:
			// Flush any pending tool calls before a regular message
			flushToolCalls()
			// only convert items that have a "role" field (messages).
			// Skip unrecognized input item types (e.g. item_reference) to avoid
			// sending malformed messages to Chat Completions providers.
			if _, hasRole := itemMap["role"]; !hasRole && itemType != "" && itemType != "message" {
				continue
			}
			msg, err := convertMessage(itemMap)
			if err != nil {
				return nil, err
			}
			messages = append(messages, msg)
		}
	}

	// Flush any remaining tool calls
	flushToolCalls()

	return messages, nil
}

// convertMessage converts an InputMessage or ResponseOutputMessage to a Chat Completions message.
// Responses-API-only fields (type, phase, status) are intentionally dropped here because
// Chat Completions providers reject unknown parameters on message objects.
func convertMessage(item map[string]interface{}) (map[string]interface{}, error) {
	role := item["role"]
	if role == "developer" {
		// "developer" is the Responses API's rename of "system" (also emitted by the
		// official OpenAI SDK). Most Chat Completions providers reached through this
		// generic converter (e.g. DeepSeek and other OpenAI-compatible backends) only
		// recognize the classic role set and reject "developer" outright, so downgrade
		// it to the universally-supported "system" — the two are semantically identical.
		role = "system"
	}
	msg := map[string]interface{}{
		"role": role,
	}

	content := item["content"]
	switch c := content.(type) {
	case string:
		msg["content"] = c
	case []interface{}:
		converted, err := convertContentParts(c)
		if err != nil {
			return nil, err
		}
		msg["content"] = converted
	default:
		msg["content"] = content
	}

	return msg, nil
}

// convertContentParts converts Responses API content parts to Chat Completions format.
func convertContentParts(parts []interface{}) ([]interface{}, error) {
	var result []interface{}
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)
		switch partType {
		case "output_text":
			result = append(result, map[string]interface{}{
				"type": "text",
				"text": partMap["text"],
			})

		case "output_refusal":
			result = append(result, map[string]interface{}{
				"type": "text",
				"text": partMap["refusal"],
			})

		case "input_text":
			result = append(result, map[string]interface{}{
				"type": "text",
				"text": partMap["text"],
			})

		case "input_image":
			imgURL := ""
			// image_url can be a plain string or an object {url: "...", detail: "..."}
			switch v := partMap["image_url"].(type) {
			case string:
				imgURL = v
			case map[string]interface{}:
				imgURL, _ = v["url"].(string)
			}
			if imgURL == "" {
				if _, hasFileID := partMap["file_id"]; hasFileID {
					return nil, fmt.Errorf("input_image with file_id is not supported in chat completions")
				}
				// Unsupported sources — avoid silent corruption
				return nil, fmt.Errorf("input_image missing image_url")
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

		case "input_file":
			if fileID, _ := partMap["file_id"].(string); fileID != "" {
				return nil, converterutil.NewRequestValidationError("input_file.file_id", "file_id is not supported for this route")
			}
			return nil, converterutil.NewRequestValidationError("input_file", "input_file is not supported in chat completions")

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
			// skip unknown content part types silently.
			// Passing unknown types through would cause provider rejection.
			continue
		}
	}
	return result, nil
}

// convertFunctionCallOutput converts a function_call_output input item to a tool message.
func convertFunctionCallOutput(item map[string]interface{}) map[string]interface{} {
	callID, _ := item["call_id"].(string)
	outputVal := item["output"]
	outputStr, ok := outputVal.(string)
	if !ok {
		if b, err := json.Marshal(outputVal); err == nil {
			outputStr = string(b)
		}
	}

	return map[string]interface{}{
		"role":         "tool",
		"tool_call_id": callID,
		"content":      outputStr,
	}
}

// convertInstructions converts the "instructions" field to Chat Completions messages.
func convertInstructions(instructions interface{}) ([]interface{}, error) {
	if instructions == nil {
		return nil, nil
	}
	if s, ok := instructions.(string); ok {
		if s == "" {
			return nil, nil
		}
		return []interface{}{
			map[string]interface{}{
				// "system", not "developer": see convertMessage for why.
				"role":    "system",
				"content": s,
			},
		}, nil
	}

	return convertInputValue(instructions)
}

// convertTools converts Responses API flat tools to Chat Completions nested format.
func convertTools(raw map[string]interface{}) error {
	toolsRaw, ok := raw["tools"]
	if !ok {
		return nil
	}
	toolsArr, ok := toolsRaw.([]interface{})
	if !ok {
		return fmt.Errorf("tools must be an array")
	}

	var converted []interface{}
	for _, t := range toolsArr {
		toolMap, ok := t.(map[string]interface{})
		if !ok {
			continue
		}

		toolType, _ := toolMap["type"].(string)

		if toolType != "function" {
			// Non-function tools (web_search_preview, file_search, code_interpreter,
			// computer_use, etc.) are Responses-API built-in constructs with no
			// equivalent in the generic Chat Completions tools array. Providers that
			// reach this path (converted fallback) don't understand these types and
			// will reject the request. Skip them — provider-specific native paths
			// (Vertex, Anthropic) handle built-in tools through their own converters.
			continue
		}

		var funcDef map[string]interface{}
		// Support both flat Responses API format and nested Chat Completions format:
		// Flat:   {type: "function", name: "x", description: "y", parameters: {...}, strict: bool}
		// Nested: {type: "function", function: {name: "x", description: "y", parameters: {...}, strict: bool}}
		if nested, ok := toolMap["function"].(map[string]interface{}); ok {
			funcDef = nested
		} else {
			funcDef = map[string]interface{}{}
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
		// tool_choice without tools is invalid for Chat Completions providers.
		delete(raw, "tool_choice")
	}
	return nil
}

// convertToolChoice converts Responses API tool_choice to Chat Completions format.
func convertToolChoice(raw map[string]interface{}) error {
	tc, ok := raw["tool_choice"]
	if !ok {
		return nil
	}

	tcMap, ok := tc.(map[string]interface{})
	if !ok {
		// string values like "auto", "none", "required" pass through unchanged
		return nil
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
		return nil
	}

	// Non-function tool_choice types (e.g. web_search_preview, file_search) reference
	// Responses-API built-in tools.  Pass them through: provider-specific converters
	// downstream (Vertex, Anthropic, OpenAI) handle what they support and ignore
	// what they don't.
	return nil
}

// convertReasoning extracts reasoning.effort and sets it as top-level reasoning_effort.
// Skips "none" effort (equivalent to no reasoning) and empty values.
// Note: reasoning.generate_summary is a Responses-API-only field with no Chat Completions
// equivalent — it is intentionally not forwarded.
func convertReasoning(raw map[string]interface{}) {
	reasoning, ok := raw["reasoning"]
	if !ok {
		return
	}
	reasoningMap, ok := reasoning.(map[string]interface{})
	if !ok {
		return
	}
	if effort, ok := reasoningMap["effort"].(string); ok && effort != "" && effort != "none" {
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
// comprehensive list of Responses-only fields that must not
// leak to Chat Completions providers.
func deleteResponsesFields(raw map[string]interface{}) {
	delete(raw, "input")
	delete(raw, "instructions")
	delete(raw, "max_output_tokens")
	delete(raw, "metadata")
	delete(raw, "previous_response_id")
	delete(raw, "store")
	delete(raw, "ttl")
	delete(raw, "reasoning")
	delete(raw, "text")
	delete(raw, "conversation")
	delete(raw, "include")
	delete(raw, "stream_options")
	delete(raw, "truncation")
	delete(raw, "safety_identifier")
	delete(raw, "service_tier")
	delete(raw, "background")
	delete(raw, "prompt")
	delete(raw, "prompt_cache_key")
	delete(raw, "prompt_cache_retention")
}
