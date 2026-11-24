package responses

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// ResponsesMetadata holds Responses-API-only fields that are extracted from
// the request body before it is converted to Chat Completions format.
type ResponsesMetadata struct {
	Store              bool
	PreviousResponseID string
	Metadata           map[string]string
	TTL                int             // seconds; 0 = no expiry
	AccumulatedInput   json.RawMessage // full input array after history prepending; used for multi-turn storage
	Text               interface{}     // {"format": <request text.format>} echoed in responses the router builds; nil = no format
}

// ExtractResponsesMetadata extracts store / previous_response_id / metadata / ttl
// and the text.format to echo from a Responses API request body without
// modifying it.
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
	// Only a well-formed format is echoed. text.verbosity is not applied on
	// converted routes, so echoing it would claim a setting the model never
	// got; a malformed format (which OpenAI rejects) would make the response's
	// text invalid.
	if text, ok := raw["text"].(map[string]interface{}); ok {
		if format, ok := text["format"].(map[string]interface{}); ok && isWellFormedTextFormat(format) {
			meta.Text = map[string]interface{}{"format": format}
		}
	}
	return meta
}

// isWellFormedTextFormat reports whether format is a valid Responses
// text.format: {type: "text"}, {type: "json_object"}, or {type: "json_schema"}
// with a string name, an object schema and, if set, a bool strict and a string
// description.
func isWellFormedTextFormat(format map[string]interface{}) bool {
	switch format["type"] {
	case "text", "json_object":
		return true
	case "json_schema":
		name, _ := format["name"].(string)
		_, hasSchema := format["schema"].(map[string]interface{})
		_, strictOK := format["strict"].(bool)
		_, descriptionOK := format["description"].(string)
		return name != "" && hasSchema &&
			(strictOK || format["strict"] == nil) &&
			(descriptionOK || format["description"] == nil)
	}
	return false
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
					part := map[string]interface{}{"type": "output_text", "text": c.Text}
					if c.ThoughtSignature != "" {
						part["thought_signature"] = c.ThoughtSignature
					}
					content = append(content, part)
				case "output_refusal", "refusal":
					content = append(content, map[string]interface{}{
						"type":    "refusal",
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
			// Skip reasoning items that have neither summary, reasoning text nor
			// encrypted_content — there's nothing useful to pass to the next turn.
			if len(item.Summary) == 0 && len(item.Content) == 0 && item.EncryptedContent == "" {
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
			// Raw reasoning_text content (e.g. gpt-oss) is the only text such an
			// item carries when it has no summary; keep it so the next turn still
			// gets the reasoning (PrepareCodexPassthrough recovers a summary from
			// it, RequestToChat reads it directly).
			if _, hasSummary := reasoningItem["summary"]; !hasSummary {
				content := make([]interface{}, 0, len(item.Content))
				for _, c := range item.Content {
					if c.Text != "" {
						content = append(content, map[string]interface{}{
							"type": c.Type,
							"text": c.Text,
						})
					}
				}
				if len(content) > 0 {
					reasoningItem["content"] = content
				} else if item.EncryptedContent != "" {
					// The Responses input schema requires an array even for encrypted-only reasoning.
					reasoningItem["summary"] = []interface{}{}
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
//  6. compaction items → synthetic user messages (native API has no compaction type).
//  7. reasoning items missing a valid "summary" list → recovered from a
//     non-standard "content" field; given an empty list if they only carry
//     encrypted_content; dropped only if nothing usable remains.
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

	// 4.5. Drop reasoning.effort="none" for native passthrough: non-reasoning
	// models such as gpt-4o-mini reject reasoning.effort. Models that take "none"
	// keep it, as omitting effort there means the default effort.
	if reasoningVal, ok := raw["reasoning"]; ok {
		if reasoningMap, ok := reasoningVal.(map[string]interface{}); ok {
			model, _ := raw["model"].(string)
			if effort, ok := reasoningMap["effort"].(string); ok && effort == "none" && !openai.SupportsReasoningEffortNone(model) {
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
				additionalTools, _ := raw["additional_tools"].([]interface{})
				if len(additionalTools) == 0 {
					delete(raw, "tool_choice")
				}
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
							flat := make(map[string]interface{}, len(toolMap)+len(funcDef))
							for k, v := range toolMap {
								if k != "function" {
									flat[k] = v
								}
							}
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

	// 7. Normalize reasoning items so they satisfy OpenAI's native validation
	// ("Invalid 'summary': summary is required and must be a list for reasoning").
	// Some upstream client SDKs / synthetic replays echo reasoning items back using
	// a non-standard {"content": [{"type": "reasoning_text", "text": "..."}]} shape
	// instead of the documented {"summary": [{"type": "summary_text", "text": "..."}]}
	// shape. Passthrough forwards the body mostly as-is, so without this the
	// malformed item reaches the provider unmodified and the whole request is
	// rejected — recover a valid summary from "content" when possible.
	//
	// A "summary" key that is already a list — even an empty one — already
	// satisfies "must be a list" and is left untouched. Older stored history
	// and client input can carry encrypted reasoning without a summary. Use an
	// empty list when there is no text to recover. Drop items with neither a
	// usable summary nor encrypted_content.
	if inputVal, ok := raw["input"]; ok {
		if inputArr, ok := inputVal.([]interface{}); ok {
			out := make([]interface{}, 0, len(inputArr))
			changed := false
			for _, item := range inputArr {
				itemMap, ok := item.(map[string]interface{})
				if !ok || itemMap["type"] != "reasoning" {
					out = append(out, item)
					continue
				}
				if _, ok := itemMap["summary"].([]interface{}); ok {
					out = append(out, item)
					continue
				}
				var recovered []interface{}
				if content, ok := itemMap["content"].([]interface{}); ok {
					for _, c := range content {
						cm, ok := c.(map[string]interface{})
						if !ok {
							continue
						}
						if text, _ := cm["text"].(string); text != "" {
							recovered = append(recovered, map[string]interface{}{
								"type": "summary_text",
								"text": text,
							})
						}
					}
				}
				if len(recovered) > 0 {
					changed = true
					itemMap["summary"] = recovered
					delete(itemMap, "content")
					out = append(out, itemMap)
					continue
				}
				if ec, ok := itemMap["encrypted_content"].(string); ok && ec != "" {
					changed = true
					itemMap["summary"] = []interface{}{}
					out = append(out, item)
					continue
				}
				// Genuinely nothing useful — drop rather than forward broken.
				changed = true
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
	if !bytes.Contains(body, []byte(`"input"`)) && bytes.IndexByte(body, '\\') < 0 {
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
		return nil, converterutil.RequestJSONValidationError(err)
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

	converter := inputMessageConverter{lastUserIdx: -1}
	for _, item := range inputArr {
		if fields, ok := item.(map[string]interface{}); ok {
			if err := converter.addItem(fields); err != nil {
				return nil, err
			}
		}
	}
	return converter.finish(), nil
}

// inputMessageConverter folds the Responses items of one assistant turn into
// a single Chat message. Reasoning waits for the next item to claim it.
type inputMessageConverter struct {
	messages           []interface{}
	lastUserIdx        int
	pendingAssistant   map[string]interface{}
	pendingToolCalls   []interface{}
	pendingReasoning   []string
	unclaimedReasoning []string
}

func (c *inputMessageConverter) claimReasoning() {
	c.pendingReasoning = append(c.pendingReasoning, c.unclaimedReasoning...)
	c.unclaimedReasoning = nil
}

// flushAssistantTurn emits the assistant turn built so far, if any.
// Unclaimed reasoning stays pending for the next item.
func (c *inputMessageConverter) flushAssistantTurn() {
	msg := c.pendingAssistant
	if msg == nil {
		if len(c.pendingToolCalls) == 0 {
			// Reasoning-only turn (e.g. max_output_tokens ran out mid-reasoning).
			// Dropped: an assistant message with empty content, or two assistant
			// messages in a row, is rejected by several Chat providers, and
			// reasoning alone gives the model nothing to continue from.
			c.pendingReasoning = nil
			return
		}
		msg = map[string]interface{}{"role": "assistant", "content": nil}
	}
	if len(c.pendingToolCalls) > 0 {
		msg["tool_calls"] = c.pendingToolCalls
	}
	if len(c.pendingReasoning) > 0 {
		msg["reasoning_content"] = strings.Join(c.pendingReasoning, "\n\n")
	}
	c.messages = append(c.messages, msg)
	c.pendingAssistant, c.pendingToolCalls, c.pendingReasoning = nil, nil, nil
}

// endAssistantTurn closes the turn before an item that is not part of it
// (user/system message, tool result, ...). Reasoning nothing claimed stays
// with the turn it ended, or is dropped with a reasoning-only turn.
func (c *inputMessageConverter) endAssistantTurn() {
	c.claimReasoning()
	c.flushAssistantTurn()
}

func (c *inputMessageConverter) addToolCall(toolCall map[string]interface{}) {
	c.claimReasoning()
	c.pendingToolCalls = append(c.pendingToolCalls, toolCall)
}

// addAssistantMessage starts a turn with an assistant message item; the
// function_call items that follow it join the same Chat message.
func (c *inputMessageConverter) addAssistantMessage(msg map[string]interface{}) {
	if c.pendingAssistant != nil || len(c.pendingToolCalls) > 0 {
		c.flushAssistantTurn()
	}
	c.claimReasoning()
	c.pendingAssistant = msg
}

func (c *inputMessageConverter) addItem(itemMap map[string]interface{}) error {
	itemType, _ := itemMap["type"].(string)

	switch itemType {
	case "function_call":
		c.addToolCall(chatFunctionCall(
			converterutil.GetString(itemMap, "call_id"),
			converterutil.GetString(itemMap, "name"),
			converterutil.GetString(itemMap, "arguments"),
		))

	case "function_call_output":
		// Flush any pending tool calls before the output
		c.endAssistantTurn()
		msg := convertFunctionCallOutput(itemMap)
		c.messages = append(c.messages, msg)

	case "reasoning":
		// Carried as reasoning_content of the assistant message it precedes —
		// the field DeepSeek-style providers require back on tool-call turns.
		// encrypted_content is dropped — it is only meaningful to the original provider.
		if text := ReasoningItemText(itemMap); text != "" {
			c.unclaimedReasoning = append(c.unclaimedReasoning, text)
		}

	case "web_search_call", "code_interpreter_call", "file_search_call", "mcp_tool_call":
		c.addSyntheticToolTurn(itemType, itemMap)

	case "computer_call_output":
		c.endAssistantTurn()
		content := computerScreenshotPart(itemMap)
		c.messages = append(c.messages, map[string]interface{}{
			"role": "user", "content": []interface{}{content},
		})

	case "image_generation_call":
		c.flushAssistantTurn()
		if url, _ := itemMap["result"].(string); url != "" {
			c.addAssistantMessage(map[string]interface{}{
				"role": "assistant", "content": []interface{}{chatImagePart(url)},
			})
			c.flushAssistantTurn()
		}

	case "compaction":
		// A compaction item carries an encrypted_content summary of prior context.
		// Inject it as a user message so the model has the compacted context.
		c.endAssistantTurn()
		ec, _ := itemMap["encrypted_content"].(string)
		if ec != "" {
			c.messages = append(c.messages, map[string]interface{}{
				"role":    "user",
				"content": "[Conversation context summary]: " + ec,
			})
		}

	default:
		return c.addMessage(itemType, itemMap)
	}
	return nil
}

func (c *inputMessageConverter) addMessage(itemType string, itemMap map[string]interface{}) error {
	// Skip unknown item types without a role, such as item_reference.
	if _, hasRole := itemMap["role"]; !hasRole && itemType != "" && itemType != "message" {
		c.flushAssistantTurn()
		return nil
	}
	msg, err := convertMessage(itemMap)
	if err != nil {
		return err
	}
	if msg["role"] == "assistant" {
		// A terminal refusal belongs to the pending turn, including its
		// tool calls. A separate assistant message would split a call
		// from the function_call_output supplied by the client next.
		if text := refusalInputText(itemMap); text != "" && (c.pendingAssistant != nil || len(c.pendingToolCalls) > 0) {
			c.claimReasoning()
			if c.pendingAssistant == nil {
				c.pendingAssistant = map[string]interface{}{"role": "assistant"}
			}
			appendAssistantText(c.pendingAssistant, text)
			return nil
		}
		c.addAssistantMessage(msg)
		return nil
	}
	c.endAssistantTurn()
	if msg["role"] == "user" {
		c.lastUserIdx = len(c.messages)
	}
	c.messages = append(c.messages, msg)
	return nil
}

func (c *inputMessageConverter) finish() []interface{} {
	c.endAssistantTurn()
	// reasoning_content is only sent back for the current tool loop — the
	// assistant turns after the last user message. DeepSeek thinking mode and
	// Kimi require it there; older turns' reasoning is useless to every
	// provider, inflates input tokens, and some (deepseek-reasoner) reject it
	// outright on past turns.
	for i := 0; i < c.lastUserIdx; i++ {
		if msg, ok := c.messages[i].(map[string]interface{}); ok && msg["role"] == "assistant" {
			delete(msg, "reasoning_content")
		}
	}

	return c.messages
}

func chatFunctionCall(id, name, arguments string) map[string]interface{} {
	return map[string]interface{}{
		"id":       id,
		"type":     "function",
		"function": map[string]interface{}{"name": name, "arguments": arguments},
	}
}

func (c *inputMessageConverter) addSyntheticToolTurn(itemType string, item map[string]interface{}) {
	// Built-in tool calls are emitted as their own assistant/tool pair. Flush
	// the previous assistant turn before reading the ID, while keeping any
	// unclaimed reasoning available for the next usable item.
	c.flushAssistantTurn()
	callID := cmp.Or(converterutil.GetString(item, "call_id"), converterutil.GetString(item, "id"))
	if callID == "" {
		return
	}

	name := strings.TrimSuffix(itemType, "_call")
	arguments := "{}"
	output := item["results"]
	if itemType == "web_search_call" || itemType == "mcp_tool_call" {
		name = cmp.Or(converterutil.GetString(item, "name"), name)
	}
	switch itemType {
	case "code_interpreter_call":
		code, _ := item["code"].(string)
		arguments = syntheticToolJSON(map[string]string{"code": code})
		output = item["outputs"]
	case "mcp_tool_call":
		if label, _ := item["server_label"].(string); label != "" {
			name = label + "_" + name
		}
		if input := syntheticToolJSON(item["input"]); input != "" {
			arguments = input
		}
		output = item["output"]
	}

	c.addToolCall(chatFunctionCall(callID, name, arguments))
	c.flushAssistantTurn()
	c.messages = append(c.messages, map[string]interface{}{
		"role":         "tool",
		"tool_call_id": callID,
		"content":      syntheticToolJSON(output),
	})
}

func syntheticToolJSON(value interface{}) string {
	if value == nil {
		return ""
	}
	data, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(data)
}

func computerScreenshotPart(item map[string]interface{}) map[string]interface{} {
	output, _ := item["output"].(map[string]interface{})
	var url string
	if output["type"] == "computer_screenshot" {
		image, _ := output["image_url"].(map[string]interface{})
		url, _ = image["url"].(string)
	} else {
		url, _ = output["image_url"].(string)
	}
	if url != "" {
		return chatImagePart(url)
	}
	return map[string]interface{}{"type": "text", "text": "[computer_call_output]"}
}

func chatImagePart(url string) map[string]interface{} {
	return map[string]interface{}{
		"type":      "image_url",
		"image_url": map[string]interface{}{"url": url},
	}
}

// ReasoningItemText returns the reasoning text of a Responses reasoning item.
// Raw reasoning_text content wins over summary_text when both are present:
// Chat's reasoning_content is the model's reasoning itself, a summary only a
// digest of it. Some clients echo reasoning back in "content" alone (see
// PrepareCodexPassthrough), so neither field may be ignored.
func ReasoningItemText(item map[string]interface{}) string {
	if text := joinReasoningParts(item["content"]); text != "" {
		return text
	}
	return joinReasoningParts(item["summary"])
}

func joinReasoningParts(raw interface{}) string {
	parts, _ := raw.([]interface{})
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		if text, _ := partMap["text"].(string); text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n\n")
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

		case "output_refusal", "refusal":
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

	// Responses API "required" and "any" mean "force any tool call" with no
	// specific function preference.  Chat Completions expresses the same intent
	// with the plain string "required".  DeepSeek (and possibly other
	// OpenAI-compatible backends) reject the object form outright.
	if tcType == "required" || tcType == "any" {
		raw["tool_choice"] = "required"
		return nil
	}

	// Non-function tool_choice types (e.g. web_search_preview, file_search) reference
	// Responses-API built-in tools.  Pass them through: provider-specific converters
	// downstream (Vertex, Anthropic, OpenAI) handle what they support and ignore
	// what they don't.
	return nil
}

// convertReasoning extracts reasoning.effort and sets it as top-level reasoning_effort.
// Skips empty values, and "none" unless openai.SupportsReasoningEffortNone.
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
	effort, _ := reasoningMap["effort"].(string)
	if effort == "" {
		return
	}
	model, _ := raw["model"].(string)
	if effort != "none" || openai.SupportsReasoningEffortNone(model) {
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
