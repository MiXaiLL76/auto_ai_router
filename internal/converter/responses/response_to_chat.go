package responses

import (
	"encoding/json"
	"fmt"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// ResponseToChat converts a Responses API response body into a Chat
// Completions response body -- the mirror of ChatToResponse, used for a
// model marked responses_only (see ChatRequestToResponses/config.go's
// ResponsesOnly): AIR called the provider's native /v1/responses on the
// client's behalf, and must hand back exactly the shape a
// /v1/chat/completions caller expects.
func ResponseToChat(body []byte) ([]byte, error) {
	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse responses API response: %w", err)
	}

	var textParts, reasoningParts []string
	var refusal string
	var toolCalls []openai.OpenAIToolCall
	var images []openai.ImageData
	var thinkingBlocks []responsesReasoningBlock
	hasFunctionCall := false

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					if c.Text != "" {
						textParts = append(textParts, c.Text)
					}
				case "output_refusal", "refusal":
					if c.Refusal != "" {
						refusal = c.Refusal
					}
				}
			}
		case "reasoning":
			for _, s := range item.Summary {
				if s.Text != "" {
					reasoningParts = append(reasoningParts, s.Text)
				}
			}
			// Preserve the reasoning item structurally (id + encrypted_content),
			// not just its flattened summary text, so a subsequent Chat
			// Completions turn can reconstruct a proper "reasoning" input item
			// ahead of any function_call it informed -- see
			// chatAssistantMessageToInputItems. This whole feature always runs
			// Responses API in the stateless/store:false mode (a fresh
			// /v1/responses call per Chat Completions turn, full input
			// rebuilt from the client's message history each time -- see
			// ChatRequestToResponses), and OpenAI's own guidance for reasoning
			// models doing multi-round function calling in that mode is that
			// the reasoning item must be echoed back alongside its
			// function_call/function_call_output pair on the next turn, or
			// the model loses the chain of thought that produced the call
			// (degraded quality at best, a rejected request at worst).
			// reasoningParts/message.ReasoningContent above stays purely for
			// human/debug visibility -- it is not what gets fed back.
			if item.ID != "" || item.EncryptedContent != "" || len(item.Summary) > 0 {
				thinkingBlocks = append(thinkingBlocks, responsesReasoningBlock{
					Type:             responsesReasoningBlockType,
					ID:               item.ID,
					EncryptedContent: item.EncryptedContent,
					Summary:          item.Summary,
				})
			}
		case "function_call":
			hasFunctionCall = true
			// item.CallID (not item.ID) becomes the Chat Completions tool_call
			// "id": that's the value the client will echo back as
			// tool_call_id on its next "tool" role message, and
			// chatMessagesToInput reads it back out under the same key
			// (call_id) to build the matching function_call_output item --
			// keeping a multi-turn tool-use conversation correlated across
			// the round trip.
			toolCalls = append(toolCalls, openai.OpenAIToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: openai.OpenAIToolFunction{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		case "image_generation_call":
			if item.Result != "" {
				images = append(images, openai.ImageData{
					Type:    "image_url",
					B64JSON: item.Result,
				})
			}
			// web_search_call, file_search_call, code_interpreter_call, etc.:
			// hosted tool calls with no Chat Completions message-visible
			// equivalent. Their token cost is already reflected in
			// resp.Usage.ServerToolUse (see responsesUsageToChat) if the
			// provider reports one; nothing further to surface here.
		}
	}

	message := openai.OpenAIResponseMessage{
		Role:    "assistant",
		Content: strings.Join(textParts, "\n\n"),
	}
	if refusal != "" {
		message.Refusal = refusal
	}
	if len(reasoningParts) > 0 {
		message.ReasoningContent = strings.Join(reasoningParts, "\n\n")
	}
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}
	if len(images) > 0 {
		message.Images = images
	}

	// A response that failed outright (status="failed" inside an otherwise
	// 2xx HTTP response -- the Responses API's async-style error shape) has
	// no output items at all; surface the embedded error text as content
	// instead of handing back an empty message with no explanation.
	if resp.Status == "failed" && message.Content == "" && len(toolCalls) == 0 {
		if msg := responsesErrorMessage(resp.Error); msg != "" {
			message.Content = msg
		}
	}

	openAIResp := openai.OpenAIResponse{
		ID:      chatCompletionIDFromResponses(resp.ID),
		Object:  "chat.completion",
		Created: resp.CreatedAt,
		Model:   resp.Model,
		Choices: []openai.OpenAIChoice{
			{
				Index:        0,
				Message:      message,
				FinishReason: responsesFinishReason(resp.Status, resp.IncompleteDetails, hasFunctionCall),
			},
		},
		Usage: responsesUsageToChat(resp.Usage),
	}

	result, err := json.Marshal(openAIResp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal chat completions response: %w", err)
	}
	if len(thinkingBlocks) > 0 {
		if patched, err := injectThinkingBlocks(result, thinkingBlocks); err == nil {
			result = patched
		}
		// On a patch failure, fall through and return the unpatched result --
		// losing reasoning continuity is a quality regression, not a reason
		// to fail the whole response.
	}
	return result, nil
}

// responsesErrorMessage pulls "message" out of a Response.Error. Shared with
// the streaming transformer so failed responses look the same either way.
func responsesErrorMessage(err interface{}) string {
	errMap, ok := err.(map[string]interface{})
	if !ok {
		return ""
	}
	msg, _ := errMap["message"].(string)
	return msg
}

// responsesReasoningBlockType discriminates this package's own
// thinking_blocks convention from openai.OpenAIThinkingBlock's
// Anthropic-flavored entries (see chatAssistantMessageToInputItems) -- the
// two must never be confused if a single conversation somehow mixes routes.
const responsesReasoningBlockType = "responses_reasoning"

// responsesReasoningBlock is this round trip's own thinking_blocks entry
// shape (see injectThinkingBlocks / chatAssistantMessageToInputItems). Chat
// Completions has no native field for an opaque, must-echo-back reasoning
// item, so -- following the same convention this codebase already uses for
// Anthropic's thinking+signature blocks (openai.OpenAIMessage.ThinkingBlocks,
// see anthropic.convertOpenAIMessagesToAnthropic) -- it rides along as an
// extra "thinking_blocks" field on the assistant message that a
// well-behaved client preserves without understanding it.
type responsesReasoningBlock struct {
	Type             string          `json:"type"`
	ID               string          `json:"id,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`
	Summary          []OutputContent `json:"summary,omitempty"`
}

// injectThinkingBlocks patches "thinking_blocks" onto choices[0].message in
// an already-marshaled Chat Completions response body. openai.
// OpenAIResponseMessage (the response-side message type) has no
// ThinkingBlocks field -- that field only exists on openai.OpenAIMessage,
// the *request*-side type, since it's normally something a client echoes
// back rather than something AIR's own response builders populate. Patching
// the raw JSON here avoids widening a struct shared by every converter in
// the codebase just for this one round trip.
func injectThinkingBlocks(body []byte, blocks []responsesReasoningBlock) ([]byte, error) {
	var resp map[string]interface{}
	if err := json.Unmarshal(body, &resp); err != nil {
		return body, err
	}
	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return body, nil
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return body, nil
	}
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return body, nil
	}
	message["thinking_blocks"] = blocks
	return json.Marshal(resp)
}

// chatCompletionIDFromResponses exposes a Chat-Completions-shaped response
// ID while keeping the upstream Responses API response ID in the value, for
// request correlation -- same convention as
// anthropic.openAIChatCompletionID/vertex's ID handling.
func chatCompletionIDFromResponses(responseID string) string {
	if responseID == "" {
		return converterutil.GenerateID()
	}
	if strings.HasPrefix(responseID, "chatcmpl-") {
		return responseID
	}
	return "chatcmpl-" + responseID
}

// responsesFinishReason maps a Responses API response's status/
// incomplete_details (and whether it produced any function_call output
// item) to a Chat Completions finish_reason. Mirrors ChatToResponse's own
// status derivation (response.go) in reverse, plus the same "tool_calls"
// override every other provider converter in this codebase already applies
// on a tool call (see anthropic.AnthropicToOpenAI / vertex.VertexToOpenAI).
func responsesFinishReason(status string, incomplete *IncompleteDetails, hasFunctionCall bool) string {
	if hasFunctionCall {
		return "tool_calls"
	}
	if status == "incomplete" && incomplete != nil {
		switch incomplete.Reason {
		case "max_output_tokens":
			return "length"
		case "content_filter":
			return "content_filter"
		}
	}
	return "stop"
}

// responsesUsageToChat converts a Responses API Usage into the Chat
// Completions OpenAIUsage shape. Mirrors ChatToResponse's usage conversion
// (response.go) in reverse.
func responsesUsageToChat(usage *Usage) *openai.OpenAIUsage {
	if usage == nil {
		return nil
	}
	result := &openai.OpenAIUsage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.TotalTokens,
	}

	d := usage.InputTokensDetails
	if d.CachedTokens > 0 || d.CachedAudioTokens > 0 || d.CacheCreationTokens > 0 || d.CacheWriteTokens > 0 || d.AudioTokens > 0 {
		result.PromptTokensDetails = &openai.TokenDetails{
			CachedTokens:        d.CachedTokens,
			CachedAudioTokens:   d.CachedAudioTokens,
			CacheCreationTokens: d.CacheCreationTokens,
			CacheWriteTokens:    d.CacheWriteTokens,
			AudioTokens:         d.AudioTokens,
		}
		if cc := d.CacheCreationTokenDetails; cc != nil {
			result.PromptTokensDetails.CacheCreationTokenDetails = &openai.CacheCreationTokenDetails{
				Ephemeral5mInputTokens: cc.Ephemeral5mInputTokens,
				Ephemeral1hInputTokens: cc.Ephemeral1hInputTokens,
			}
		}
	}

	od := usage.OutputTokensDetails
	if od.ReasoningTokens > 0 || od.AudioTokens > 0 || od.ImageTokens > 0 {
		result.CompletionTokensDetails = &openai.CompletionTokenDetails{
			ReasoningTokens: od.ReasoningTokens,
			AudioTokens:     od.AudioTokens,
			ImageTokens:     od.ImageTokens,
		}
	}

	if usage.ServerToolUse != nil && usage.ServerToolUse.WebSearchRequests > 0 {
		result.ServerToolUse = &openai.ServerToolUseDetails{
			WebSearchRequests: usage.ServerToolUse.WebSearchRequests,
		}
	}

	return result
}
