package anthropicresponses

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// AnthropicToResponsesResponse converts an Anthropic Messages API response body to a
// responses.Response, using displayModelID as the echoed model name.
func AnthropicToResponsesResponse(body []byte, displayModelID, responseID string, createdAt int64) (*responses.Response, error) {
	var ar anthropic.AnthropicResponse
	if err := json.Unmarshal(body, &ar); err != nil {
		return nil, fmt.Errorf("AnthropicToResponsesResponse: parse: %w", err)
	}
	if responseID == "" {
		responseID = responses.GenerateResponseID()
	}
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	if displayModelID == "" {
		displayModelID = ar.Model
	}
	return buildAnthropicResponse(&ar, displayModelID, responseID, createdAt), nil
}

func buildAnthropicResponse(
	ar *anthropic.AnthropicResponse,
	model, responseID string,
	createdAt int64,
) *responses.Response {
	status, incompleteDetails := anthropicStopReasonToStatus(ar.StopReason)
	output := anthropicContentToOutputItems(ar.Content)
	usage := anthropicUsageToUsage(ar.Usage)
	completedAt := createdAt
	return responses.BuildCompletedResponse(responses.CompletedResponseParams{
		ID:                responseID,
		Model:             model,
		CreatedAt:         createdAt,
		CompletedAt:       &completedAt,
		Status:            status,
		IncompleteDetails: incompleteDetails,
		Output:            output,
		Usage:             usage,
	})
}

// anthropicContentToOutputItems converts Anthropic content blocks to Responses API output items.
func anthropicContentToOutputItems(blocks []anthropic.ContentBlock) []responses.OutputItem {
	var output []responses.OutputItem
	var msgContent []responses.OutputContent

	flushMessage := func() {
		if len(msgContent) == 0 {
			return
		}
		output = append(output, responses.OutputItem{
			Type:    "message",
			ID:      responses.GenerateItemID("msg_"),
			Status:  "completed",
			Role:    "assistant",
			Content: msgContent,
		})
		msgContent = nil
	}

	for _, block := range blocks {
		switch block.Type {
		case "text":
			msgContent = append(msgContent, responses.OutputContent{
				Type:        "output_text",
				Text:        block.Text,
				Annotations: webSearchCitationsToAnnotations(block.Text, block.Citations),
			})

		case "server_tool_use":
			// Flush accumulated text before a search call.
			flushMessage()
			if block.Name == "web_search" {
				output = append(output, responses.OutputItem{
					Type:    "web_search_call",
					ID:      responses.GenerateItemID("ws_"),
					Status:  "completed",
					Queries: webSearchQueryFromInput(block.Input),
				})
			}
			// Other server tools (if any are added in future) have no
			// Responses API equivalent yet — skip them rather than guess.

			// web_search_tool_result blocks carry no separate Responses API
			// item (OpenAI's web_search_call doesn't expose raw results
			// either); their content surfaces via the text block's
			// citations above instead — nothing to do here.

		case "thinking":
			// Flush any accumulated text first.
			flushMessage()
			reasoningItem := responses.OutputItem{
				Type:   "reasoning",
				ID:     responses.GenerateItemID("rs_"),
				Status: "completed",
			}
			if block.Thinking != "" {
				reasoningItem.Summary = []responses.OutputContent{
					{Type: "summary_text", Text: block.Thinking},
				}
			}
			if block.Signature != "" {
				reasoningItem.EncryptedContent = block.Signature
			}
			output = append(output, reasoningItem)

		case "tool_use":
			// Flush accumulated text before a tool call.
			flushMessage()
			// Detect computer_use tool call by tool name.
			// The name "computer" is the canonical discriminator; the action-key
			// heuristic is kept as a fallback for non-standard names.
			isComputerCall := block.Name == "computer"
			if !isComputerCall {
				if inputMap, ok := block.Input.(map[string]interface{}); ok {
					_, isComputerCall = inputMap["action"]
				}
			}
			if isComputerCall {
				output = append(output, responses.OutputItem{
					Type:   "computer_call",
					ID:     responses.GenerateItemID("cc_"),
					Status: "completed",
					CallID: block.ID,
					Name:   block.Name,
					Action: block.Input,
				})
				continue
			}
			argsJSON := "{}"
			if block.Input != nil {
				if b, err := json.Marshal(block.Input); err == nil {
					argsJSON = string(b)
				}
			}
			output = append(output, responses.OutputItem{
				Type:      "function_call",
				ID:        responses.GenerateItemID("fc_"),
				Status:    "completed",
				CallID:    block.ID,
				Name:      block.Name,
				Arguments: argsJSON,
			})
		}
	}

	flushMessage()

	if len(output) == 0 {
		output = []responses.OutputItem{{
			Type:    "message",
			ID:      responses.GenerateItemID("msg_"),
			Status:  "completed",
			Role:    "assistant",
			Content: []responses.OutputContent{{Type: "output_text", Text: "", Annotations: []responses.Annotation{}}},
		}}
	}

	return output
}

// webSearchCitationsToAnnotations converts a text block's Anthropic web_search
// citations into Responses API url_citation annotations. Anthropic citations
// carry the quoted substring (cited_text) rather than character offsets, so
// offsets are recovered with a best-effort substring search against the
// block's own text; a citation whose cited_text can't be located (e.g. minor
// whitespace differences) is skipped rather than emitted with a wrong range.
//
// The search cursor only moves forward as citations are consumed (rather than
// always searching from the start of text) so that two citations quoting the
// same substring — e.g. the same phrase cited from two different places in
// the block — resolve to their own distinct occurrence instead of both
// collapsing onto the first match.
func webSearchCitationsToAnnotations(text string, citations []anthropic.AnthropicCitation) []responses.Annotation {
	annotations := []responses.Annotation{}
	searchFrom := 0
	for _, c := range citations {
		if c.Type != "web_search_result_location" || c.URL == "" || c.CitedText == "" {
			continue
		}
		rel := strings.Index(text[searchFrom:], c.CitedText)
		if rel < 0 {
			continue
		}
		idx := searchFrom + rel
		searchFrom = idx + len(c.CitedText)
		annotations = append(annotations, responses.Annotation{
			Type:       "url_citation",
			URL:        c.URL,
			Title:      c.Title,
			StartIndex: idx,
			EndIndex:   idx + len(c.CitedText),
		})
	}
	return annotations
}

// webSearchQueryFromInput extracts the search query from a server_tool_use
// block's input (e.g. {"query": "..."}) for the web_search_call item's
// Queries field.
func webSearchQueryFromInput(input interface{}) []string {
	m, ok := input.(map[string]interface{})
	if !ok {
		return nil
	}
	if q, ok := m["query"].(string); ok && q != "" {
		return []string{q}
	}
	return nil
}

// anthropicStopReasonToStatus maps Anthropic stop_reason to Responses API status.
func anthropicStopReasonToStatus(stopReason string) (string, *responses.IncompleteDetails) {
	switch stopReason {
	case "end_turn", "tool_use", "":
		return "completed", nil
	case "max_tokens":
		return "incomplete", &responses.IncompleteDetails{Reason: "max_output_tokens"}
	case "stop_sequence":
		return "completed", nil
	default:
		return "completed", nil
	}
}

// anthropicUsageToUsage converts Anthropic usage to responses.Usage.
func anthropicUsageToUsage(au *anthropic.AnthropicUsage) *responses.Usage {
	if au == nil {
		return nil
	}
	cacheCreationTokens, cacheCreation5mTokens, cacheCreation1hTokens := anthropic.NormalizeCacheCreationUsage(
		au.CacheCreationInputTokens, au.CacheCreation,
	)
	totalInputTokens := au.InputTokens + au.CacheReadInputTokens + cacheCreationTokens
	inputDetails := responses.InputDetails{
		CachedTokens:        au.CacheReadInputTokens,
		CacheCreationTokens: cacheCreationTokens,
	}
	if cacheCreation5mTokens > 0 || cacheCreation1hTokens > 0 {
		inputDetails.CacheCreationTokenDetails = &responses.CacheCreationTokenDetails{
			Ephemeral5mInputTokens: cacheCreation5mTokens,
			Ephemeral1hInputTokens: cacheCreation1hTokens,
		}
	}
	usage := &responses.Usage{
		InputTokens:         totalInputTokens,
		OutputTokens:        au.OutputTokens,
		TotalTokens:         totalInputTokens + au.OutputTokens,
		InputTokensDetails:  inputDetails,
		OutputTokensDetails: responses.OutputDetails{},
	}
	if au.ServerToolUse != nil && au.ServerToolUse.WebSearchRequests > 0 {
		usage.ServerToolUse = &responses.ServerToolUseDetails{
			WebSearchRequests: au.ServerToolUse.WebSearchRequests,
		}
	}
	return usage
}
