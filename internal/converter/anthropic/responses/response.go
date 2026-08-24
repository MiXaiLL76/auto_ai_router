package anthropicresponses

import (
	"encoding/json"
	"fmt"
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
					Type:   "web_search_call",
					ID:     responses.GenerateItemID("ws_"),
					Status: "completed",
					Action: webSearchActionFromInput(block.Input),
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
// citations into Responses API url_citation annotations.
//
// Anthropic's cited_text is an excerpt of the *source* page (up to 150 chars),
// not a substring of Claude's own response text — the official example has a
// text block "Claude Shannon was born on April 30, 1916, in Petoskey,
// Michigan" cited by cited_text "Claude Elwood Shannon (April 30, 1916 –
// February 24, 2001) was an American mathematician...", which never occurs
// verbatim in the response. Anthropic also cites at whole-block granularity
// (a text block is the minimal citable unit, not a substring within it), so
// each citation is annotated over the entire block rather than a located
// range.
func webSearchCitationsToAnnotations(text string, citations []anthropic.AnthropicCitation) []responses.Annotation {
	annotations := []responses.Annotation{}
	textLen := len(text)
	for _, c := range citations {
		if c.Type != "web_search_result_location" || c.URL == "" {
			continue
		}
		annotations = append(annotations, responses.Annotation{
			Type:       "url_citation",
			URL:        c.URL,
			Title:      c.Title,
			StartIndex: 0,
			EndIndex:   textLen,
		})
	}
	return annotations
}

// webSearchActionFromInput extracts the search query from a server_tool_use
// block's input (e.g. {"query": "..."}) into the {"type":"search","query":...}
// shape the Responses API's web_search_call.action field expects. Anthropic
// only ever provides one query per server_tool_use call.
func webSearchActionFromInput(input interface{}) interface{} {
	m, ok := input.(map[string]interface{})
	if !ok {
		return nil
	}
	q, ok := m["query"].(string)
	if !ok || q == "" {
		return nil
	}
	return map[string]interface{}{
		"type":  "search",
		"query": q,
	}
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
	case "pause_turn":
		// A long-running server-tool turn (e.g. an extended web_search loop)
		// was paused mid-flight, not finished. Per Anthropic's docs, the
		// paused assistant content must be sent back unchanged to continue —
		// reporting "completed" here would tell the client the answer is
		// final when more server-tool work remains.
		return "incomplete", &responses.IncompleteDetails{Reason: "pause_turn"}
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
