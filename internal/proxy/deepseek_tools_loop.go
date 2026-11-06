package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
)

// maxDeepSeekToolSteps bounds the adapter's loop: the model may call the
// router-run tools repeatedly, and each round is another upstream request
// (and another bill). The cap keeps a model that keeps calling the tool from
// looping forever; the last step's answer is returned as it is.
const maxDeepSeekToolSteps = 8

// deepSeekToolState is the adapter's state for one client request: the tools
// the model was given, what the tools returned, and what the model has
// already been told.
//
// The counters are the billing evidence: only a successful, non-empty
// Search or Fetch is charged, and a tool's result is charged once, however
// many times the same result is re-read (a replayed SSE frame, a retry of
// the same step).
type deepSeekToolState struct {
	// searchCalls counts every Search REST call the adapter made.
	searchCalls int
	// fetchCalls counts every Fetch REST call the adapter made.
	fetchCalls int
	// billableSearch counts the successful, non-empty searches.
	billableSearch int
	// billableFetch counts the successful fetches that returned content.
	billableFetch int
	// toolResults maps a tool call id to the result already produced for it,
	// so a replayed step does not charge the same call twice.
	toolResults map[string]string
}

// newDeepSeekToolState builds an empty state.
func newDeepSeekToolState() *deepSeekToolState {
	return &deepSeekToolState{toolResults: make(map[string]string)}
}

// deepSeekToolRunner executes the adapter's tool calls and feeds the results
// back to the model.
type deepSeekToolRunner struct {
	// client talks to DeepSeek's Search and Fetch endpoints.
	client *deepseektools.Client
	// quota admits the calls against DeepSeek's per-account QPS limits. It
	// may be nil, in which case the calls are not limited.
	quota *deepseektools.Quota
	// account identifies the DeepSeek account the calls belong to, for the
	// per-account QPS pools.
	account string
	// state carries the billing counters across the steps.
	state *deepSeekToolState
}

// runSearch executes one search and returns the tool result to send back to
// the model.
//
// A failed search is not a billable event: the provider's error is handed to
// the model as the tool result, so the model can retry or answer without the
// router inventing a price.
func (runner *deepSeekToolRunner) runSearch(ctx context.Context, arguments string) string {
	args, err := deepseektools.ParseSearchArguments(arguments)
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	if message, blocked := runner.admit(ctx, runner.quotaAllowSearch); blocked {
		return message
	}
	runner.state.searchCalls++
	status, result, raw, err := runner.client.Search(ctx, deepseektools.SearchRequest(args))
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	if status < 200 || status >= 300 {
		return string(deepseektools.EncodeToolError(providerErrorMessage(raw)))
	}
	if len(result.Results) == 0 {
		// An empty result set is not billable (DeepSeek bills only a request
		// that produced a result).
		return string(deepseektools.EncodeToolError("the search returned no results"))
	}
	runner.state.billableSearch++
	encoded, err := deepseektools.EncodeSearchResult(result)
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	return string(encoded)
}

// runFetch executes one fetch and returns the tool result to send back to
// the model.
func (runner *deepSeekToolRunner) runFetch(ctx context.Context, arguments string) string {
	args, err := deepseektools.ParseFetchArguments(arguments)
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	if message, blocked := runner.admit(ctx, runner.quotaAllowFetch); blocked {
		return message
	}
	runner.state.fetchCalls++
	status, result, raw, err := runner.client.Fetch(ctx, deepseektools.FetchRequest(args))
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	if status < 200 || status >= 300 {
		return string(deepseektools.EncodeToolError(providerErrorMessage(raw)))
	}
	if strings.TrimSpace(result.Content) == "" {
		// DeepSeek bills a fetch only when it produced page content.
		return string(deepseektools.EncodeToolError("the page had no fetchable content"))
	}
	runner.state.billableFetch++
	encoded, err := deepseektools.EncodeFetchResult(result)
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	return string(encoded)
}

// quotaAllowSearch admits one search against the account's Search pool.
func (runner *deepSeekToolRunner) quotaAllowSearch(ctx context.Context) (bool, time.Duration, error) {
	if runner.quota == nil {
		return true, 0, nil
	}
	return runner.quota.AllowSearch(ctx, runner.account)
}

// quotaAllowFetch admits one fetch against the account's Fetch pool.
func (runner *deepSeekToolRunner) quotaAllowFetch(ctx context.Context) (bool, time.Duration, error) {
	if runner.quota == nil {
		return true, 0, nil
	}
	return runner.quota.AllowFetch(ctx, runner.account)
}

// admit waits for the account's QPS pool to have room and reports whether
// the call may go ahead.
//
// A call that could not be admitted is not billed: the provider counts only
// the calls it served, and a call the router never made is not one of them.
// The model is told the tool was rate-limited, so it can answer without it
// rather than the request failing outright.
func (runner *deepSeekToolRunner) admit(ctx context.Context, allow func(context.Context) (bool, time.Duration, error)) (string, bool) {
	admitted, wait, err := allow(ctx)
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error())), true
	}
	if admitted {
		return "", false
	}
	if wait > 0 {
		if waitErr := deepseektools.SleepWithContext(ctx, wait); waitErr != nil {
			return string(deepseektools.EncodeToolError(waitErr.Error())), true
		}
		// The pool may have room now; try once more.
		admitted, _, err = allow(ctx)
		if err != nil {
			return string(deepseektools.EncodeToolError(err.Error())), true
		}
		if admitted {
			return "", false
		}
	}
	return string(deepseektools.EncodeToolError("the search tool is rate limited, try again later")), true
}

// providerErrorMessage extracts a human-readable message from a provider's
// OpenAI-compatible error body.
func providerErrorMessage(raw []byte) string {
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error.Message != "" {
		return envelope.Error.Message
	}
	if trimmed := strings.TrimSpace(string(raw)); trimmed != "" {
		return trimmed
	}
	return "the provider returned an error"
}

// deepSeekToolCallsFromResponse reads the tool calls out of a Chat
// Completions response, keeping only the ones the adapter owns.
func deepSeekToolCallsFromResponse(body []byte) []openai.OpenAIToolCall {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil
	}
	if len(response.Choices) == 0 {
		return nil
	}
	var owned []openai.OpenAIToolCall
	for _, call := range response.Choices[0].Message.ToolCalls {
		if call.Function.Name == deepseektools.SearchToolName || call.Function.Name == deepseektools.FetchToolName {
			owned = append(owned, call)
		}
	}
	return owned
}

// appendToolResults adds the assistant's tool-call message and the tool
// result messages to the conversation, so the next upstream call sees what
// happened.
//
// The assistant message keeps the model's tool calls verbatim (the ids have
// to match) and its reasoning content, so the model's own state survives the
// round trip.
func appendToolResults(messages []map[string]any, assistant map[string]any, results []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(messages)+1+len(results))
	out = append(out, messages...)
	out = append(out, assistant)
	out = append(out, results...)
	return out
}

// assistantToolCallMessage builds the assistant message that carries the
// model's tool calls.
func assistantToolCallMessage(response openai.OpenAIResponse) map[string]any {
	message := map[string]any{"role": "assistant"}
	if len(response.Choices) == 0 {
		return message
	}
	choice := response.Choices[0]
	if choice.Message.Content != "" {
		message["content"] = choice.Message.Content
	}
	if choice.Message.ReasoningContent != "" {
		message["reasoning_content"] = choice.Message.ReasoningContent
	}
	calls := make([]map[string]any, 0, len(choice.Message.ToolCalls))
	for _, call := range choice.Message.ToolCalls {
		calls = append(calls, map[string]any{
			"id":   call.ID,
			"type": call.Type,
			"function": map[string]any{
				"name":      call.Function.Name,
				"arguments": call.Function.Arguments,
			},
		})
	}
	if len(calls) > 0 {
		message["tool_calls"] = calls
	}
	return message
}

// toolResultMessage builds the tool message carrying one result.
func toolResultMessage(toolCallID, name, content string) map[string]any {
	return map[string]any{
		"role":         "tool",
		"tool_call_id": toolCallID,
		"name":         name,
		"content":      content,
	}
}

// readAndCloseResponseBody reads a response body and closes it.
func readAndCloseResponseBody(resp *http.Response) ([]byte, error) {
	defer func() { _ = resp.Body.Close() }()
	return io.ReadAll(resp.Body)
}

// deepSeekAccountKey identifies the DeepSeek account a credential belongs
// to, for the per-account QPS pools.
//
// DeepSeek counts its Search and Fetch limits per account, not per key, so
// two credentials of the same account must share one pool. Credentials that
// share a base URL and key are the same account as far as the router can
// tell; the credential name is the fallback, so two credentials of one
// account that use different keys are (conservatively) pooled separately
// rather than wrongly sharing a budget.
func deepSeekAccountKey(cred *config.CredentialConfig) string {
	if cred == nil {
		return ""
	}
	base := strings.TrimSuffix(cred.BaseURL, "/")
	if cred.APIKey == "" {
		return cred.Name
	}
	return base + "|" + cred.APIKey
}

// deepSeekToolsEnabledFor reports whether the adapter should handle this
// request: the credential must be one whose provider runs its own Search and
// Fetch tools (the explicit deepseek_tools opt-in), and the model must be
// one that cannot run them itself.
func deepSeekToolsEnabledFor(modelID string, cred *config.CredentialConfig) bool {
	if cred == nil || !cred.DeepSeekTools {
		return false
	}
	return isDeepSeekModel(modelID)
}

// describeToolPlan renders the plan for a log line.
func describeToolPlan(plan *deepseektools.ToolPlan) string {
	if plan == nil {
		return "none"
	}
	return fmt.Sprintf("search=%v fetch=%v", plan.NeedsSearch, plan.NeedsFetch)
}
