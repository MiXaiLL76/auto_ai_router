package proxy

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
	"github.com/mixaill76/auto_ai_router/internal/toolloop"
)

// deepSeekToolState is the adapter's state for one client request: what the
// tools returned and what the model has already been told.
//
// The counters are the billing evidence: only a successful, non-empty Search
// or Fetch is charged, and a tool's result is charged once, however many
// times the same result is re-read (a replayed SSE frame, a retry of the
// same step).
type deepSeekToolState struct {
	// searchCalls counts every Basic Search REST call the adapter made.
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
//
// It is the router's toolloop.Runner for DeepSeek: it owns the two tools
// DeepSeek runs, runs them, and counts the billable ones.
type deepSeekToolRunner struct {
	// client talks to DeepSeek's Search and Fetch endpoints.
	client *deepseektools.Client
	// logger records the tool calls the model asked for.
	logger *slog.Logger
	// quota admits the calls against DeepSeek's per-account QPS limits. It
	// may be nil, in which case the calls are not limited.
	quota *deepseektools.Quota
	// account identifies the DeepSeek account the calls belong to, for the
	// per-account QPS pools.
	account string
	// state carries the billing counters across the steps.
	state *deepSeekToolState
}

// Owns reports whether the tool is one the router runs for DeepSeek.
func (runner *deepSeekToolRunner) Owns(name string) bool {
	return name == deepseektools.SearchToolName || name == deepseektools.FetchToolName
}

// Billable counts the successful calls, per tool.
func (runner *deepSeekToolRunner) Billable() map[string]int {
	return map[string]int{
		deepseektools.SearchToolName: runner.state.billableSearch,
		deepseektools.FetchToolName:  runner.state.billableFetch,
	}
}

// Run executes one tool call and returns the result to send back to the
// model.
//
// A failed call is not a billable event: the provider's error is handed to
// the model as the tool result, so the model can retry or answer without the
// router inventing a price.
func (runner *deepSeekToolRunner) Run(ctx context.Context, name, arguments string) string {
	switch name {
	case deepseektools.SearchToolName:
		return runner.runSearch(ctx, arguments)
	case deepseektools.FetchToolName:
		return runner.runFetch(ctx, arguments)
	default:
		return string(deepseektools.EncodeToolError(fmt.Sprintf("unknown tool %q", name)))
	}
}

// runSearch executes one search and returns the tool result to send back to
// the model.
func (runner *deepSeekToolRunner) runSearch(ctx context.Context, arguments string) string {
	args, err := deepseektools.ParseSearchArguments(arguments)
	if err != nil {
		return string(deepseektools.EncodeToolError(err.Error()))
	}
	if message, blocked := runner.admit(ctx, runner.quotaAllowSearch); blocked {
		return message
	}
	runner.state.searchCalls++
	runner.logToolCall(ctx, "search", map[string]any{
		"query": args.Query, "count": args.Count, "need_content": args.NeedContent,
	})
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

// logToolCall records the tool call the model asked for, with the arguments
// it chose: the query it wants searched, the page it wants fetched. It is
// what makes the router's own calls to DeepSeek visible in the router's logs.
func (runner *deepSeekToolRunner) logToolCall(ctx context.Context, tool string, args map[string]any) {
	if runner.logger == nil {
		return
	}
	runner.logger.DebugContext(ctx, "The model asked for a tool the router runs",
		"tool", tool, "arguments", args)
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
	runner.logToolCall(ctx, "fetch", map[string]any{"url": args.URL})
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

// admit waits for the account's QPS pool to have room and reports whether the
// call may go ahead.
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
	message := fmt.Sprintf("the tool is rate limited; retry in %s", wait.Round(time.Millisecond))
	return string(deepseektools.EncodeToolError(message)), true
}

// compile-time check: the runner is the router's tool loop's runner.
var _ toolloop.Runner = (*deepSeekToolRunner)(nil)
