package proxy

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// moonshotToolState is the adapter's state for one client request: the tools
// the model was given, what the tools returned, and what the model has
// already been told.
//
// The counters are the billing evidence: only a successful, non-empty
// Search, Search Pro or Fetch is charged, and a tool's result is charged
// once, however many times the same result is re-read (a replayed SSE frame,
// a retry of the same step).
type moonshotToolState struct {
	// searchCalls counts every Basic Search REST call the adapter made.
	searchCalls int
	// searchProCalls counts every Pro Search REST call the adapter made.
	searchProCalls int
	// fetchCalls counts every Fetch REST call the adapter made.
	fetchCalls int
	// billableSearch counts the successful, non-empty Basic searches.
	billableSearch int
	// billableSearchPro counts the successful, non-empty Pro searches.
	billableSearchPro int
	// billableFetch counts the successful fetches that returned content.
	billableFetch int
	// toolResults maps a tool call id to the result already produced for it,
	// so a replayed step does not charge the same call twice.
	toolResults map[string]string
}

// newMoonshotToolState builds an empty state.
func newMoonshotToolState() *moonshotToolState {
	return &moonshotToolState{toolResults: make(map[string]string)}
}

// moonshotToolRunner executes the adapter's tool calls and feeds the results
// back to the model.
type moonshotToolRunner struct {
	// client talks to Moonshot's Search, Search Pro and Fetch endpoints.
	client *moonshottools.Client
	// logger records the tool calls the model asked for.
	logger *slog.Logger
	// quota admits the calls against Moonshot's per-account QPS limits. It
	// may be nil, in which case the calls are not limited.
	quota *moonshottools.Quota
	// account identifies the Moonshot account the calls belong to, for the
	// per-account QPS pools.
	account string
	// state carries the billing counters across the steps.
	state *moonshotToolState
}

// runSearch executes one Basic search and returns the tool result to send
// back to the model.
//
// A failed search is not a billable event: the provider's error is handed to
// the model as the tool result, so the model can retry or answer without the
// router inventing a price.
func (runner *moonshotToolRunner) runSearch(ctx context.Context, arguments string) string {
	args, err := moonshottools.ParseSearchArguments(arguments)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	if message, blocked := runner.admit(ctx, runner.quotaAllowSearch); blocked {
		return message
	}
	runner.state.searchCalls++
	runner.logToolCall(ctx, "search", map[string]any{
		"text_query": args.QueryText(), "limit": args.Limit,
		"timeout_seconds": args.TimeoutSeconds, "include_content": args.WantContent(),
	})
	status, result, raw, err := runner.client.Search(ctx, args.ToSearchRequest())
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	if status < 200 || status >= 300 {
		return string(moonshottools.EncodeToolError(providerErrorMessage(raw)))
	}
	if len(result.Results) == 0 {
		// An empty result set is not billable: only a request that produced
		// a result is charged.
		return string(moonshottools.EncodeToolError("the search returned no results"))
	}
	runner.state.billableSearch++
	encoded, err := moonshottools.EncodeSearchResult(&result)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	return string(encoded)
}

// runSearchPro executes one Pro search and returns the tool result to send
// back to the model.
//
// The Pro search is billed on the same terms as the Basic one: a
// successful call that found something is charged, an empty result set or a
// failure is not.
func (runner *moonshotToolRunner) runSearchPro(ctx context.Context, arguments string) string {
	args, err := moonshottools.ParseSearchProArguments(arguments)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	if message, blocked := runner.admit(ctx, runner.quotaAllowSearchPro); blocked {
		return message
	}
	runner.state.searchProCalls++
	runner.logToolCall(ctx, "search_pro", map[string]any{
		"text_query": args.QueryText(), "limit": args.Limit,
		"timeout_seconds": args.TimeoutSeconds, "include_content": args.WantContent(),
		"sites": args.Sites, "time_window": args.TimeWindow,
	})
	status, result, raw, err := runner.client.SearchPro(ctx, args.ToSearchProRequest())
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	if status < 200 || status >= 300 {
		return string(moonshottools.EncodeToolError(providerErrorMessage(raw)))
	}
	if len(result.Results) == 0 {
		return string(moonshottools.EncodeToolError("the search returned no results"))
	}
	runner.state.billableSearchPro++
	encoded, err := moonshottools.EncodeSearchResult(&result)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	return string(encoded)
}

// runFetch executes one fetch and returns the tool result to send back to
// the model.
func (runner *moonshotToolRunner) runFetch(ctx context.Context, arguments string) string {
	args, err := moonshottools.ParseFetchArguments(arguments)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	if message, blocked := runner.admit(ctx, runner.quotaAllowFetch); blocked {
		return message
	}
	runner.state.fetchCalls++
	runner.logToolCall(ctx, "fetch", map[string]any{"url": args.URL})
	status, result, raw, err := runner.client.Fetch(ctx, args.ToFetchRequest())
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	if status < 200 || status >= 300 {
		return string(moonshottools.EncodeToolError(providerErrorMessage(raw)))
	}
	if result.Error != "" {
		// The provider answered 200 with its own error code (for instance
		// "markdown_not_found"). It served nothing, so it is not billable.
		return string(moonshottools.EncodeToolError(result.Error))
	}
	if strings.TrimSpace(result.Markdown) == "" {
		// A fetch is billed only when it produced page content.
		return string(moonshottools.EncodeToolError("the page had no fetchable content"))
	}
	runner.state.billableFetch++
	encoded, err := moonshottools.EncodeFetchResult(&result)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error()))
	}
	return string(encoded)
}

// logToolCall records the tool call the model asked for, with the arguments
// it chose: the query it wants searched, the page it wants fetched. It is
// what makes the router's own calls to Moonshot visible in the router's logs.
func (runner *moonshotToolRunner) logToolCall(ctx context.Context, tool string, args map[string]any) {
	if runner.logger == nil {
		return
	}
	runner.logger.DebugContext(ctx, "The model asked for a tool the router runs",
		"tool", tool, "arguments", args)
}

// quotaAllowSearch admits one Basic search against the account's Search pool.
func (runner *moonshotToolRunner) quotaAllowSearch(ctx context.Context) (bool, time.Duration, error) {
	if runner.quota == nil {
		return true, 0, nil
	}
	return runner.quota.AllowSearch(ctx, runner.account)
}

// quotaAllowSearchPro admits one Pro search against the account's Pro pool.
func (runner *moonshotToolRunner) quotaAllowSearchPro(ctx context.Context) (bool, time.Duration, error) {
	if runner.quota == nil {
		return true, 0, nil
	}
	return runner.quota.AllowSearchPro(ctx, runner.account)
}

// quotaAllowFetch admits one fetch against the account's Fetch pool.
func (runner *moonshotToolRunner) quotaAllowFetch(ctx context.Context) (bool, time.Duration, error) {
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
func (runner *moonshotToolRunner) admit(ctx context.Context, allow func(context.Context) (bool, time.Duration, error)) (string, bool) {
	admitted, wait, err := allow(ctx)
	if err != nil {
		return string(moonshottools.EncodeToolError(err.Error())), true
	}
	if admitted {
		return "", false
	}
	message := fmt.Sprintf("the tool is rate limited; retry in %s", wait.Round(time.Millisecond))
	return string(moonshottools.EncodeToolError(message)), true
}

// moonshotToolsEnabledFor reports whether the router should run the search
// tools itself for this request.
//
// It is an explicit opt-in on the credential (moonshot_tools), not a guess
// from the base URL: the router must not start calling a provider's
// endpoints because its address happens to look like Moonshot's.
func moonshotToolsEnabledFor(modelID string, cred *config.CredentialConfig) bool {
	if cred == nil || !cred.MoonshotTools {
		return false
	}
	return isMoonshotModel(modelID)
}

// describeMoonshotToolPlan renders the plan for a log line.
func describeMoonshotToolPlan(plan *moonshottools.ToolPlan) string {
	if plan == nil {
		return "none"
	}
	return fmt.Sprintf("search=%v search_pro=%v fetch=%v",
		plan.NeedsSearch, plan.NeedsSearchPro, plan.NeedsFetch)
}

// isMoonshotModel reports whether the model is one of Moonshot's own (a
// kimi model), which is what the router's search adapter serves.
func isMoonshotModel(modelID string) bool {
	lowered := strings.ToLower(modelID)
	return strings.Contains(lowered, "kimi") || strings.Contains(lowered, "moonshot")
}

// moonshotAccountKey names the Moonshot account the calls belong to, for
// the per-account QPS pools.
//
// The account is the credential's name: Moonshot counts its limits per
// account, and one credential here is one account. The key is namespaced by
// the credential type so that two providers' credentials of the same name
// cannot share a pool.
func moonshotAccountKey(cred *config.CredentialConfig) string {
	if cred == nil {
		return ""
	}
	return string(cred.Type) + ":" + cred.Name
}
