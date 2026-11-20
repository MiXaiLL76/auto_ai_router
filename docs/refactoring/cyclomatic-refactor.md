# Cyclomatic complexity refactor: `proxyRequest` and `Config.Validate`

Status: proposed
Scope: `internal/proxy/proxy.go`, `internal/config/config.go`
Baseline commit: `dad0d1d` (main)

## 1. Why

The `gocyclo` gate added in #288 reports 156 functions above the
`min-complexity: 15` threshold set in `.golangci.yml`. Thirty of them were
silenced with `//nolint:gocyclo`; the rest are unguarded, so the gate cannot
pass. This document covers the two largest offenders and the extraction plan
for them. Every other function keeps its current status until those two land,
because they account for the bulk of the risk: `proxyRequest` is the request
lifecycle itself, and `Validate` is the only guard between a bad config and a
running router.

| Complexity | Function                | Location                              | Lines |
| ---------: | ----------------------- | ------------------------------------- | ----: |
|        394 | `(*Proxy).proxyRequest` | `internal/proxy/proxy.go:2128-4274`   |  2147 |
|        147 | `(*Config).Validate`    | `internal/config/config.go:2678-3109` |   432 |

Both functions are also the reason their files are unwieldy: `proxy.go` is
4434 lines, `config.go` is 3109.

## 2. Goals and non-goals

Goals:

- no function in either file above the complexity threshold of 15;
- no behavioural change — same status codes, error strings, log fields,
  headers, metrics, billing side effects and mutation order;
- every extracted unit testable on its own;
- each phase mergeable independently.

Non-goals:

- changing retry/fallback policy, routing, billing or validation rules;
- redesigning `RequestLogContext` or the converter interfaces;
- removing the existing `//nolint:gocyclo` suppressions (that is a separate
  ratchet step, see §7);
- touching `internal/proxy` code outside `proxyRequest`'s call tree.

## 3. Invariants (must hold after every phase)

1. **Retry semantics.** `max_provider_retries` counts extra credentials of any
   type; a proxy-like failure can hand the next attempt to a direct credential
   and vice versa, and the attempt counter continues across that switch. The
   existing `goto dispatch` behaviour is a contract, not an accident.
2. **Error mapping.** Client-visible status/body is unchanged:
   `statusCodeFromProviderBodyError`, `maskedUpstreamErrorBody`,
   `WriteErrorBadGateway` / `WriteErrorTimeout` / `WriteErrorClientClosed`,
   429 + `Retry-After` when the cascade was rate-limited, 502 only for a
   genuine transport failure with no response to relay.
3. **Ordering of side effects.** Spend logging, budget reconciliation, session
   binding, metrics (`RecordRequest`, `RecordTokenUsage`) and attempt tracing
   fire once, in the same order, on the same paths.
4. **Log/telemetry fields.** `logCtx` is filled at the same points with the
   same values (`Status`, `HTTPStatus`, `ErrorMsg`, `ErrorOrigin`,
   `TargetURL`, `ActualCredentialName`, `TokenUsage`, stream outcome).
5. **Validation order and text.** `Validate` returns the *first* error in the
   current order, with byte-identical messages; normalization mutations
   (`response_compatibility`, `response_headers.mode`, `logging_level`,
   `default_models_rpm`, `read_timeout`, `idle_timeout`,
   `NormalizeFallbackPriority`) still happen only when no earlier check failed.

## 4. Workstream A — `Config.Validate` (147 → ≤ 15 per function)

The function is a straight sequence of independent checks, which makes this
the lowest-risk, highest-clarity refactor in the repository. It also matches
an existing convention: `Video.Validate`, `YandexSearch.Validate`,
`LoopGate.Validate`, `VisionFallback.Validate`,
`ValidateOrganizationPolicies` already live in their own files.

### A.1 Target layout

Move `Validate` and its new section methods into
`internal/config/validate.go`; put the per-credential check in
`internal/config/validate_credential.go`. `config.go` keeps the types, `Load`
and normalization.

```go
// validate.go
func (c *Config) Validate() error {
    for _, check := range []func() error{
        c.validateDelegated,
        c.validateServerEndpoint,
        c.validateServerResponseHeaders,
        c.validateServerTimeouts,
        c.validateServerLogging,
        c.validateServerAuth,
        c.validateServerLimits,
        c.validateFail2BanLimits,
        c.validateTopology,
        c.validateFail2BanRules,
        c.validateCredentials,
        c.validateModels,
        c.validateClientModelIDs,
        c.validateRedis,
        c.validateOTEL,
        c.validateLiteLLMDB,
        c.validateKafka,
        c.validateKafkaRawBodies,
        c.validateKafkaRouterResponses,
        c.validateSpendLogConsistency,
    } {
        if err := check(); err != nil {
            return err
        }
    }
    return nil
}
```

A slice of checks — rather than a hand-written sequence of `if err := …` —
keeps the driver at complexity 3 and makes the order explicit. It changes
nothing else: each entry returns the same error at the same position.

### A.2 Sections and their source ranges

| Method                          | Source lines |                                                                                                                                                                                       Contents | Est. complexity |
| ------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------: |
| `validateDelegated`             | 2679-2703    |                                                                sub-config validators, per-credential/model `loop_gate`, `validateLoopGateCombinations`, `vision_fallback.describe_model` check |               7 |
| `validateServerEndpoint`        | 2704-2723    |                                                                                              port, `max_body_size_mb`, `response_body_multiplier`, `response_compatibility` normalize/validate |               6 |
| `validateServerResponseHeaders` | 2725-2729    |                                                                                                                                                     `response_headers.mode` normalize/validate |               3 |
| `validateServerTimeouts`        | 2731-2744    |                                                                                                                            `request_timeout`, `model_prices_sync_interval`, `shutdown_timeout` |               5 |
| `validateServerLogging`         | 2746-2754    |                                                                                                                                                             `logging_level` validate + default |               3 |
| `validateServerAuth`            | 2756-2759    |                                                                                                                                                                                   `master_key` |               2 |
| `validateServerLimits`          | 2761-2802    | `default_models_rpm`, `read_timeout`, `anthropic_default_max_tokens`, `max_provider_retries`, `max_fallback_attempts`, `max_connection_lifetime`, `session_sticky_ttl_minutes`, `idle_timeout` |              10 |
| `validateFail2BanLimits`        | 2804-2806    |                                                                                                                                                                        `fail2ban.max_attempts` |               2 |
| `validateTopology`              | 2808-2815    |                                                                                                                               at least one credential unless LiteLLM DB, organization policies |               4 |
| `validateFail2BanRules`         | 2817-2835    |                                                                                                                                         duplicate `error_code_rules`, per-credential overrides |               5 |
| `validateCredentials`           | 2836-2968    |                                                                                                                                         loop → `cred.validate()` + `NormalizeFallbackPriority` |               3 |
| `validateModels`                | 2969-2974    |                                                                                                                                                                                   model weight |               3 |
| `validateClientModelIDs`        | 2975-3004    |                                                                                                                                            canonical IDs, duplicates, alias targets/collisions |               8 |
| `validateRedis`                 | 3006-3011    |                                                                                                                                                                         addresses when enabled |               3 |
| `validateOTEL`                  | 3013-3024    |                                                                                                                                                               protocol, endpoint, sample ratio |               5 |
| `validateLiteLLMDB`             | 3026-3046    |                                                                                                                                                            DSN, reservation knobs, log workers |               7 |
| `validateKafka`                 | 3048-3081    |                                                                                                                                                brokers, topic, queue/batch/flush/workers, SASL |               9 |
| `validateKafkaRawBodies`        | 3083-3093    |                                                                                                                                                             enablement, topic, topic collision |               4 |
| `validateKafkaRouterResponses`  | 3095-3102    |                                                                                                                                                            enablement, topic, topic collisions |               4 |
| `validateSpendLogConsistency`   | 3104-3108    |                                                                                                                                                  kafka required for `disable_spend_logs_write` |               3 |

Line ranges are for `dad0d1d`; the split is otherwise position-independent.

### A.3 Per-credential check

`validateCredentials` reduces to the loop:

```go
for i := range c.Credentials {
    cred := &c.Credentials[i]
    if err := cred.validate(i); err != nil {
        return err
    }
    cred.NormalizeFallbackPriority()
    c.Credentials[i].NormalizeFallbackPriority() // keep the current write path
}
```

`(*CredentialConfig).validate(index int) error` holds the 132-line body and is
further split by the same rule:

| Method                           | Contents                                                                                       | Est. complexity |
| -------------------------------- | ---------------------------------------------------------------------------------------------- | --------------: |
| `validate(index)`                | name required, `proxy_url`, `auth_type`, protocol flags, headers, then the two below           |               8 |
| `validateProviderRequirements()` | the `switch cred.Type` block (base_url/api_key/project/location/credentials_file per provider) |               8 |
| `validateLimits()`               | `rpm`, `tpm`, `weight`, `fallback_priority`, `priority` and their combinations                 |               7 |

`index` is only needed for the `"credential %d: name is required"` message;
keep that wording exactly.

### A.4 Verification

- `go test ./internal/config/...` — 21 test files, 6.2k lines, including
  table-driven `Validate` cases that assert messages; these are the
  characterization suite and must pass untouched.
- `go test ./... -run 'Config|Validate'`.
- Complexity check per function:
  `golangci-lint run --enable-only gocyclo ./internal/config/...` must report
  zero issues for `internal/config`.

## 5. Workstream B — `(*Proxy).proxyRequest` (394 → ≤ 15 per function)

### 5.1 Current shape

The body is one 2147-line function with five stages and one `goto`:

| Stage | Lines | Size | Branch tokens (`if/for/case/&&/||`) |
|\---|---:|---:|---:|
| Setup: IDs, AIR markers, denylist, `logCtx`, deferred logging, span attrs, `saveResponseFn`, request classification, compat routing, budget/limits admission, retry state | 2128-2339 | 212 | 20 |
| Proxy-like path: retry loop, fallback, transport errors, streaming write | 2340-2854 | 515 | 93 |
| Direct path: retry loop | 2857-3199 | 343 | 50 |
| Shared tail: fallback, no-response classification, credential realignment | 3200-3428 | 229 | 40 |
| "Process final response": non-streaming transform pipeline, headers, streaming dispatch, logging/billing | 3429-4274 | 846 | 160 |

Two facts drive the plan:

- **The tail is the biggest single block.** 846 lines, 160 branch tokens, of
  which ~360 lines are five near-identical streaming branches.
- **There are 15 call sites of streaming handlers** in the function
  (`handleMessagesAPIStreaming`, `handleResponsesAPIStreaming`,
  `handlePassthroughResponsesStreaming`, `handleNativeResponsesStreaming`,
  `handleChatFromResponsesStreaming`, `handleProviderStreaming`,
  `handleStreamingWithTokens`, `writeProxyStreamingResponseWithTokens`,
  `streamResponseBody`), each wrapped in its own try/report/`streamCompleted`
  boilerplate. The proxy path and the direct path call the *same* handlers with
  different argument lists (e.g. `handlePassthroughResponsesStreaming` gets 6
  arguments in one path and 7 in the other) — this has already drifted.

### 5.2 Phase 1 — mechanical stage extraction (no control-flow change)

Target: `proxyRequest` becomes a driver of ≤ 60 lines, each stage a method on
`*Proxy` in a new file next to `proxy.go`.

| New function                                                                                                     | Source lines |                                   Est. complexity |
| ---------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------: |
| `newRequestScope(r) *requestScope` (IDs, markers, denylist, `logCtx`, attempts context, span attrs)              | 2128-2195    |                                                10 |
| `p.classifyRequest(r, body, logCtx)` (image/websearch/embeddings detection, prompt estimate inputs)              | 2269-2284    |                                                12 |
| `p.serveProxyLike(w, r, prepared, scope) (handled bool)`                                                         | 2340-2854    |                                      60 → see 5.3 |
| `p.serveDirect(w, r, prepared, scope) (handled bool)`                                                            | 2857-3199    |                                      40 → see 5.4 |
| `p.finalizeNoResponse(w, r, scope, attemptState)` (shared tail: fallback, transport classification, 429/502/499) | 3200-3428    | 20 → 8 after splitting fallback vs classification |
| `p.relayResponse(w, r, prepared, scope, resp)`                                                                   | 3429-4274    |                                      80 → see 5.5 |

Mechanical rule for every phase: **move code, do not rewrite it**. Signatures
change, statements do not. This keeps the diff reviewable and the invariants
checkable by reading.

### 5.2.1 What Phase 1 delivered

Three goto-free stages were extracted, each ending the request, so each could
be moved verbatim:

| New function                   | File                      | Complexity now | Source    |
| ------------------------------ | ------------------------- | -------------: | --------- |
| `(*Proxy).newRequestScope`     | `request_scope.go`        |             12 | 2129-2176 |
| `(*Proxy).relayProxyResponse`  | `request_stages.go`       |             73 | 2497-2807 |
| `(*Proxy).relayDirectResponse` | `request_relay_direct.go` |            187 | 3084-3925 |

Measured effect: `proxyRequest` 394 → 133, `proxy.go` 4434 → 3268 lines.

A constraint the plan above underestimated: **the retry loops cannot be
extracted before `goto dispatch` is gone.** The proxy path, the direct path
and the shared attempt counter are one block joined by that jump, so
`serveProxyLike`/`serveDirect` require Phase 2 first — and Phase 2 itself
re-indents the remaining body (a `for` around both paths), which is why it is
cheaper after the response tails have moved out.

Revised order for Workstream B:

1. ~~extract the goto-free response tails~~ (done, this branch);
2. split the two relays along their own seams — the streaming plan table
   (§5.3) for both of them, then the non-streaming pipeline (§5.4);
3. convert `goto dispatch` to a `for` around the two paths, extract
   `serveProxyLike`/`serveDirect` with `attemptOutcome`/`retryState` (§5.3);
4. reduce the driver to the setup → orchestrate → dispatch → relay chain.

Steps 2 and 3 are independent and can land in either order; step 3 is easier
once step 2 has shrunk the relays it will call.

### 5.3 Phase 2 — retry loop → attempt result

`serveProxyLike` is still ~60 because the loop carries 6 mutable locals and
two fallback/error blocks. Replace the mutable set with one result value:

```go
type attemptOutcome struct {
    resp       *ProxyResponse   // or *upstreamResponse, see 5.6
    cred       *config.CredentialConfig
    retry      bool
    reason     RetryReason
    transport  error
    switchTo   pathKind // pathStay | pathProxy | pathDirect | pathStop
}
```

```go
func (p *Proxy) proxyAttempt(w, r, prepared, scope, state *retryState, attempt int) attemptOutcome
func (p *Proxy) directAttempt(...) attemptOutcome
func (p *Proxy) retryAfter(o attemptOutcome, state *retryState, attempt int, w, r, prepared, scope) (next pathKind, stop bool)
```

This is also where `goto dispatch` becomes data: the direct attempt returns
`switchTo: pathProxy` (and the proxy attempt returns `pathDirect`), and the
driver loop continues. The attempt counter and `retryStart` fold into
`retryState`:

```go
type retryState struct {
    attempt   int
    tried     map[string]bool
    attempted map[string]bool
    carry     *retryCarry   // priorRetryResp/Cred/RealModelID/PriceModelID/Price
}
```

Expected: `proxyAttempt` and `directAttempt` at ≤ 20, then split their
"all attempts exhausted" blocks into `finishProxyAttempts` /
`finishDirectAttempts` to reach ≤ 15.

### 5.4 Phase 3 — streaming dispatch → plan table

Both paths branch over `prepared` flags to pick a streaming handler. Encode
the choice as data, once, in one place:

```go
type streamPlan struct {
    name        string                    // used in the error log
    wanted      func(*orchestratedRequest) bool
    endpointOK  bool                      // whether 2xx is required for this plan
    markAudio   func(http.ResponseWriter) // audio contract header, may be nil
    run         func(http.ResponseWriter, *http.Response) error
}
```

`p.streamPlans(prepared, cred, logCtx, saveResponseFn, provResponses) []streamPlan`
returns the ordered candidates; `p.relayStream(...)` takes the first matching
plan and owns the repeated boilerplate (headers, `Content-Type: text/event-stream`,
`X-Credential-Name`, prompt-token estimate, vision injection, error reporting,
`streamCompleted`).

Result: one ~40-line relay instead of five ~35-line branches in the direct
path and four in the proxy path. The signature drift disappears because every
handler is adapted once, in the plan constructor.

### 5.5 Phase 4 — non-streaming pipeline and finalization

The 462-line non-streaming stretch (3429-3890) is a linear pipeline whose
steps are already separate helpers. Extract each step, in order:

| Step                                                                | Est. complexity |
| ------------------------------------------------------------------- | --------------: |
| `decodeUpstreamBody` (gzip, size limits, error body capture)        |               8 |
| `convertSuccessBody` (chat ⇄ responses, messages, model aliasing)   |              10 |
| `runDeepSeekToolLoop` (search/fetch loop)                           |              12 |
| `applyVisionFallback`                                               |               6 |
| `buildNativeResponsesBody`                                          |              10 |
| `extractUsageAndPrice` (token extraction, fan-out accounting, cost) |              12 |

The remaining piece — headers + `http.ResponseWriter` hand-off — becomes
`p.writeClientResponse`, and the logging/billing block becomes
`p.finalizeRequestLog` (spend log, budget reconciliation, metrics,
`annotateRequestSpan`). The deferred cleanup at the top of the function keeps
calling the same helpers; only the call sites move.

### 5.6 Deliberately not unified

The proxy path yields `*ProxyResponse` (already-read body, headers, stream
body), the direct path yields `*http.Response` plus the decoded body. Do not
force them into one type in this refactor: the conversion itself is where
retry and streaming regressions would hide. Two relay functions sharing the
plan table and the finalization helpers are enough; unifying the types can be
a follow-up if it ever pays for itself.

## 6. Sequencing and PRs

| PR  | Contents                                                                | Risk        | Est. effort |
| --- | ----------------------------------------------------------------------- | ----------- | ----------- |
| 1   | Workstream A (`Validate`) + this document                               | low         | 0.5 day     |
| 2   | Phase 1: setup + both response relays extracted (this branch)           | low-medium  | done        |
| 3   | Streaming plan table in both relays (removes ~350 lines of duplication) | medium      | 1 day       |
| 4   | Non-streaming pipeline + finalization in both relays                    | medium-high | 1-2 days    |
| 5   | Attempt result/driver, `goto` removal, `serveProxyLike`/`serveDirect`   | high        | 1-2 days    |

Rationale for the order: PRs 2-4 are local moves that preserve control flow
and can be reviewed statement by statement; the `goto` driver (PR 5) is the
only phase where the retry state machine itself changes, so it goes last, when
the surrounding code is already small and covered.

Each PR must keep `go test -race ./...` green and must not raise the number of
`gocyclo` findings (the target is strictly decreasing).

## 7. Baseline gate (separate decision)

With 156 functions above the threshold, a gate that fails on the whole backlog
cannot merge any improvement. Two options, independent of this refactor:

- **Ratchet (recommended).** `cmd/cycloreport -baseline cyclo.baseline -check`:
  fail only on functions that are new above the threshold or that grew past
  their recorded value. Every refactor deletes its lines from the baseline, so
  the trend is visible in the diff.
- **`--new-from-rev=origin/main`** in the lint invocation: two lines in CI, no
  new tooling, but no ratchet.

Without one of them this work can be measured locally (§8) but not enforced
in CI.

## 8. How to measure

```bash
export PATH=/usr/local/go/bin:$(go env GOPATH)/bin:$PATH
golangci-lint run --enable-only gocyclo --timeout=5m \
  --max-issues-per-linter=0 --max-same-issues=0 \
  --output.text.path /tmp/gocyclo.log ./...
go run ./cmd/cycloreport -top 25 < /tmp/gocyclo.json      # needs #288
```

Per-function check while refactoring:

```bash
golangci-lint run --enable-only gocyclo ./internal/proxy/... \
  | grep -E 'proxyRequest|serveProxyLike|relayStream|Validate'
```

## 9. Acceptance criteria

1. `golangci-lint run --enable-only gocyclo ./...` reports no issue for
   `internal/config`, and for `internal/proxy/proxy.go` reports only functions
   outside `proxyRequest`'s call tree (ideally none).
2. `(*Proxy).proxyRequest` ≤ 60 lines and ≤ 15 complexity; every extracted
   function ≤ 15 (split further if not).
3. No change to: HTTP status/body for any error path, retry/fallback attempt
   counts, log field names/values, response headers, spend-log rows, metric
   names/labels.
4. `go test -race ./...` green; coverage for `internal/config` and
   `internal/proxy` not lower than before the branch.
5. `Validate` returns the first error in the same order with the same message
   for every existing negative test case.

## 10. Risks

| Risk                       | Why                                           | Mitigation                                                                                                            |
| -------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Silent retry-policy change | `goto dispatch` + shared `attempt` counter    | Phase 5 last; attempt-trace assertions (`httputil.AttemptsFrom`) in tests; compare attempt counts on a fixture matrix |
| Billing drift              | `logCtx` mutations are spread across the body | Extract first, move calls unchanged; spend-log golden tests; kafka raw-body comparisons                               |
| Streaming regressions      | 15 handler call sites, per-format headers     | Plan table built from the existing branches one-for-one; SSE fixtures per format                                      |
| Validation message drift   | tests assert exact strings                    | move code verbatim, keep order, no `errors.Join`                                                                      |
| Review fatigue             | 2k-line function                              | one phase per PR, mechanical moves only, no reformatting in the same diff                                             |
