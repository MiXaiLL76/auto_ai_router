package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
	"github.com/mixaill76/auto_ai_router/internal/security"
)

// directRelay is everything the direct-provider response path needs in order
// to transform, write and account a response. The values are read-only: the
// request state itself is owned by proxyRequest.
type directRelay struct {
	prepared           *orchestratedRequest
	resp               *http.Response
	responseBody       []byte
	targetURL          string
	conv               *converter.ProviderConverter
	provResponses      responses.ProviderResponses
	cred               *config.CredentialConfig
	modelID            string
	realModelID        string
	streaming          bool
	isStreamingResp    bool
	isEmbeddings       bool
	isImageEdit        bool
	body               []byte
	saveResponseFn     func(*responses.Response)
	triedCreds         map[string]bool
	attemptedCreds     map[string]bool
	embedFanOutReplies embedContentFanOutReplies
	fannedOut          bool
	start              time.Time
	logCtx             *RequestLogContext
}

// relayDirectResponse processes and writes a direct-provider response: the
// body transformations (Responses/Chat conversions, tool loops, vision
// fallback), token accounting, the spend log, and the streaming or
// non-streaming write. Every path ends the request.
func (p *Proxy) relayDirectResponse(w http.ResponseWriter, r *http.Request, rel *directRelay) {
	prepared := rel.prepared
	resp := rel.resp
	responseBody := rel.responseBody
	targetURL := rel.targetURL
	conv := rel.conv
	provResponses := rel.provResponses
	cred := rel.cred
	modelID := rel.modelID
	realModelID := rel.realModelID
	streaming := rel.streaming
	isStreamingResp := rel.isStreamingResp
	isEmbeddings := rel.isEmbeddings
	isImageEdit := rel.isImageEdit
	body := rel.body
	saveResponseFn := rel.saveResponseFn
	triedCreds := rel.triedCreds
	attemptedCreds := rel.attemptedCreds
	embedFanOutReplies := rel.embedFanOutReplies
	fannedOut := rel.fannedOut
	start := rel.start
	logCtx := rel.logCtx

	// === Process final response ===
	var finalResponseBody []byte
	outgoingAudioUsageContract := audioUsageContractUnknown

	if isStreamingResp {
		// force_stream_for_ttft: the client asked for a non-streaming response,
		// but applyForceStream sent "stream": true upstream. Consume the SSE,
		// measure TTFT from the first content delta and reassemble the response
		// into the non-stream JSON the client expects.
		if p.forceStreamForTTFT && !streaming && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices && p.simulatedNonStreamSupported(prepared) {
			p.handleSimulatedNonStream(w, r, resp, prepared, cred, modelID, realModelID, logCtx, saveResponseFn, targetURL, body,
				streaming, isEmbeddings, isImageEdit, start, triedCreds, attemptedCreds)
			return
		}
		if prepared.forcedNativeNonStream && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices && provResponses != nil {
			// Native Responses (Anthropic, Vertex) forced to stream: consume the
			// provider SSE, fold it back into the single Responses JSON the
			// client asked for, and record TTFT on the way.
			p.handleSimulatedNativeNonStream(w, r, resp, prepared, cred, provResponses, modelID, realModelID, logCtx, saveResponseFn, targetURL, body,
				streaming, isEmbeddings, isImageEdit, start, triedCreds, attemptedCreds)
			return
		}
		if prepared.forcedPassthroughNonStream && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			// Codex/OpenAI passthrough Responses forced to stream: the upstream
			// SSE is already Responses-shaped, so the terminal response.completed
			// event is written back as non-stream JSON, with TTFT recorded.
			p.handleSimulatedPassthroughNonStream(w, r, resp, prepared, cred, modelID, realModelID, logCtx, saveResponseFn, targetURL, body,
				streaming, isEmbeddings, isImageEdit, start, triedCreds, attemptedCreds)
			return
		}
		p.logger.DebugContext(r.Context(), "Response is streaming", "credential", cred.Name)
	} else {
		// Decode the response body for logging (handles gzip, etc.)
		contentEncoding := resp.Header.Get("Content-Encoding")
		decodedBody := decodeResponseBody(responseBody, contentEncoding)
		// Do this before conversion/history storage as well as native Messages
		// passthrough: private refusal extensions must never become client output.
		if isAnthropicResponse(cred, realModelID, r.URL.Path) && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			status, clean, changed := sanitizeRefusalResponse(resp.StatusCode, []byte(decodedBody), logCtx)
			resp.StatusCode = status
			decodedBody = string(clean)
			if changed {
				dropRepresentationIntegrityHeaders(resp.Header)
			}
		}
		// deepSeekToolUsage carries the tool counters of the router-run
		// search/fetch loop. They are applied to the request's own usage
		// below, where the billing path reads it -- not through the body,
		// because a body that is converted between the API shapes (a
		// Responses request answered in Chat, or the other way round) loses
		// the counters on the way.
		var deepSeekToolUsage *converter.TokenUsage

		// The router-run DeepSeek Search/Fetch loop: the model asked for one of
		// the tools the router owns, so the router runs it and asks the model
		// again. The loop's steps are separate upstream requests, so their
		// usage is summed and carried into the final body -- otherwise the
		// client would be billed for the last step only.
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// The router-run tool loop: the model asked for one of the tools
			// the router owns, so the router runs it and asks the model
			// again. The loop's steps are separate upstream requests, so their
			// usage is summed and carried into the final body -- otherwise
			// the client would be billed for the last step only.
			//
			// The two adapters never overlap: each engages only for its own
			// provider's credential (deepseek_tools / moonshot_tools), so
			// exactly one of them can run for a request.
			// The two adapters never overlap: each engages only for its own
			// provider's credential (deepseek_tools / moonshot_tools), so
			// exactly one of them can run for a request. Both answer with the
			// same shape, so the dispatch is one block.
			var toolRun toolLoopRun
			var toolErr error
			if p.moonshotToolsEnabled && moonshotToolsEnabledFor(modelID, cred) {
				moonshotRun, moonshotErr := p.runMoonshotToolLoop(r, w, logCtx, prepared, cred, modelID, realModelID,
					body, []byte(decodedBody), start, attemptedCreds)
				toolErr = moonshotErr
				if moonshotRun != nil {
					toolRun = moonshotRun
				}
			} else {
				deepSeekRun, deepSeekErr := p.runDeepSeekToolLoop(r, w, logCtx, prepared, cred, modelID, realModelID,
					body, []byte(decodedBody), start, attemptedCreds)
				toolErr = deepSeekErr
				if deepSeekRun != nil {
					toolRun = deepSeekRun
				}
			}
			switch {
			case toolErr != nil:
				if errToolLoopAborted(toolErr) {
					return
				}
				p.logger.WarnContext(r.Context(), "The router-run search/fetch loop failed",
					"model", modelID, "credential", cred.Name, "error", toolErr,
					"request_id", logCtx.RequestID)
				// A tool that already ran and returned results is billed even
				// though the model that followed it failed, so the loop's
				// usage rides along here too.
				if toolRun != nil {
					deepSeekToolUsage = toolRun.usage()
				}
			case toolRun != nil:
				decodedBody = string(toolRun.body())
				deepSeekToolUsage = toolRun.usage()
				p.logger.DebugContext(r.Context(), "The router ran the model's search/fetch tools",
					"model", modelID, "credential", cred.Name,
					"search_calls", toolRun.searchCalls(),
					"search_pro_calls", toolRun.searchProCalls(),
					"fetch_calls", toolRun.fetchCalls())
			}
		}

		// Transform response to OpenAI format (only for successful responses).
		// For error responses (4xx/5xx) pass the provider body through unchanged.
		// nativeResponses path skips this step — provider→Responses API conversion
		// happens further down via provResponses.ResponseTo().
		tokenUsageOptions := tokenUsageExtractionOptionsForResponse(cred, resp.Header)
		if resp.StatusCode >= 200 && resp.StatusCode < 300 && !prepared.nativeResponses && conv != nil && !conv.IsPassthrough() {
			convertedBody, convErr := conv.ResponseTo([]byte(decodedBody))
			if convErr != nil {
				args := []any{
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID,
				}
				args = appendResponseBodyForLogs(args, cred, decodedBody)
				p.logger.ErrorContext(r.Context(), "Failed to transform provider response to OpenAI format", args...)
				finalResponseBody = []byte(decodedBody)
			} else {
				finalResponseBody = convertedBody
				tokenUsageOptions.AudioInputIncludesCachedAudio = false
				p.logTransformedResponse(r.Context(), cred.Name, string(cred.Type), finalResponseBody)
				if conv.EmbeddingUsageEstimated() {
					// Billed (in part) from a text-length estimate: media parts
					// of the inputs without usage are not in it.
					p.logger.WarnContext(r.Context(), "Embedding response lacked usageMetadata for some or all inputs, their prompt tokens estimated from the request text",
						"credential", cred.Name, "provider", string(cred.Type),
						"model", modelID, "request_id", logCtx.RequestID)
				}
			}
		} else {
			finalResponseBody = []byte(decodedBody)
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 && conv != nil && conv.IsPassthrough() {
			p.observeSystemOneReplyModel(r.Context(), logCtx, cred.Name, realModelID, finalResponseBody)
			finalResponseBody = rewriteResponseModelAlias(finalResponseBody, realModelID, modelID)
			// Before billing picks its body: the stripped searches are billed from usage.
			finalResponseBody = openai.StripServerToolCalls(finalResponseBody)
		}

		// Vision fallback: prepend the image descriptions to the answer while the body
		// is still in the upstream format; the conversions below carry them along.
		if inj := visionInjectionFor(logCtx, prepared, resp.StatusCode); inj != nil {
			if injected, ok := injectVisionIntoResponse(finalResponseBody, inj, prepared.passthroughResponses); ok {
				finalResponseBody = injected
				dropRepresentationIntegrityHeaders(resp.Header)
			}
		}

		// bodyForTokenExtraction is set to finalResponseBody now and may be updated
		// after Responses API conversion (for nativeResponses the raw provider body
		// uses a provider-specific format that ExtractTokenUsage cannot parse).
		bodyForTokenExtraction := finalResponseBody
		if len(bodyForTokenExtraction) == 0 {
			bodyForTokenExtraction = []byte(decodedBody)
		}

		// Handle Responses API response body.
		if prepared.passthroughMessages && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// The provider answered in the Messages API format already: return its body
			// as is, so thinking signatures, block order, server tool blocks and
			// input_transformations survive. Billing keeps reading the Chat form above.
			finalResponseBody = rewriteResponseModelAlias([]byte(decodedBody), realModelID, clientVisibleResponseModel(logCtx, modelID))
		} else if prepared.convertedMessages && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			messagesBody, convErr := anthropicconv.ChatToMessages(finalResponseBody, prepared.messagesMetadata,
				tokenUsageOptions.CacheWriteTTLHeader5mTokens, tokenUsageOptions.CacheWriteTTLHeader1hTokens)
			if convErr != nil {
				p.logger.ErrorContext(r.Context(), "Failed to convert to Messages API format",
					"credential", cred.Name, "model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
			} else {
				finalResponseBody = messagesBody
			}
		} else if prepared.nativeResponses && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// Native Responses converter: convert provider response → *responses.Response.
			nativeResp, convErr := provResponses.ResponseTo([]byte(decodedBody), modelID)
			if convErr != nil {
				args := []any{
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID,
				}
				args = appendResponseBodyForLogs(args, cred, decodedBody)
				p.logger.ErrorContext(r.Context(), "Failed to convert native Responses API response", args...)
				// finalResponseBody already holds decodedBody — return as-is
			} else {
				applyConvertedResponsesMetadata(nativeResp, prepared.responsesMetadata)
				if enriched, marshalErr := json.Marshal(nativeResp); marshalErr == nil {
					finalResponseBody = enriched
					// Use the converted Responses API body for token extraction:
					// the raw provider body (e.g. Vertex usageMetadata) is not
					// parseable by ExtractTokenUsage which expects OpenAI-compatible fields.
					bodyForTokenExtraction = finalResponseBody
					tokenUsageOptions.AudioInputIncludesCachedAudio = false
				}
				if saveResponseFn != nil {
					saveResponseFn(nativeResp)
				}
			}
		} else if prepared.passthroughResponses && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// Codex passthrough: body is already in Responses API format — enrich and optionally save.
			var respObj responses.Response
			if err := json.Unmarshal(finalResponseBody, &respObj); err == nil {
				applyResponsesMetadata(&respObj, prepared.responsesMetadata)
				if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
					finalResponseBody = enriched
				}
				if saveResponseFn != nil {
					saveResponseFn(&respObj)
				}
			}
		} else if prepared.convertedResp && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// Non-codex: convert Chat Completions response back to Responses API format.
			responsesBody, convErr := responses.ChatToResponse(
				finalResponseBody,
				responses.WithAudioInputIncludesCachedAudio(tokenUsageOptions.AudioInputIncludesCachedAudio),
			)
			if convErr != nil {
				p.logger.ErrorContext(r.Context(), "Failed to convert to Responses API format",
					"credential", cred.Name, "model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
				// fallback: use Chat Completions body
			} else {
				// Enrich the response with request-echoed fields (store, previous_response_id,
				// metadata, text) for both the client payload and the store record.
				var respObj responses.Response
				if err := json.Unmarshal(responsesBody, &respObj); err == nil {
					applyConvertedResponsesMetadata(&respObj, prepared.responsesMetadata)
					if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
						responsesBody = enriched
					}
					if saveResponseFn != nil {
						saveResponseFn(&respObj)
					}
				} else if saveResponseFn != nil {
					var r2 responses.Response
					if json.Unmarshal(responsesBody, &r2) == nil {
						saveResponseFn(&r2)
					}
				}
				finalResponseBody = responsesBody
				bodyForTokenExtraction = finalResponseBody
				tokenUsageOptions.AudioInputIncludesCachedAudio = false
			}
		} else if prepared.convertedToResponses && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// responses_only model: the upstream call itself was to
			// /v1/responses (see ChatRequestToResponses), so finalResponseBody
			// is currently Responses-API-shaped -- convert it back to Chat
			// Completions shape for the client, which called
			// /v1/chat/completions and expects that shape back.
			chatBody, convErr := responses.ResponseToChat(finalResponseBody)
			if convErr != nil {
				args := []any{
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID,
				}
				args = appendResponseBodyForLogs(args, cred, decodedBody)
				p.logger.ErrorContext(r.Context(), "Failed to convert Responses API response to Chat Completions format", args...)
				// finalResponseBody already holds the raw Responses-API body — return as-is.
			} else {
				finalResponseBody = chatBody
				bodyForTokenExtraction = finalResponseBody
				tokenUsageOptions.AudioInputIncludesCachedAudio = false
			}
		}

		if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			normalizedBody := normalizeSuccessfulResponseModel(
				finalResponseBody,
				endpointFromLogContext(logCtx),
				clientVisibleResponseModel(logCtx, modelID),
			)
			if !bytes.Equal(normalizedBody, finalResponseBody) {
				finalResponseBody = normalizedBody
				bodyForTokenExtraction = normalizedBody
				for key := range resp.Header {
					if isRepresentationIntegrityHeader(key) {
						resp.Header.Del(key)
					}
				}
			}
			if normalizedBody, changed := modelutils.NormalizeCompletionUsage(
				finalResponseBody,
				clientVisibleResponseModel(logCtx, modelID),
			); changed {
				finalResponseBody = normalizedBody
				bodyForTokenExtraction = normalizedBody
				dropRepresentationIntegrityHeaders(resp.Header)
			}
			// Ensure standard cache-write naming is present whenever it is only
			// known under an alternate spelling (e.g. Requesty's caching_tokens on
			// an OpenAI→OpenAI passthrough). Pricing jsonpaths match the standard
			// name, so without this cache writes are dropped from billing. No-op
			// when the standard count is already set.
			if normalizedBody, changed := modelutils.NormalizeCacheCreationUsage(finalResponseBody); changed {
				finalResponseBody = normalizedBody
				bodyForTokenExtraction = normalizedBody
				dropRepresentationIntegrityHeaders(resp.Header)
			}
		}

		if mappedStatus, ok := statusCodeFromProviderBodyError(resp.StatusCode, finalResponseBody); ok {
			resp.StatusCode = mappedStatus
		}
		rawErrorBody := finalResponseBody
		clientBody, clientBodyChanged, clientBodyMasked := clientResponseBodyForCredential(resp.StatusCode, finalResponseBody, cred, modelID, logCtx.RequestID)
		if clientBodyChanged {
			finalResponseBody = clientBody
			bodyForTokenExtraction = clientBody
		}
		if logCtx.HideWebSearchResults && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			if stripped, ok := stripWebSearchResults(finalResponseBody); ok {
				finalResponseBody = stripped
				dropRepresentationIntegrityHeaders(resp.Header)
			}
		}
		if clientBodyChanged || clientBodyMasked {
			dropRepresentationIntegrityHeaders(resp.Header)
		}
		if clientBodyMasked {
			resp.Header.Set("Content-Type", "application/json")
		}
		logCtx.captureClientResponseID(finalResponseBody)
		// The body the client is about to receive, captured for the
		// router-response write-path. Taken here, AFTER the transformations
		// and BEFORE the cost injection below, so the row shows what the
		// client got for the content -- the cost fields are re-derived from
		// the spend log and the row's own usage, and injecting them here
		// would make the body depend on a toggle that is off in most
		// deployments.
		p.captureRouterResponse(logCtx, finalResponseBody, isStreamingResp)

		// Single shared decode of bodyForTokenExtraction for both token-accounting
		// consumers (plan item G) — the rate-limiter's total-tokens count
		// (RPM/TPM accounting) and the spend-logging TokenUsage below used to
		// each run their own independent full-body Unmarshal. They compute
		// genuinely different numbers (see
		// converter.ExtractTotalTokensAndUsageWithOptions's doc comment), so
		// only the decode is shared, not the computation.
		// bodyForTokenExtraction is not mutated between here and its next use
		// below.
		tokens, directProviderTokenUsage := extractOpenAITokensAndUsage(bodyForTokenExtraction, tokenUsageOptions)
		tokens, directProviderTokenUsage = p.systemOneUsage(r.Context(), logCtx, cred.Name, resp.StatusCode, body, realModelID, tokens, directProviderTokenUsage)
		// A fan-out consumed its replies' tokens on the credentials that served
		// them, call by call.
		if tokens > 0 && !fannedOut {
			p.rateLimiter.ConsumeTokens(cred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
			}
			p.logger.DebugContext(r.Context(), "Token usage recorded",
				"credential", cred.Name, "model", modelID, "tokens", tokens)
		}

		if p.logger.Enabled(context.Background(), slog.LevelDebug) {
			if resp.StatusCode >= 400 && shouldMaskUpstreamErrors(cred) {
				p.logger.DebugContext(r.Context(), "Proxy response body masked",
					"credential", cred.Name,
					"content_encoding", contentEncoding,
					"status_code", resp.StatusCode)
			} else {
				p.logger.DebugContext(r.Context(), "Proxy response body",
					"credential", cred.Name, "content_encoding", contentEncoding,
					"body", logger.TruncateLongFields(string(finalResponseBody), 500))
			}
		}

		resp.Body = io.NopCloser(bytes.NewReader(finalResponseBody))

		// Log to LiteLLM DB (non-streaming)
		logCtx.TokenUsage = directProviderTokenUsage
		// The router-run tool loop's own counters (the billable searches and
		// fetches) and its summed token counts ride alongside the provider's
		// usage: the loop made several upstream requests, and the client is
		// charged for all of them.
		// The loop's usage is applied to the request's own usage here: the
		// spend log (written in the deferred call) reads it from there.
		if logCtx.TokenUsage == nil {
			logCtx.TokenUsage = &converter.TokenUsage{}
		}
		applyDeepSeekToolUsageTo(logCtx.TokenUsage, deepSeekToolUsage)
		logCtx.Status = "success"
		logCtx.HTTPStatus = resp.StatusCode
		logCtx.TargetURL = targetURL

		if resp.StatusCode >= 400 {
			logCtx.Status = "failure"
			// Same classification the client's own response body already went
			// through (clientResponseBodyForCredential -> maskedUpstreamErrorBody)
			// — keeps the spend-log/analytics record in sync with what the client
			// actually saw instead of a generic placeholder.
			logCtx.ErrorMsg = classifiedErrorMessage(resp.StatusCode, rawErrorBody)
			// This is the main non-streaming, non-retried direct-provider-error
			// path -- found missing its ErrorBodyRaw capture via an actual
			// end-to-end smoke test (docker-compose.kafka.yml): a request that
			// gets a real upstream error and isn't retried lands here, and
			// air.raw_bodies.response_body came back NULL for it despite the
			// other 4 capture sites all being covered.
			logCtx.ErrorBodyRaw = extractErrorBodyRaw(rawErrorBody)
			// clientBody is already the post-masking body computed above by
			// clientResponseBodyForCredential (line ~2154) -- this is the exact
			// bytes actually written to the client via resp.Body below, not a
			// second, possibly-drifting recomputation.
			logCtx.ClientResponseBody = extractErrorBodyRaw(clientBody)
			// Final error returned to the client — single unified ERROR record
			// with everything needed for debugging.
			p.logUpstreamError(r.Context(), "Upstream request completed with error status", resp.StatusCode, cred, modelID, rawErrorBody,
				"url", security.RedactURLForLog(targetURL),
				"request_id", logCtx.RequestID)
			p.billKeptEmbedFanOutReplies(r.Context(), logCtx, &embedFanOutReplies)
		} else if logCtx.TokenUsage != nil {
			p.metrics.RecordTokenUsage(cred.Name, modelID,
				logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens,
				logCtx.TokenUsage.ReasoningTokens, logCtx.TokenUsage.CachedInputTokens)
		}

		if logCtx.IsImageGeneration && resp.StatusCode < 400 {
			logCtx.observeImageResponseBody(bodyForTokenExtraction)
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			if tokenUsageOptions.AudioInputIncludesCachedAudio {
				outgoingAudioUsageContract = audioUsageContractIncludesCached
			} else {
				outgoingAudioUsageContract = audioUsageContractExcludesCached
			}
		}
		if logCtx.Token != "" && logCtx.Credential != nil {
			if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
				p.logger.WarnContext(r.Context(), "Failed to queue spend log",
					"error", err, "request_id", logCtx.RequestID)
			}
		}
		logCtx.Logged = true
		// logSpendToLiteLLMDB above just computed logCtx.responseCostUSD --
		// add it to the client's own usage object before it goes out, for
		// organizations that opted into response_cost_in_body (see
		// client_cost.go). Independent of setResponseCostHeader below, which
		// gates the header form on response_cost_mode instead.
		if logCtx.responseCostKnown && shouldExposeCostToClient(logCtx) {
			if withCost, ok := injectCostIntoJSONUsage(finalResponseBody, logCtx.responseCostUSD); ok {
				finalResponseBody = withCost
			}
		}
	}
	if isStreamingResp {
		outgoingAudioUsageContract = directStreamingAudioUsageContract(prepared, cred, resp.StatusCode)
	}

	// Copy response headers (skip hop-by-hop headers and transformation-related headers)
	p.copyResponseHeaders(w, resp.Header, cred, false)
	switch outgoingAudioUsageContract {
	case audioUsageContractExcludesCached:
		p.markAudioUsageExcludesCachedForClient(w, logCtx)
	case audioUsageContractIncludesCached:
		p.markAudioUsageIncludesCachedForClient(w, logCtx)
	}
	// Return credential name only to internal proxy clients, not to end users.
	p.setCredentialResponseHeader(w, logCtx, cred.Name)
	// Client-billed cost of this request. Only set when it is already known —
	// for a plain streaming response it is not, and the header is skipped.
	p.setResponseCostHeader(w, logCtx)

	rc := http.NewResponseController(w)

	if isStreamingResp {
		if resp.StatusCode >= 400 {
			// Error status on a streaming response — the body is forwarded to the
			// client as a stream and is not available here for logging.
			p.logUpstreamError(r.Context(), "Upstream returned error status on streaming response", resp.StatusCode, cred, modelID, nil,
				"url", security.RedactURLForLog(targetURL),
				"request_id", logCtx.RequestID)
			body := maskedUpstreamErrorBody(resp.StatusCode, logCtx.RequestID)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
			dropRepresentationIntegrityHeaders(w.Header())
			w.WriteHeader(resp.StatusCode)
			_, _ = w.Write(body)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = resp.StatusCode
			// No raw body captured here (streamed, not buffered) — same limitation
			// maskedUpstreamErrorBody above already has, so this stays consistent
			// with what the client just received.
			logCtx.ErrorMsg = classifiedErrorMessage(resp.StatusCode, nil)
			logCtx.TargetURL = targetURL
			return
		}
		setSuccessfulSSEHeaders(w.Header(), resp.StatusCode)

		// Arms the lazy fallback estimate; the actual json-decode + tiktoken pass only
		// runs later, inside finalizeStreamingLog, and only if the provider never sent
		// usage — logging it eagerly here would defeat that (DebugContext's args are
		// evaluated unconditionally regardless of whether debug level is enabled).
		p.setPromptTokensEstimate(logCtx, body, realModelID)

		p.logger.DebugContext(r.Context(), "Streaming handler selection",
			"is_responses_api", prepared.isResponsesAPI,
			"is_messages_api", prepared.isMessagesAPI,
			"converted_resp", prepared.convertedResp,
			"converted_messages", prepared.convertedMessages,
			"converted_to_responses", prepared.convertedToResponses,
			"provider", cred.Type,
			"model", modelID,
			"resp_content_type", resp.Header.Get("Content-Type"),
			"resp_status", resp.StatusCode)

		// Vision fallback: the image descriptions start the answer text. Injected into
		// the upstream stream, before any conversion to the client format.
		if inj := visionInjectionFor(logCtx, prepared, resp.StatusCode); inj != nil {
			resp.Body = newVisionStreamInjector(resp.Body, inj, prepared.passthroughResponses)
		}

		streamCompleted := false
		if prepared.convertedMessages {
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				err := p.handleMessagesAPIStreaming(w, resp, cred, modelID, logCtx, prepared.messagesMetadata, prepared.passthroughMessages)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle Messages API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else {
					streamCompleted = true
				}
			} else {
				err := p.handleProviderStreaming(w, resp, cred, realModelID, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle provider streaming response", err,
						"credential", cred.Name, "provider", cred.Type, "model", modelID, "request_id", logCtx.RequestID)
				}
			}
		} else if prepared.nativeResponses {
			// Native Responses converter (Vertex AI, Anthropic): convert provider SSE
			// directly to Responses API SSE via ProviderResponses.StreamTo().
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				err := p.handleNativeResponsesStreaming(w, resp, provResponses, modelID, logCtx, saveResponseFn, prepared.responsesMetadata)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle native Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream as-is
				err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle streaming response", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				}
			}
		} else if prepared.convertedResp {
			// For Responses API: only transform successful responses (2xx status).
			// For error responses (4xx/5xx), pass through the provider error as-is.
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				// The router-run tool loop, streaming variant: the model may
				// ask for one of the router's tools mid-stream, and the
				// client cannot be shown a tool call for a tool it never
				// declared. The stream is buffered, the tool run, and the
				// model asked again; the final step's stream is what the
				// client gets, converted to the Responses shape on the way
				// out.
				//
				// Without this the client received the model's own tool
				// call -- for the router's own function, which it cannot
				// run -- and no answer at all.
				responsesLoopRan := false
				if p.deepSeekToolsEnabled && deepSeekToolsEnabledFor(modelID, cred) ||
					p.moonshotToolsEnabled && moonshotToolsEnabledFor(modelID, cred) {
					var buffered []byte
					var loopUsage *converter.TokenUsage
					var loopErr error
					if moonshotToolsEnabledFor(modelID, cred) {
						moonshotRun, moonshotErr := p.runMoonshotStreamToolLoopBuffered(r, w, logCtx, prepared, cred,
							modelID, realModelID, body, resp, start, attemptedCreds)
						loopErr = moonshotErr
						if moonshotRun != nil {
							responsesLoopRan = true
							buffered = moonshotRun.body
							loopUsage = moonshotRun.usage
						}
					} else {
						deepSeekRun, deepSeekErr := p.runDeepSeekStreamToolLoopBuffered(r, w, logCtx, prepared, cred,
							modelID, realModelID, body, resp, start, attemptedCreds)
						loopErr = deepSeekErr
						if deepSeekRun != nil {
							responsesLoopRan = true
							buffered = deepSeekRun.body
							loopUsage = deepSeekRun.usage
						}
					}
					switch {
					case errToolLoopAborted(loopErr):
						// The client went away (or the provider's response was
						// already handled): the loop stops here, and the router
						// must not write anything else to the dead connection.
						return
					case loopErr != nil:
						p.logger.WarnContext(r.Context(), "The router-run search/fetch loop failed",
							"model", modelID, "credential", cred.Name, "error", loopErr,
							"request_id", logCtx.RequestID)
						// A tool that already ran and returned results is
						// billed even though the model that followed it
						// failed, so the loop's usage rides along here too.
						if responsesLoopRan && loopUsage != nil {
							logCtx.DeepSeekToolUsage = loopUsage
						}
					case responsesLoopRan:
						// The loop buffered the stream itself, so the usual
						// streaming handler never ran: it neither counted the
						// tokens nor wrote the spend row. The loop's usage is
						// applied to the request's own usage after the
						// streaming log wrote its figures (the log read the
						// last chunk's usage, which carries only the last
						// step's).
						logCtx.DeepSeekToolUsage = loopUsage
					}
					if responsesLoopRan && buffered != nil {
						// The final step's stream, in the provider's own Chat
						// Completions shape: the conversion to Responses SSE
						// happens below, as it does for a stream without a
						// tool call.
						resp.Body = io.NopCloser(bytes.NewReader(buffered))
						resp.ContentLength = int64(len(buffered))
					}
				}
				// Transform to Responses API SSE format
				err := p.handleResponsesAPIStreaming(w, resp, cred, modelID, logCtx, saveResponseFn, prepared.responsesMetadata)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
					// Note: finalizeStreamingLog inside handleTransformedStreaming already
					// logged the spend. We only update error metadata here for the defer
					// safety net, but don't reset Logged to avoid double logging.
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream using provider's native format instead
				err := p.handleProviderStreaming(w, resp, cred, realModelID, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle provider streaming response", err,
						"credential", cred.Name, "provider", cred.Type, "model", modelID, "request_id", logCtx.RequestID)
				} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					streamCompleted = true
				}
			}
		} else if prepared.passthroughResponses {
			// Codex passthrough: provider returns native Responses API SSE — forward as-is.
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				if err := p.handlePassthroughResponsesStreaming(w, resp, cred.Name, realModelID, logCtx, saveResponseFn); err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle passthrough Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream as-is
				err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle streaming response", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					streamCompleted = true
				}
			}
		} else if prepared.convertedToResponses {
			// responses_only model: upstream streams native Responses API SSE
			// (we called its /v1/responses) — convert back to Chat Completions
			// SSE for the client, which called /v1/chat/completions.
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				err := p.handleChatFromResponsesStreaming(w, resp, cred, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle responses_only streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream using provider's native format instead.
				err := p.handleProviderStreaming(w, resp, cred, realModelID, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle provider streaming response", err,
						"credential", cred.Name, "provider", cred.Type, "model", modelID, "request_id", logCtx.RequestID)
				} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					streamCompleted = true
				}
			}
		} else {
			// The router-run DeepSeek Search/Fetch loop, streaming variant: the
			// model may ask for one of the router's tools mid-stream, and the
			// client cannot be shown a tool call for a tool it never declared.
			// The stream is buffered, the tool run, and the model asked again.
			if resp.StatusCode >= 200 && resp.StatusCode < 300 &&
				((p.deepSeekToolsEnabled && deepSeekToolsEnabledFor(modelID, cred)) ||
					(p.moonshotToolsEnabled && moonshotToolsEnabledFor(modelID, cred))) {
				// The two adapters never overlap: each engages only for its
				// own provider's credential, so exactly one of them runs.
				var streamErr error
				var loopUsage *converter.TokenUsage
				var loopTotalTokens int
				var loopLastChunk []byte
				streamLoopRan := false
				if moonshotToolsEnabledFor(modelID, cred) {
					moonshotRun, moonshotErr := p.runMoonshotStreamToolLoop(r, w, logCtx, prepared, cred,
						modelID, realModelID, body, resp, start, attemptedCreds)
					streamErr = moonshotErr
					if moonshotRun != nil {
						streamLoopRan = true
						loopUsage = moonshotRun.usage
						loopTotalTokens = moonshotRun.totalTokens
						loopLastChunk = moonshotRun.lastChunk
					}
				} else {
					deepSeekRun, deepSeekErr := p.runDeepSeekStreamToolLoop(r, w, logCtx, prepared, cred,
						modelID, realModelID, body, resp, start, attemptedCreds)
					streamErr = deepSeekErr
					if deepSeekRun != nil {
						streamLoopRan = true
						loopUsage = deepSeekRun.usage
						loopTotalTokens = deepSeekRun.totalTokens
						loopLastChunk = deepSeekRun.lastChunk
					}
				}
				switch {
				case errToolLoopAborted(streamErr):
					// The client went away (or the provider's response was
					// already handled): the loop stops here, and the router
					// must not write anything else to the dead connection.
					return
				case streamErr != nil:
					p.logger.WarnContext(r.Context(), "The router-run search/fetch loop failed",
						"model", modelID, "credential", cred.Name, "error", streamErr,
						"request_id", logCtx.RequestID)
					// A tool that already ran and returned results is billed
					// even though the model that followed it failed, so the
					// loop's usage rides along here too.
					if streamLoopRan && loopUsage != nil {
						if logCtx.TokenUsage == nil {
							logCtx.TokenUsage = &converter.TokenUsage{}
						}
						applyDeepSeekToolUsageTo(logCtx.TokenUsage, loopUsage)
						logCtx.DeepSeekToolUsage = loopUsage
					}
					// The loop read the provider's stream whole (and closed it)
					// before it ran the tool, so the original response cannot
					// be relayed: reading it would give the client a 200 with
					// an empty stream and the log an "invalid Read on closed
					// Body" instead of the real failure.
					//
					// The client gets the failure instead.
					logCtx.Status = "failure"
					logCtx.HTTPStatus = http.StatusBadGateway
					logCtx.ErrorMsg = "the router-run tool loop failed"
					logCtx.StreamOutcome = "stream_error"
					WriteErrorBadGateway(w, "the upstream failed to answer after the router ran a tool")
					streamCompleted = true
				default:
					if streamLoopRan {
						// The loop buffered the stream itself, so the usual
						// streaming handler never ran: it neither counted the
						// tokens nor wrote the spend row. The loop does both
						// here -- and the streaming log reads its figures from
						// the last chunk, so the chunk carries the loop's
						// summed usage rather than the last step's.
						//
						// The loop's usage is handed to the log rather than
						// applied here: the log reads the chunk's own usage
						// afterwards, and only then does the loop's figure --
						// the sum of every step's -- replace it. Applied
						// before, the log's own reading would overwrite the
						// cached and reasoning tokens with the last step's.
						logCtx.DeepSeekToolUsage = loopUsage
						p.finalizeStreamingLog(logCtx, loopTotalTokens,
							withDeepSeekToolUsage(loopLastChunk, loopUsage),
							"openai", resp.StatusCode, false)
					}
					streamCompleted = true
				}
			} else {
				err := p.handleProviderStreaming(w, resp, cred, realModelID, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle provider streaming response", err,
						"credential", cred.Name, "provider", cred.Type, "model", modelID, "request_id", logCtx.RequestID)
				} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					streamCompleted = true
				}
			}
		}
		if streamCompleted {
			logCtx.RequestCompleted = true
			p.bindSessionAfterSuccess(logCtx, modelID, cred.Name)
		}

	} else {
		acceptEncoding := r.Header.Get("Accept-Encoding")
		acceptedEncodings := ParseAcceptEncoding(acceptEncoding)
		targetEncoding := SelectBestEncoding(acceptedEncodings)

		outputBody := finalResponseBody
		if targetEncoding != "identity" && len(finalResponseBody) > 0 {
			compressedBody, usedEncoding, compErr := CompressBody(finalResponseBody, targetEncoding)
			if compErr != nil {
				p.logger.WarnContext(r.Context(), "Failed to compress response body",
					"credential", cred.Name, "encoding", targetEncoding, "error", compErr)
			} else {
				p.logger.DebugContext(r.Context(), "Response body compressed for client",
					"credential", cred.Name, "encoding", usedEncoding,
					"original_size", len(finalResponseBody), "compressed_size", len(compressedBody))
				outputBody = compressedBody
				w.Header().Set("Content-Encoding", usedEncoding)
			}
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(outputBody)))

		_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		w.WriteHeader(resp.StatusCode)

		_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		if _, err := p.streamResponseBody(w, bytes.NewReader(outputBody)); err != nil {
			if isClientDisconnectError(err) {
				p.logger.DebugContext(r.Context(), "Client disconnected during response body copy", "error", err)
				p.recordAbortedRequest(cred.Name, r.URL.Path, modelID)
			} else {
				p.logger.ErrorContext(r.Context(), "Failed to copy response body", "error", err)
			}
		} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			logCtx.RequestCompleted = true
			p.bindSessionAfterSuccess(logCtx, modelID, cred.Name)
		}
	}
}
