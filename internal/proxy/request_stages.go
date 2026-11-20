package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
	"github.com/mixaill76/auto_ai_router/internal/security"
)

// proxyRelay is everything the proxy/AIR response path needs in order to
// write a response and its accounting. The values are read-only here: the
// request state itself (credential, body, model ids) is owned by proxyRequest.
type proxyRelay struct {
	prepared       *orchestratedRequest
	resp           *ProxyResponse
	cred           *config.CredentialConfig
	modelID        string
	realModelID    string
	body           []byte
	proxyBody      []byte
	saveResponseFn func(*responses.Response)
	start          time.Time
	logCtx         *RequestLogContext
}

// relayProxyResponse writes a proxy/AIR response to the client: the streaming
// formats, the non-streaming body, and the per-request accounting that goes
// with it. Every path ends the request.
func (p *Proxy) relayProxyResponse(w http.ResponseWriter, r *http.Request, rel *proxyRelay) {
	prepared := rel.prepared
	proxyResp := rel.resp
	cred := rel.cred
	modelID := rel.modelID
	realModelID := rel.realModelID
	body := rel.body
	proxyBody := rel.proxyBody
	saveResponseFn := rel.saveResponseFn
	start := rel.start
	logCtx := rel.logCtx

	// Write response (streaming or non-streaming)
	tokenUsageOptions := tokenUsageExtractionOptionsForResponse(cred, proxyResp.Headers)
	// Populated in the non-streaming branch below via extractOpenAITokensAndUsage
	// (plan item G — one shared decode of proxyResp.Body instead of the
	// rate-limiter's extractTokensFromResponse and the spend-logging
	// converter.ExtractTokenUsageWithOptions each re-decoding it independently)
	// and consumed further down in the shared !proxyResp.IsStreaming logging block.
	var nonStreamingTokenUsage *converter.TokenUsage
	if proxyResp.IsStreaming {
		p.logger.DebugContext(r.Context(), "Response is streaming (no retry for streaming)",
			"credential", cred.Name, "status", proxyResp.StatusCode)
		streamCompleted := false

		// A non-2xx response from a proxy-like credential is not a stream to
		// relay -- every format-specific branch below (convertedMessages,
		// passthroughResponses, convertedResp, and the generic writer they
		// fall through to) is written for a 2xx body, and none of them masks
		// an error: a production incident showed exactly this (a credential's
		// raw 400 body -- its own nonstandard error shape -- reaching the
		// client byte-for-byte because nothing here ever classified it, the
		// same gap handlePassthroughResponsesStreaming had for the
		// Codex-passthrough branch specifically, fixed separately). Read the
		// body and classify it the same way the non-streaming path
		// (statusCodeFromProviderBodyError + writeProxyResponse, above) does,
		// instead of falling into any of the branches below.
		if proxyResp.StatusCode >= http.StatusBadRequest {
			body, readErr := p.readLimitedResponseBody(proxyResp.StreamBody)
			_ = proxyResp.StreamBody.Close()
			if readErr != nil {
				body = nil
			}
			masked := maskedUpstreamErrorBody(proxyResp.StatusCode, logCtx.RequestID, body)
			p.logUpstreamError(r.Context(), "Proxy request completed with error status (streaming)", proxyResp.StatusCode, cred, modelID, body,
				"url", security.RedactURLForLog(cred.BaseURL),
				"request_id", logCtx.RequestID)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(masked)))
			dropRepresentationIntegrityHeaders(w.Header())
			w.WriteHeader(proxyResp.StatusCode)
			_, _ = w.Write(masked)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = proxyResp.StatusCode
			logCtx.ErrorMsg = classifiedErrorMessage(proxyResp.StatusCode, body)
			logCtx.ErrorBodyRaw = extractErrorBodyRaw(body)
			logCtx.ClientResponseBody = extractErrorBodyRaw(masked)
			logCtx.TargetURL = cred.BaseURL
			p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, proxyResp.StatusCode, time.Since(start))
			return
		}

		if prepared.convertedMessages && proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
			defer func() {
				if closeErr := proxyResp.StreamBody.Close(); closeErr != nil {
					p.logger.WarnContext(r.Context(), "Failed to close proxy streaming response body", "error", closeErr)
				}
			}()
			p.copyResponseHeaders(w, proxyResp.Headers, cred, true)
			setSuccessfulSSEHeaders(w.Header(), proxyResp.StatusCode)
			if logCtx.IsProxyRequest && logCtx.ActualCredentialName != "" {
				w.Header().Set("X-Credential-Name", logCtx.ActualCredentialName)
			}
			p.setPromptTokensEstimate(logCtx, body, realModelID)
			fakeResp := &http.Response{
				StatusCode: proxyResp.StatusCode,
				Header:     proxyResp.Headers,
				Body:       proxyResp.StreamBody,
			}
			if err := p.handleMessagesAPIStreaming(w, fakeResp, cred, realModelID, logCtx, prepared.messagesMetadata, prepared.passthroughMessages); err != nil {
				p.logStreamHandlerError(r.Context(), "Failed to handle proxy Messages API streaming", err,
					"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
			} else {
				streamCompleted = true
			}
		} else if prepared.convertedResp {
			// Proxy streaming + Responses API: need to convert Chat Completions SSE
			// to Responses API SSE. Wrap StreamBody in http.Response for handleResponsesAPIStreaming.
			defer func() {
				if closeErr := proxyResp.StreamBody.Close(); closeErr != nil {
					p.logger.WarnContext(r.Context(), "Failed to close proxy streaming response body", "error", closeErr)
				}
			}()
			p.copyResponseHeaders(w, proxyResp.Headers, cred, false)
			p.setCredentialResponseHeader(w, logCtx, logCtx.ActualCredentialName)
			if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				p.markAudioUsageExcludesCachedForClient(w, logCtx)
			}
			p.setPromptTokensEstimate(logCtx, body, realModelID)
			fakeResp := &http.Response{
				StatusCode: proxyResp.StatusCode,
				Header:     proxyResp.Headers,
				Body:       proxyResp.StreamBody,
			}
			err := p.handleResponsesAPIStreaming(w, fakeResp, cred, realModelID, logCtx, saveResponseFn, prepared.responsesMetadata)
			if err != nil {
				p.logStreamHandlerError(r.Context(), "Failed to handle proxy Responses API streaming", err,
					"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
			} else if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				streamCompleted = true
			}
		} else if prepared.passthroughResponses {
			// Codex passthrough: provider returns native Responses API SSE — stream as-is.
			defer func() {
				if closeErr := proxyResp.StreamBody.Close(); closeErr != nil {
					p.logger.WarnContext(r.Context(), "Failed to close proxy streaming response body", "error", closeErr)
				}
			}()
			p.copyResponseHeaders(w, proxyResp.Headers, cred, false)
			p.setCredentialResponseHeader(w, logCtx, logCtx.ActualCredentialName)
			if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				p.markAudioUsageContractForClient(w, logCtx, tokenUsageOptions.AudioInputIncludesCachedAudio)
			}
			p.setPromptTokensEstimate(logCtx, body, realModelID)
			fakeResp := &http.Response{
				StatusCode: proxyResp.StatusCode,
				Header:     proxyResp.Headers,
				Body:       proxyResp.StreamBody,
			}
			if err := p.handlePassthroughResponsesStreaming(w, fakeResp, cred.Name, realModelID, logCtx, saveResponseFn, tokenUsageOptions); err != nil {
				p.logStreamHandlerError(r.Context(), "Failed to handle proxy passthrough Responses API streaming", err,
					"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
			} else if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				streamCompleted = true
			}
		} else {
			p.setCredentialResponseHeader(w, logCtx, logCtx.ActualCredentialName)
			tokenizerModelID := realModelID
			if tokenizerModelID == "" {
				tokenizerModelID = modelID
			}
			p.setPromptTokensEstimate(logCtx, proxyBody, tokenizerModelID)
			streamUsage, err := p.writeProxyStreamingResponseWithTokens(
				w, proxyResp, r, cred, modelID, tokenizerModelID, logCtx, tokenUsageOptions,
			)
			if err != nil {
				p.logStreamHandlerError(r.Context(), "Failed to write streaming proxy response", err,
					"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
			} else if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				streamCompleted = true
			}
			if streamUsage != nil {
				// Backfill PromptTokens from estimate when provider didn't include it
				// (e.g. stream cut before usage chunk, or provider omits prompt tokens).
				if streamUsage.PromptTokens == 0 && logCtx.UsageSource != "provider" {
					streamUsage.PromptTokens = logCtx.promptTokensEstimate()
				}
				logCtx.TokenUsage = streamUsage
				if proxyResp.StatusCode < 400 {
					p.metrics.RecordTokenUsage(cred.Name, modelID,
						streamUsage.PromptTokens, streamUsage.CompletionTokens,
						streamUsage.ReasoningTokens, streamUsage.CachedInputTokens)
					totalTokens := streamUsage.Total()
					if totalTokens > 0 {
						p.rateLimiter.ConsumeTokens(cred.Name, totalTokens)
						if modelID != "" {
							p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, totalTokens)
						}
					}
					p.logger.DebugContext(r.Context(), "Proxy streaming token usage recorded",
						"credential", cred.Name, "model", modelID,
						"prompt_tokens", streamUsage.PromptTokens,
						"completion_tokens", streamUsage.CompletionTokens)
				}
			}
		}
		if streamCompleted {
			logCtx.RequestCompleted = true
			p.bindSessionAfterSuccess(logCtx, modelID, cred.Name)
		}
	} else {
		// Save passthrough Responses API response or convert Chat Completions response if needed
		if prepared.convertedMessages && proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
			messagesBody, convErr := anthropicconv.ChatToMessages(proxyResp.Body, prepared.messagesMetadata,
				tokenUsageOptions.CacheWriteTTLHeader5mTokens, tokenUsageOptions.CacheWriteTTLHeader1hTokens)
			if convErr != nil {
				p.logger.ErrorContext(r.Context(), "Failed to convert proxy response to Messages API format",
					"credential", cred.Name, "model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
			} else {
				proxyResp.Body = messagesBody
			}
		} else if prepared.passthroughResponses && proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
			// Codex passthrough: body is already in Responses API format — just enrich and save.
			var respObj responses.Response
			if err := json.Unmarshal(proxyResp.Body, &respObj); err == nil {
				applyResponsesMetadata(&respObj, prepared.responsesMetadata)
				if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
					proxyResp.Body = enriched
				}
				if saveResponseFn != nil {
					saveResponseFn(&respObj)
				}
			}
		} else if prepared.convertedResp && proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
			responsesBody, convErr := responses.ChatToResponse(
				proxyResp.Body,
				responses.WithAudioInputIncludesCachedAudio(tokenUsageOptions.AudioInputIncludesCachedAudio),
			)
			if convErr != nil {
				p.logger.ErrorContext(r.Context(), "Failed to convert proxy response to Responses API format",
					"credential", cred.Name, "model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
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
					// Fallback: save unenriched on unmarshal failure (shouldn't happen)
					var r2 responses.Response
					if json.Unmarshal(responsesBody, &r2) == nil {
						saveResponseFn(&r2)
					}
				}
				proxyResp.Body = responsesBody
				tokenUsageOptions.AudioInputIncludesCachedAudio = false
			}
		}

		if proxyResp.StatusCode >= http.StatusOK && proxyResp.StatusCode < http.StatusMultipleChoices {
			modelIDs := []string{modelID, logCtx.PublicModelID, logCtx.RealModelID}
			for _, usageModelID := range modelIDs {
				if normalizedBody, changed := modelutils.NormalizeCompletionUsage(proxyResp.Body, usageModelID); changed {
					proxyResp.Body = normalizedBody
					break
				}
			}
		}

		p.setCredentialResponseHeader(w, logCtx, logCtx.ActualCredentialName)
		p.writeProxyResponse(w, proxyResp, r, cred, modelID, logCtx, tokenUsageOptions)
		tokens, usage := extractOpenAITokensAndUsage(proxyResp.Body, tokenUsageOptions)
		tokens, usage = p.systemOneUsage(r.Context(), logCtx, cred.Name, proxyResp.StatusCode, proxyBody, realModelID, tokens, usage)
		nonStreamingTokenUsage = usage
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(cred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
			}
			p.logger.DebugContext(r.Context(), "Proxy token usage recorded",
				"credential", cred.Name, "model", modelID, "tokens", tokens)
		}
		if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
			logCtx.RequestCompleted = true
			p.bindSessionAfterSuccess(logCtx, modelID, cred.Name)
		}
	}

	// Log proxy response.
	// For streaming, the write path above may already have classified this
	// request as a terminal stream error or client abort (body-detected
	// provider error remapped mid-stream, or the client disconnecting) and
	// set logCtx.Status/HTTPStatus/ErrorMsg accordingly — via
	// markProxyProviderStreamError / markStreamFailure / finalizeStreamingLog,
	// all reached from the streaming handlers called above. proxyResp.StatusCode
	// itself is never mutated for streaming (only the non-streaming branch
	// remaps it), so blindly overwriting here would silently revert an
	// already-correct 429/500/499 back to the original upstream 200.
	streamTerminalErrorAlreadyLogged := proxyResp.IsStreaming &&
		(logCtx.StreamOutcome == "stream_error" || logCtx.StreamOutcome == "client_aborted")
	if !streamTerminalErrorAlreadyLogged {
		logCtx.Status = "success"
		if proxyResp.StatusCode >= 400 {
			logCtx.Status = "failure"
			// Final error returned to the client — single unified ERROR record.
			// For streaming responses the body was forwarded to the client and is
			// not available here (response_body is omitted).
			p.logUpstreamError(r.Context(), "Proxy request completed with error status", proxyResp.StatusCode, cred, modelID, proxyResp.Body,
				"url", cred.BaseURL,
				"streaming", proxyResp.IsStreaming,
				"actual_credential", logCtx.ActualCredentialName,
				"request_id", logCtx.RequestID)
		}
		logCtx.HTTPStatus = proxyResp.StatusCode
	}
	logCtx.TargetURL = cred.BaseURL

	// Client-facing outcome decided (this response — success or the last
	// retryable error with no more attempts left — is what gets written to
	// the client; for streaming, logCtx.HTTPStatus reflects any body/SSE
	// remap applied during the write above) — record exactly once here
	// with genuine end-to-end duration, not once per retry attempt.
	p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, logCtx.HTTPStatus, time.Since(start))

	if !proxyResp.IsStreaming {
		// Reuses the decode already done above (extractOpenAITokensAndUsage)
		// instead of a second independent converter.ExtractTokenUsageWithOptions
		// Unmarshal of the same proxyResp.Body.
		logCtx.TokenUsage = nonStreamingTokenUsage
		if logCtx.TokenUsage != nil && proxyResp.StatusCode < 400 {
			p.metrics.RecordTokenUsage(cred.Name, modelID,
				logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens,
				logCtx.TokenUsage.ReasoningTokens, logCtx.TokenUsage.CachedInputTokens)
		}
		// Image responses are billed per image, not by token usage: record what
		// the response delivered for finalizeImageUsage.
		if logCtx.IsImageGeneration && proxyResp.StatusCode < 400 {
			logCtx.observeImageResponseBody(proxyResp.Body)
		}
		if proxyResp.StatusCode >= 400 {
			logCtx.ErrorMsg = extractErrorMessage(proxyResp.Body)
			logCtx.ErrorBodyRaw = extractErrorBodyRaw(proxyResp.Body)
		}
	}
	return
}
