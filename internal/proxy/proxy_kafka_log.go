package proxy

import (
	"context"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"go.opentelemetry.io/otel/trace"
)

// milliseconds is d in whole milliseconds, nil for nil.
func milliseconds(d *time.Duration) *int64 {
	if d == nil {
		return nil
	}
	ms := d.Milliseconds()
	return &ms
}

// traceID is the id of the trace that ctx's span belongs to, "" without one.
func traceID(ctx context.Context) string {
	if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
		return sc.TraceID().String()
	}
	return ""
}

// logSpendToKafka publishes an expanded copy of the spend entry to Kafka
// (internal/kafkalog) for downstream ClickHouse analytics. Best-effort in the
// sense that Kafka availability never affects request processing or blocks
// the caller beyond kafkalog's own bounded wait (kafkalog.Manager.IsHealthy
// reflects broker connectivity independently, see
// auto_ai_router_kafka_spend_log_tz.md section 6) — but the returned error is
// still surfaced to the caller (logSpendToLiteLLMDB) so a queue-full failure
// can be flagged on the request's Postgres row for later re-send, instead of
// being silently dropped.
func (p *Proxy) logSpendToKafka(
	logCtx *RequestLogContext,
	credName, modelIDFormatted, hashedToken string,
	userID, teamID, organizationID, endUser, apiBase, status string,
	cost float64,
	tokenCosts *converter.TokenCosts,
	overheadMs float64,
	endTime time.Time,
) error {
	event := p.buildKafkaSpendEvent(logCtx, credName, modelIDFormatted, hashedToken,
		userID, teamID, organizationID, endUser, apiBase, status,
		cost, tokenCosts, overheadMs, endTime)

	if err := p.kafkaLog.LogSpend(event); err != nil {
		p.logger.WarnContext(logCtx.Context(), "Failed to queue Kafka spend event",
			"error", err,
			"request_id", logCtx.RequestID,
		)
		return err
	}
	return nil
}

// buildKafkaSpendEvent maps a RequestLogContext plus the values already
// computed by logSpendToLiteLLMDB (cost, tokenCosts, end time, ...) onto the
// flat kafkalog.SpendEvent schema. Credential/server metadata is broken out
// into typed fields here instead of being folded into one JSON blob, as
// decided in the ТЗ (section 4) so ClickHouse can query it directly.
func (p *Proxy) buildKafkaSpendEvent(
	logCtx *RequestLogContext,
	_, modelIDFormatted, hashedToken string,
	userID, teamID, organizationID, endUser, apiBase, status string,
	cost float64,
	tokenCosts *converter.TokenCosts,
	overheadMs float64,
	endTime time.Time,
) *kafkalog.SpendEvent {
	usage := logCtx.TokenUsage
	if usage == nil {
		usage = &converter.TokenUsage{}
	} else {
		normalizedUsage := *usage
		usage = normalizedUsage.Normalize()
	}

	realModel := logCtx.RealModelID
	if realModel == "" {
		realModel = logCtx.ModelID
	}

	var completionStartTime *time.Time
	var ttftMs *int64
	if !logCtx.CompletionStartTime.IsZero() {
		cst := logCtx.CompletionStartTime
		completionStartTime = &cst
		ttft := cst.Sub(logCtx.StartTime).Milliseconds()
		ttftMs = &ttft
	}

	var upstreamSendMs *int64
	if !logCtx.UpstreamSendTime.IsZero() {
		us := logCtx.UpstreamSendTime.Sub(logCtx.StartTime).Milliseconds()
		upstreamSendMs = &us
	}

	attempts := httputil.AttemptsFrom(logCtx.Context())
	var retryMs, connWaitMs, ttfbMs, exitRTTMs, exitConnectMs *int64
	var connReused *bool
	if stages, ok := attempts.Stages(logCtx.Credential.Name); ok {
		retryMs = milliseconds(&stages.Retry)
		connWaitMs = milliseconds(stages.ConnWait)
		connReused = stages.Reused
		ttfbMs = milliseconds(stages.TTFB)
		exitRTTMs = milliseconds(stages.ExitRTT)
		exitConnectMs = milliseconds(stages.ExitConnect)
	}

	var keyAlias, userAlias, teamAlias, organizationAlias string
	if logCtx.TokenInfo != nil {
		keyAlias = logCtx.TokenInfo.KeyAlias
		userAlias = logCtx.TokenInfo.UserAlias
		teamAlias = logCtx.TokenInfo.TeamAlias
		organizationAlias = logCtx.TokenInfo.OrganizationAlias
	}

	event := &kafkalog.SpendEvent{
		RequestID:           logCtx.spendRequestID(),
		TraceID:             traceID(logCtx.Context()),
		StartTime:           logCtx.StartTime,
		EndTime:             endTime,
		CompletionStartTime: completionStartTime,
		UpstreamSendMs:      upstreamSendMs,
		RetryMs:             retryMs,
		ConnWaitMs:          connWaitMs,
		ConnReused:          connReused,
		TTFBMs:              ttfbMs,
		UpstreamAttempts:    attempts.Count(),
		Exit:                attempts.Exit(logCtx.Credential.Name),
		ExitRTTMs:           exitRTTMs,
		ExitConnectMs:       exitConnectMs,
		DurationMs:          endTime.Sub(logCtx.StartTime).Milliseconds(),
		TTFTMs:              ttftMs,

		CallType:     litellmCallType(logCtx.Request.URL.Path),
		APIBase:      apiBase,
		Status:       status,
		HTTPStatus:   logCtx.HTTPStatus,
		ErrorMessage: errorMessageForLog(logCtx),

		Model:      logCtx.ModelID,
		RealModel:  realModel,
		ModelID:    modelIDFormatted,
		ModelGroup: logCtx.spendModelGroup(),

		CredentialName:                 logCtx.Credential.Name,
		CredentialType:                 string(logCtx.Credential.Type),
		CredentialBaseURL:              logCtx.Credential.BaseURL,
		CredentialIsProxyRequest:       logCtx.IsProxyRequest,
		CredentialActualCredentialName: logCtx.ActualCredentialName,

		ServerRouterID: p.routerID,
		ServerVersion:  p.version,
		ServerCommit:   p.commit,
		ServiceName:    p.serviceName,

		PromptTokens:             usage.PromptTokens,
		CompletionTokens:         usage.CompletionTokens,
		TotalTokens:              usage.Total(),
		AudioInputTokens:         usage.AudioInputTokens,
		AudioOutputTokens:        usage.AudioOutputTokens,
		CachedInputTokens:        usage.CachedInputTokens,
		CachedAudioInputTokens:   usage.CachedAudioInputTokens,
		CacheCreationTokens:      usage.CacheCreationTokens,
		CacheCreation5mTokens:    usage.CacheCreation5mTokens,
		CacheCreation1hTokens:    usage.CacheCreation1hTokens,
		CacheCreation12hTokens:   usage.CacheCreation12hTokens,
		CacheType:                usage.CacheType,
		CachedOutputTokens:       usage.CachedOutputTokens,
		ReasoningTokens:          usage.ReasoningTokens,
		AcceptedPredictionTokens: usage.AcceptedPredictionTokens,
		RejectedPredictionTokens: usage.RejectedPredictionTokens,
		ImageCount:               usage.ImageCount,
		ImageTokens:              usage.ImageTokens,
		VideoInputTokens:         usage.VideoInputTokens,
		OutputImageTokens:        usage.OutputImageTokens,
		WebSearchRequests:        usage.WebSearchRequests,
		WebSearchContextSize:     usage.WebSearchContextSize,
		WebFetchRequests:         usage.WebFetchRequests,
		XSearchCalls:             usage.XSearchCalls,
		XSearchPosts:             usage.XSearchPosts,
		XSearchProfiles:          usage.XSearchProfiles,
		CodeExecutionCalls:       usage.CodeExecutionCalls,
		AttachmentSearchCalls:    usage.AttachmentSearchCalls,
		CollectionsSearchCalls:   usage.CollectionsSearchCalls,
		MCPCalls:                 usage.MCPCalls,
		ImageToolGenerations:     usage.ImageToolGenerations,
		ImageToolEdits:           usage.ImageToolEdits,

		TotalCost: cost,

		APIKeyHash:        hashedToken,
		UserID:            userID,
		TeamID:            teamID,
		OrganizationID:    organizationID,
		EndUser:           endUser,
		KeyAlias:          keyAlias,
		UserAlias:         userAlias,
		TeamAlias:         teamAlias,
		OrganizationAlias: organizationAlias,

		RequesterIP: getClientIP(logCtx.Request),
		SessionID:   logCtx.SessionID,
		OverheadMs:  overheadMs,
	}

	if tokenCosts != nil {
		event.InputCost = tokenCosts.InputCost
		event.OutputCost = tokenCosts.OutputCost
		event.AudioInputCost = tokenCosts.AudioInputCost
		event.AudioOutputCost = tokenCosts.AudioOutputCost
		event.ReasoningCost = tokenCosts.ReasoningCost
		event.CachedInputCost = tokenCosts.CachedInputCost
		event.ExplicitCacheReadCost = tokenCosts.ExplicitCachedInputCost
		event.CacheCreationCost = tokenCosts.CacheCreationCost
		event.CachedOutputCost = tokenCosts.CachedOutputCost
		event.PredictionCost = tokenCosts.PredictionCost
		event.ImageCost = tokenCosts.ImageCost
		event.VideoInputCost = tokenCosts.VideoInputCost
		event.WebSearchCost = tokenCosts.WebSearchCost
		event.WebFetchCost = tokenCosts.WebFetchCost
		event.XSearchCost = tokenCosts.XSearchCost
		event.CodeExecutionCost = tokenCosts.CodeExecutionCost
		event.AttachmentSearchCost = tokenCosts.AttachmentSearchCost
		event.CollectionsSearchCost = tokenCosts.CollectionsSearchCost
		event.ImageToolCost = tokenCosts.ImageGenerationToolCost
		event.ToolUsageCost = tokenCosts.ToolUsageCost
	}
	if usage.ProviderCostUSD > 0 {
		providerCost := usage.ProviderCostUSD
		event.ProviderReportedCost = &providerCost
	}

	if status == "failure" {
		event.ErrorClass = mapHTTPStatusToErrorClass(logCtx.HTTPStatus)
	}

	return event
}

// shouldLogRawBody reports whether this request's raw body is published to
// the raw-bodies topic, independent of whether that write-path is enabled at
// all (the caller checks rawBodyLogEnabled separately):
//
//   - a failed request always qualifies, matching the feature's original,
//     failures-only scope;
//   - with StoreOnlyErrors off, every request qualifies;
//   - a successful request that wrote to the provider's prompt cache
//     (TokenUsage.CacheCreationTokens > 0) qualifies too when
//     StoreCacheWrites is on, even while StoreOnlyErrors is still on for
//     everything else: reconciling a client's own cache-write accounting
//     against ours (see #355/#363) needs the actual bodies for requests that
//     succeeded, not just the ones that failed -- those already qualify via
//     the first rule, so this only ever adds the successful-cache-write case.
//
// A proxy/chain-audit request never qualifies: its body is the client's raw
// passthrough traffic, not a provider interaction this write-path exists to
// capture.
func (p *Proxy) shouldLogRawBody(logCtx *RequestLogContext, status string) bool {
	if logCtx.IsProxyRequest {
		return false
	}
	if status == "failure" || !p.rawBodyStoreOnlyErrors {
		return true
	}
	return p.rawBodyStoreCacheWrites && logCtx.TokenUsage.CacheCreationTokens > 0
}

// logRawBodyToKafka publishes the raw request/response body for a request
// that qualifies (see shouldLogRawBody) to the separate raw-bodies topic
// (internal/kafkalog), if that write-path is enabled. Best-effort, mirroring
// logSpendToKafka: never affects request processing, failures are logged and
// swallowed rather than surfaced to the caller -- unlike the spend event,
// there's no Postgres row to flag a fallback reason on for this one, it's
// purely supplementary.
func (p *Proxy) logRawBodyToKafka(logCtx *RequestLogContext, status string, endTime time.Time) {
	event := p.buildRawBodyEvent(logCtx, status, endTime)
	if err := p.rawBodyLog.LogRawBody(event); err != nil {
		p.logger.WarnContext(logCtx.Context(), "Failed to queue Kafka raw-body event",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
}

// logRouterResponseToKafka publishes what the router answered to the
// client, on its own topic.
func (p *Proxy) logRouterResponseToKafka(logCtx *RequestLogContext, status string, endTime time.Time) {
	event := p.buildRouterResponseEvent(logCtx, status, endTime)
	if err := p.routerResponseLog.LogRouterResponse(event); err != nil {
		p.logger.WarnContext(logCtx.Context(), "Failed to queue Kafka router-response event",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
}

// buildRouterResponseEvent maps a RequestLogContext onto
// kafkalog.RouterResponseEvent.
//
// The body was already captured -- and masked -- when the response was
// produced (see captureRouterResponse), so this only carries it over: it
// is not re-read from the response, which is gone by now.
func (p *Proxy) buildRouterResponseEvent(logCtx *RequestLogContext, status string, endTime time.Time) *kafkalog.RouterResponseEvent {
	event := &kafkalog.RouterResponseEvent{
		RequestID:      logCtx.spendRequestID(),
		ServerRouterID: p.routerID,
		StartTime:      logCtx.StartTime,
		EndTime:        endTime,
		HTTPStatus:     logCtx.HTTPStatus,
		Status:         status,
		ResponseBody:   logCtx.RouterResponseBody,
		IsStreaming:    logCtx.RouterResponseIsStreaming,
	}
	if logCtx.Credential != nil {
		event.Credential = logCtx.Credential.Name
	}
	if logCtx.ModelID != "" {
		event.Model = logCtx.ModelID
	}
	if logCtx.RealModelID != "" {
		event.RealModel = logCtx.RealModelID
	}
	if status == "failure" {
		event.ErrorClass = mapHTTPStatusToErrorClass(logCtx.HTTPStatus)
		event.ErrorOrigin = string(logCtx.ErrorOrigin)
	}
	return event
}

// buildRawBodyEvent maps a RequestLogContext onto kafkalog.RawBodyEvent.
// Caller (logSpendToLiteLLMDB) calls this for every failure, and additionally
// for successes when StoreOnlyErrors is disabled -- so, unlike
// buildKafkaSpendEvent, this function must check status itself rather than
// trust the caller's gate. Takes the same canonical status string
// buildKafkaSpendEvent uses (not a re-derived one), and gates ErrorClass on
// it rather than on a raw HTTPStatus >= 400 check: a mid-stream SSE error
// (provider returns HTTP 2xx, then sends an error event inside the stream --
// see stream.go's finalizeStreamingLog) sets Status = "failure" without
// touching HTTPStatus, which stays 2xx. Gating on HTTPStatus alone left that
// row's ErrorClass empty despite ResponseBody/ClientResponseBody being
// populated and the row being published -- same bug this function's
// original "don't trust a 2xx HTTPStatus" comment was trying to avoid, just
// missed the case where a 2xx HTTPStatus and a genuine failure coexist.
func (p *Proxy) buildRawBodyEvent(logCtx *RequestLogContext, status string, endTime time.Time) *kafkalog.RawBodyEvent {
	event := &kafkalog.RawBodyEvent{
		RequestID:          logCtx.spendRequestID(),
		ServerRouterID:     p.routerID,
		StartTime:          logCtx.StartTime,
		EndTime:            endTime,
		HTTPStatus:         logCtx.HTTPStatus,
		ResponseBody:       logCtx.ErrorBodyRaw,
		ClientResponseBody: logCtx.ClientResponseBody,
	}
	if p.rawBodyRedactSensitiveFields {
		event.ResponseBody = security.RedactBodyForLog(event.ResponseBody)
		event.ClientResponseBody = security.RedactBodyForLog(event.ClientResponseBody)
	}
	if status == "failure" {
		event.ErrorClass = mapHTTPStatusToErrorClass(logCtx.HTTPStatus)
		event.ErrorOrigin = string(logCtx.ErrorOrigin)
	}
	if p.rawBodyStoreRawBody {
		event.RequestBody = logCtx.RequestBodyRaw
	}
	return event
}
