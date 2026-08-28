package proxy

import (
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
)

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

	var keyAlias, userAlias, teamAlias string
	if logCtx.TokenInfo != nil {
		keyAlias = logCtx.TokenInfo.KeyAlias
		userAlias = logCtx.TokenInfo.UserAlias
		teamAlias = logCtx.TokenInfo.TeamAlias
	}

	event := &kafkalog.SpendEvent{
		RequestID:           logCtx.spendRequestID(),
		StartTime:           logCtx.StartTime,
		EndTime:             endTime,
		CompletionStartTime: completionStartTime,
		DurationMs:          endTime.Sub(logCtx.StartTime).Milliseconds(),
		TTFTMs:              ttftMs,

		CallType:     litellmCallType(logCtx.Request.URL.Path),
		APIBase:      apiBase,
		Status:       status,
		HTTPStatus:   logCtx.HTTPStatus,
		ErrorMessage: logCtx.ErrorMsg,

		Model:      logCtx.ModelID,
		RealModel:  realModel,
		ModelID:    modelIDFormatted,
		ModelGroup: logCtx.ModelID,

		CredentialName:                 logCtx.Credential.Name,
		CredentialType:                 string(logCtx.Credential.Type),
		CredentialBaseURL:              logCtx.Credential.BaseURL,
		CredentialIsProxyRequest:       logCtx.IsProxyRequest,
		CredentialActualCredentialName: logCtx.ActualCredentialName,

		ServerRouterID: p.routerID,
		ServerVersion:  p.version,
		ServerCommit:   p.commit,

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
		CachedOutputTokens:       usage.CachedOutputTokens,
		ReasoningTokens:          usage.ReasoningTokens,
		AcceptedPredictionTokens: usage.AcceptedPredictionTokens,
		RejectedPredictionTokens: usage.RejectedPredictionTokens,
		ImageCount:               usage.ImageCount,
		ImageTokens:              usage.ImageTokens,
		OutputImageTokens:        usage.OutputImageTokens,
		WebSearchRequests:        usage.WebSearchRequests,
		WebSearchContextSize:     usage.WebSearchContextSize,

		TotalCost: cost,

		APIKeyHash:     hashedToken,
		UserID:         userID,
		TeamID:         teamID,
		OrganizationID: organizationID,
		EndUser:        endUser,
		KeyAlias:       keyAlias,
		UserAlias:      userAlias,
		TeamAlias:      teamAlias,

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
		event.CacheCreationCost = tokenCosts.CacheCreationCost
		event.CachedOutputCost = tokenCosts.CachedOutputCost
		event.PredictionCost = tokenCosts.PredictionCost
		event.ImageCost = tokenCosts.ImageCost
		event.WebSearchCost = tokenCosts.WebSearchCost
	}

	if status == "failure" {
		event.ErrorClass = mapHTTPStatusToErrorClass(logCtx.HTTPStatus)
	}

	return event
}
