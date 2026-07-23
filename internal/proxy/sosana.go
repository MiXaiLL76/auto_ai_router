package proxy

import (
	"context"
	"errors"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/sosana"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const (
	sosanaPollInterval = time.Duration(sosana.PollInterval)
)

const unsupportedImageProviderRequestMessage = "request parameters are not supported by available image providers"
const unsupportedProviderEndpointMessage = "request endpoint is not supported by available providers"

type sosanaAttemptResult struct {
	body        []byte
	statusCode  int
	retryable   bool
	retryReason RetryReason
}

func (p *Proxy) applySosanaCompatibilityRouting(
	w http.ResponseWriter,
	r *http.Request,
	prepared *orchestratedRequest,
	modelID string,
	cred **config.CredentialConfig,
	body *[]byte,
	proxyBody *[]byte,
	realModelID *string,
	isImageGeneration bool,
	isImageEdit bool,
	logCtx *RequestLogContext,
	start time.Time,
) bool {
	if (*cred).Type != config.ProviderTypeSosana || (!isImageGeneration && !isImageEdit) {
		return true
	}

	reason := sosana.UnsupportedModel(*realModelID)
	if reason == "" {
		reason = sosana.UnsupportedRequest(r.URL.Path, *body, r.Header.Get("Content-Type"))
	}
	if reason == "" {
		return true
	}

	nextCred, nextReq, routed := p.nextPrimaryAfterUnsupportedSosana(r, prepared, modelID, *cred, logCtx.Scope, reason)
	if routed {
		*cred = nextCred
		*body = nextReq.body
		*proxyBody = nextReq.proxyBody
		*realModelID = nextReq.realModelID
		r.URL.Path = nextReq.path
		prepared.body = nextReq.body
		prepared.proxyBody = nextReq.proxyBody
		prepared.proxyPath = nextReq.proxyPath
		prepared.realModelID = nextReq.realModelID
		prepared.convertedResp = nextReq.convertedResp
		prepared.passthroughResponses = nextReq.passthroughResponses
		prepared.nativeResponses = nextReq.nativeResponses
		logCtx.RealModelID = *realModelID
		if span := trace.SpanFromContext(r.Context()); span.IsRecording() {
			span.SetAttributes(
				attribute.String("aar.real_model", *realModelID),
				attribute.String("aar.credential", nextCred.Name),
				attribute.String("aar.provider", string(nextCred.Type)),
				attribute.Bool("aar.provider_compatibility_skip", true),
			)
		}
		return true
	}

	success, fallbackReason := p.TryFallbackProxy(
		w,
		requestWithPath(r, prepared.proxyPath),
		modelID,
		(*cred).Name,
		http.StatusBadRequest,
		RetryReasonServerErr,
		*proxyBody,
		start,
		logCtx,
	)
	if success {
		return false
	}
	p.logger.DebugContext(r.Context(), "No fallback handled unsupported image provider request",
		"credential", (*cred).Name,
		"model", modelID,
		"reason", reason,
		"fallback_reason", fallbackReason)
	logCtx.Credential = *cred
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusBadRequest
	logCtx.ErrorMsg = unsupportedImageProviderRequestMessage
	WriteErrorBadRequest(w, unsupportedImageProviderRequestMessage)
	return false
}

func (p *Proxy) nextPrimaryAfterUnsupportedSosana(
	r *http.Request,
	prepared *orchestratedRequest,
	modelID string,
	currentCred *config.CredentialConfig,
	visibility scope.Context,
	reason string,
) (*config.CredentialConfig, credentialPreparedRequest, bool) {
	triedCreds := GetTried(r.Context())
	triedCreds[currentCred.Name] = true

	for attempts := 0; attempts < 128; attempts++ {
		candidate, err := p.balancer.NextForModelExcludingScoped(modelID, triedCreds, visibility)
		if err != nil {
			p.logger.DebugContext(r.Context(), "No compatible primary credential available for unsupported image request",
				"model", modelID,
				"credential", currentCred.Name,
				"reason", reason,
				"error", err)
			return nil, credentialPreparedRequest{}, false
		}
		triedCreds[candidate.Name] = true
		if candidate.Type == config.ProviderTypeSosana {
			continue
		}

		nextReq, prepErr := p.prepareRequestForCredential(
			r,
			prepared.baseBody,
			prepared.baseProxyBody,
			modelID,
			prepared.baseRealModelID,
			prepared.basePath,
			prepared.streaming,
			candidate,
			prepared.isResponsesAPI,
			prepared.responsesPrevHandled,
			prepared.stickyCacheEligible,
		)
		if prepErr != nil {
			p.logger.WarnContext(r.Context(), "Failed to prepare alternate primary request after image compatibility skip",
				"credential", candidate.Name,
				"provider", string(candidate.Type),
				"model", modelID,
				"reason", reason,
				"error", prepErr)
			continue
		}

		p.logger.InfoContext(r.Context(), "Skipping incompatible image credential for unsupported image request",
			"credential", currentCred.Name,
			"next_credential", candidate.Name,
			"model", modelID,
			"reason", reason)
		return candidate, nextReq, true
	}

	p.logger.WarnContext(r.Context(), "Image compatibility skip exhausted primary credential scan",
		"credential", currentCred.Name,
		"model", modelID,
		"reason", reason)
	return nil, credentialPreparedRequest{}, false
}

func (p *Proxy) handleSosanaRequest(
	w http.ResponseWriter,
	r *http.Request,
	body []byte,
	cred *config.CredentialConfig,
	modelID string,
	realModelID string,
	isImageGeneration bool,
	isImageEdit bool,
	logCtx *RequestLogContext,
	start time.Time,
) {
	logCtx.Credential = cred
	logCtx.TargetURL = cred.BaseURL

	if !isImageGeneration && !isImageEdit {
		message := unsupportedProviderEndpointMessage
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = message
		WriteErrorBadRequest(w, message)
		return
	}

	baseRealModelID := realModelID

	ctx := r.Context()
	var cancel context.CancelFunc
	if p.requestTimeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, p.requestTimeout)
		defer cancel()
	}

	result := sosanaAttemptResult{statusCode: http.StatusBadGateway, body: maskedUpstreamErrorBody(http.StatusBadGateway)}
	triedCreds := GetTried(r.Context())
	for attempt := 0; attempt <= p.maxProviderRetries; attempt++ {
		if attempt > 0 {
			nextCred, err := p.balancer.NextSameTypeForModelExcludingScoped(
				modelID,
				config.ProviderTypeSosana,
				triedCreds,
				logCtx.Scope,
			)
			if err != nil {
				p.logger.DebugContext(r.Context(), "No more Sosana credentials for retry",
					"model", modelID, "attempt", attempt, "error", err)
				break
			}
			cred = nextCred
			triedCreds[cred.Name] = true
			logCtx.Credential = cred
			logCtx.TargetURL = cred.BaseURL

			p.logger.InfoContext(r.Context(), "Retrying Sosana create request with next credential",
				"credential", cred.Name, "model", modelID,
				"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1,
				"retry_reason", result.retryReason)
			time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)
		}

		attemptRealModelID := p.sosanaRealModelIDForCredential(modelID, baseRealModelID, cred)
		createBody, concreteModelID, err := p.buildSosanaCreateBody(body, r.Header.Get("Content-Type"), attemptRealModelID, isImageEdit)
		if err != nil {
			p.logger.DebugContext(r.Context(), "Failed to prepare provider image request",
				"credential", cred.Name,
				"model", modelID,
				"real_model", attemptRealModelID,
				"error", err)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusBadRequest
			logCtx.ErrorMsg = unsupportedImageProviderRequestMessage
			WriteErrorBadRequest(w, unsupportedImageProviderRequestMessage)
			return
		}
		logCtx.RealModelID = concreteModelID

		result = p.createAndPollSosanaTask(ctx, cred, modelID, createBody, logCtx)
		p.balancer.RecordResponse(cred.Name, modelID, result.statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, result.statusCode, time.Since(start))
		if !result.retryable {
			break
		}

		p.logger.WarnContext(r.Context(), "Sosana create request returned retryable error, will retry",
			"error_code", result.statusCode,
			"credential", cred.Name,
			"reason", result.retryReason,
			"model", modelID,
			"attempt", attempt+1,
			"max_attempts", p.maxProviderRetries+1,
			"response_body_masked", true)
	}

	if result.statusCode >= 400 {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = result.statusCode
		logCtx.ErrorMsg = "Upstream provider error"
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(result.statusCode)
		_, _ = w.Write(result.body)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.body)

	logCtx.Status = "success"
	logCtx.HTTPStatus = http.StatusOK
	logCtx.TokenUsage = &converter.TokenUsage{ImageCount: logCtx.ImageCount}
	logCtx.RequestCompleted = true
	logCtx.Logged = true
	if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
		p.logger.WarnContext(r.Context(), "Failed to queue spend log",
			"error", err,
			"request_id", logCtx.RequestID)
	}
}

func (p *Proxy) sosanaRealModelIDForCredential(modelID, fallbackRealModelID string, cred *config.CredentialConfig) string {
	if p.modelManager != nil && cred != nil {
		if realModelID, ok := p.modelManager.GetRealModelNameForCredential(modelID, cred.Name); ok {
			return realModelID
		}
	}
	if strings.TrimSpace(fallbackRealModelID) != "" {
		return fallbackRealModelID
	}
	return modelID
}

func (p *Proxy) buildSosanaCreateBody(body []byte, contentType, realModelID string, isImageEdit bool) ([]byte, string, error) {
	if isImageEdit {
		return sosana.ImageEditRequest(body, contentType, realModelID)
	}
	return sosana.ImageGenerationRequest(body, realModelID)
}

func (p *Proxy) createAndPollSosanaTask(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	createBody []byte,
	logCtx *RequestLogContext,
) sosanaAttemptResult {
	task, rawBody, statusCode, err := p.doSosanaTaskRequest(ctx, http.MethodPost, sosana.CreateURL(cred.BaseURL), cred, createBody)
	if err != nil {
		body, code := p.sosanaTransportError(ctx, err, cred, modelID, sosana.CreateURL(cred.BaseURL), logCtx)
		return sosanaAttemptResult{
			body:       body,
			statusCode: code,
			retryable:  false,
		}
	}
	if statusCode >= 400 {
		p.logUpstreamError(ctx, "Sosana create request completed with error status", statusCode, cred, modelID, rawBody,
			"url", sosana.CreateURL(cred.BaseURL),
			"request_id", logCtx.RequestID)
		retryable, reason := ShouldRetryWithFallback(statusCode, rawBody)
		return sosanaAttemptResult{
			body:        maskedUpstreamErrorBody(statusCode),
			statusCode:  statusCode,
			retryable:   retryable,
			retryReason: reason,
		}
	}

	immediatePoll := true
	for {
		body, code, done := p.sosanaTaskBody(ctx, cred, modelID, task, rawBody, statusCode, logCtx)
		if done {
			return sosanaAttemptResult{body: body, statusCode: code}
		}

		if !immediatePoll {
			select {
			case <-ctx.Done():
				p.logUpstreamError(context.Background(), "Sosana task polling timed out", http.StatusRequestTimeout, cred, modelID, rawBody,
					"url", sosana.PollURL(cred.BaseURL, task.UID),
					"request_id", logCtx.RequestID,
					"error", ctx.Err())
				return sosanaAttemptResult{body: maskedUpstreamErrorBody(http.StatusRequestTimeout), statusCode: http.StatusRequestTimeout}
			case <-time.After(sosanaPollInterval):
			}
		}
		immediatePoll = false

		task, rawBody, statusCode, err = p.doSosanaTaskRequest(ctx, http.MethodGet, sosana.PollURL(cred.BaseURL, task.UID), cred, nil)
		if err != nil {
			body, code := p.sosanaTransportError(ctx, err, cred, modelID, sosana.PollURL(cred.BaseURL, task.UID), logCtx)
			return sosanaAttemptResult{body: body, statusCode: code}
		}
		if statusCode >= 400 {
			p.logUpstreamError(ctx, "Sosana poll request completed with error status", statusCode, cred, modelID, rawBody,
				"url", sosana.PollURL(cred.BaseURL, task.UID),
				"request_id", logCtx.RequestID)
			return sosanaAttemptResult{body: maskedUpstreamErrorBody(statusCode), statusCode: statusCode}
		}
	}
}

func (p *Proxy) sosanaTaskBody(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	task sosana.BananaTaskResponse,
	rawBody []byte,
	statusCode int,
	logCtx *RequestLogContext,
) ([]byte, int, bool) {
	switch task.Status {
	case sosana.StatusCompleted:
		body, statusCode := p.sosanaCompletedImageBody(ctx, cred, modelID, task, rawBody, logCtx)
		return body, statusCode, true
	case sosana.StatusFailed:
		p.logUpstreamError(ctx, "Sosana task failed", http.StatusBadGateway, cred, modelID, rawBody,
			"url", sosana.PollURL(cred.BaseURL, task.UID),
			"request_id", logCtx.RequestID)
		return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
	case sosana.StatusModerated:
		p.logUpstreamError(ctx, "Sosana task moderated", http.StatusBadRequest, cred, modelID, rawBody,
			"url", sosana.PollURL(cred.BaseURL, task.UID),
			"request_id", logCtx.RequestID)
		return maskedContentPolicyBody(), http.StatusBadRequest, true
	case sosana.StatusProcessing:
		if task.UID == "" {
			p.logUpstreamError(ctx, "Sosana processing task missing uid", http.StatusBadGateway, cred, modelID, rawBody,
				"url", cred.BaseURL,
				"request_id", logCtx.RequestID)
			return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
		}
		return nil, statusCode, false
	default:
		p.logUpstreamError(ctx, "Sosana task returned unknown status", http.StatusBadGateway, cred, modelID, rawBody,
			"url", sosana.PollURL(cred.BaseURL, task.UID),
			"request_id", logCtx.RequestID,
			"status", task.Status)
		return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
	}
}

func (p *Proxy) sosanaCompletedImageBody(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	task sosana.BananaTaskResponse,
	rawBody []byte,
	logCtx *RequestLogContext,
) ([]byte, int) {
	image, contentType, statusCode, err := p.downloadSosanaResultImage(ctx, cred, modelID, task, rawBody, logCtx)
	if err != nil {
		return maskedUpstreamErrorBody(statusCode), statusCode
	}
	body, err := sosana.OpenAIImageResponse(task, image)
	if err != nil {
		p.logUpstreamError(ctx, "Sosana completed task could not be converted", http.StatusBadGateway, cred, modelID, nil,
			"request_id", logCtx.RequestID,
			"error", err)
		return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway
	}
	p.logger.DebugContext(ctx, "Downloaded Sosana result image",
		"credential", cred.Name,
		"model", modelID,
		"result_host", sosana.ResultHost(task),
		"image_bytes", len(image),
		"content_type", contentType,
		"request_id", logCtx.RequestID)
	return body, http.StatusOK
}

func (p *Proxy) downloadSosanaResultImage(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	task sosana.BananaTaskResponse,
	rawTaskBody []byte,
	logCtx *RequestLogContext,
) ([]byte, string, int, error) {
	image, err := sosana.DownloadResultImage(ctx, p.client, task)
	if err == nil {
		return image.Bytes, image.ContentType, http.StatusOK, nil
	}

	statusCode := http.StatusBadGateway
	resultHost := sosana.ResultHost(task)
	var imageErr *sosana.ResultImageError
	if errors.As(err, &imageErr) {
		statusCode = imageErr.StatusCode
		resultHost = imageErr.Host
	}

	switch {
	case imageErr == nil:
		p.logUpstreamError(ctx, "Sosana result image download failed", statusCode, cred, modelID, nil,
			"result_host", resultHost,
			"request_id", logCtx.RequestID,
			"error", err)
	case imageErr.ResponseBody != nil:
		message := "Sosana result image download returned error status"
		if imageErr.SniffedContentType != "" {
			message = "Sosana result URL returned non-PNG content"
		}
		attrs := []any{
			"result_host", resultHost,
			"content_type", imageErr.ContentType,
			"request_id", logCtx.RequestID,
			"error", imageErr.Err,
		}
		if imageErr.UpstreamStatus != 0 {
			attrs = append(attrs, "upstream_status", imageErr.UpstreamStatus)
		}
		if imageErr.SniffedContentType != "" {
			attrs = append(attrs, "sniffed_content_type", imageErr.SniffedContentType)
		}
		p.logUpstreamError(ctx, message, statusCode, cred, modelID, imageErr.ResponseBody, attrs...)
	default:
		message := "Sosana result image download failed"
		if strings.Contains(err.Error(), "invalid") || strings.Contains(err.Error(), "without result_file_url") {
			message = "Sosana completed task returned invalid result URL"
		}
		if strings.Contains(err.Error(), "host is not allowed") ||
			strings.Contains(err.Error(), "private address") ||
			strings.Contains(err.Error(), "must use https") {
			message = "Sosana completed task returned unsafe result URL"
		}
		p.logUpstreamError(ctx, message, statusCode, cred, modelID, nil,
			"result_host", resultHost,
			"request_id", logCtx.RequestID,
			"error", err)
	}
	return nil, "", statusCode, err
}

func (p *Proxy) doSosanaTaskRequest(ctx context.Context, method, url string, cred *config.CredentialConfig, body []byte) (sosana.BananaTaskResponse, []byte, int, error) {
	result, err := sosana.DoTaskRequest(ctx, p.client, method, url, cred.APIKey, body)
	if err != nil {
		return result.Task, result.RawBody, result.StatusCode, err
	}
	return result.Task, result.RawBody, result.StatusCode, nil
}

func (p *Proxy) sosanaTransportError(ctx context.Context, err error, cred *config.CredentialConfig, modelID, url string, logCtx *RequestLogContext) ([]byte, int) {
	statusCode := http.StatusBadGateway
	if isTimeoutError(err) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
		statusCode = http.StatusRequestTimeout
	}
	p.logUpstreamError(context.Background(), "Sosana upstream request failed", statusCode, cred, modelID, nil,
		"url", url,
		"request_id", logCtx.RequestID,
		"error", err)
	return maskedUpstreamErrorBody(statusCode), statusCode
}
