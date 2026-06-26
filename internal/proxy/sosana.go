package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math/rand"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/sosana"
)

const sosanaPollInterval = 2 * time.Second

type sosanaAttemptResult struct {
	body        []byte
	statusCode  int
	retryable   bool
	retryReason RetryReason
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
		message := "sosana provider supports only image generation"
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = message
		WriteErrorBadRequest(w, message)
		return
	}

	createBody, err := p.buildSosanaCreateBody(body, r.Header.Get("Content-Type"), realModelID, isImageEdit)
	if err != nil {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = err.Error()
		WriteErrorBadRequest(w, err.Error())
		return
	}

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
			nextCred, err := p.balancer.NextSameTypeForModelExcluding(modelID, config.ProviderTypeSosana, triedCreds)
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

func (p *Proxy) buildSosanaCreateBody(body []byte, contentType, realModelID string, isImageEdit bool) ([]byte, error) {
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
			body:        body,
			statusCode:  code,
			retryable:   ctx.Err() == nil,
			retryReason: RetryReasonNetErr,
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
		body, err := sosana.OpenAIImageResponse(task)
		if err != nil {
			p.logUpstreamError(ctx, "Sosana completed task missing result", http.StatusBadGateway, cred, modelID, rawBody,
				"url", sosana.PollURL(cred.BaseURL, task.UID),
				"request_id", logCtx.RequestID,
				"error", err)
			return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
		}
		return body, http.StatusOK, true
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

func (p *Proxy) doSosanaTaskRequest(ctx context.Context, method, url string, cred *config.CredentialConfig, body []byte) (sosana.BananaTaskResponse, []byte, int, error) {
	var reader *bytes.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	} else {
		reader = bytes.NewReader(nil)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return sosana.BananaTaskResponse{}, nil, http.StatusInternalServerError, err
	}
	req.Header.Set("Authorization", "Bearer "+cred.APIKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return sosana.BananaTaskResponse{}, nil, http.StatusBadGateway, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.WarnContext(ctx, "Failed to close Sosana response body", "error", closeErr)
		}
	}()

	rawBody, err := p.readLimitedResponseBody(resp.Body)
	if err != nil {
		return sosana.BananaTaskResponse{}, nil, http.StatusBadGateway, err
	}
	var task sosana.BananaTaskResponse
	if len(rawBody) > 0 {
		_ = json.Unmarshal(rawBody, &task)
	}
	return task, rawBody, resp.StatusCode, nil
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
