package proxy

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const (
	sosanaPollInterval                 = 2 * time.Second
	maxSosanaMultipartImageBytes       = 20 * 1024 * 1024
	maxSosanaInputImages               = 14
	maxSosanaResultImageBytes    int64 = 32 * 1024 * 1024
	maxSosanaResultErrorBytes    int64 = 16 * 1024
)

const (
	sosanaStatusProcessing = "PROCESSING"
	sosanaStatusCompleted  = "COMPLETED"
	sosanaStatusFailed     = "FAILED"
	sosanaStatusModerated  = "MODERATED"
)

const unsupportedImageProviderRequestMessage = "request parameters are not supported by available image providers"
const unsupportedProviderEndpointMessage = "request endpoint is not supported by available providers"

var allowPrivateSosanaResultURLForTests func(*url.URL) bool

type sosanaAttemptResult struct {
	body        []byte
	statusCode  int
	retryable   bool
	retryReason RetryReason
}

type sosanaBananaCreateRequest struct {
	Prompt             string   `json:"prompt"`
	ImageURLs          []string `json:"image_urls,omitempty"`
	Model              string   `json:"model,omitempty"`
	AspectRatio        string   `json:"aspect_ratio,omitempty"`
	ImageSize          string   `json:"image_size,omitempty"`
	PromptOptimization bool     `json:"prompt_optimization"`
}

type sosanaBananaTaskResponse struct {
	UID             string  `json:"uid"`
	Status          string  `json:"status"`
	Prompt          string  `json:"prompt"`
	CreatedAt       string  `json:"created_at"`
	OptimizedPrompt string  `json:"optimized_prompt"`
	ResultFileURL   *string `json:"result_file_url"`
	Error           *string `json:"error"`
}

type sosanaOpenAIImageRequest struct {
	openai.OpenAIImageRequest
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Ratio       string `json:"ratio,omitempty"`
	ImageSize   string `json:"image_size,omitempty"`
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

	reason := unsupportedSosanaModel(*realModelID)
	if reason == "" {
		reason = unsupportedSosanaRequest(r.URL.Path, *body, r.Header.Get("Content-Type"))
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
		return buildSosanaImageEditRequest(body, contentType, realModelID)
	}
	return buildSosanaImageGenerationRequest(body, realModelID)
}

func (p *Proxy) createAndPollSosanaTask(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	createBody []byte,
	logCtx *RequestLogContext,
) sosanaAttemptResult {
	task, rawBody, statusCode, err := p.doSosanaTaskRequest(ctx, http.MethodPost, sosanaCreateURL(cred.BaseURL), cred, createBody)
	if err != nil {
		body, code := p.sosanaTransportError(ctx, err, cred, modelID, sosanaCreateURL(cred.BaseURL), logCtx)
		return sosanaAttemptResult{
			body:       body,
			statusCode: code,
			retryable:  false,
		}
	}
	if statusCode >= 400 {
		p.logUpstreamError(ctx, "Sosana create request completed with error status", statusCode, cred, modelID, rawBody,
			"url", sosanaCreateURL(cred.BaseURL),
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
					"url", sosanaPollURL(cred.BaseURL, task.UID),
					"request_id", logCtx.RequestID,
					"error", ctx.Err())
				return sosanaAttemptResult{body: maskedUpstreamErrorBody(http.StatusRequestTimeout), statusCode: http.StatusRequestTimeout}
			case <-time.After(sosanaPollInterval):
			}
		}
		immediatePoll = false

		task, rawBody, statusCode, err = p.doSosanaTaskRequest(ctx, http.MethodGet, sosanaPollURL(cred.BaseURL, task.UID), cred, nil)
		if err != nil {
			body, code := p.sosanaTransportError(ctx, err, cred, modelID, sosanaPollURL(cred.BaseURL, task.UID), logCtx)
			return sosanaAttemptResult{body: body, statusCode: code}
		}
		if statusCode >= 400 {
			p.logUpstreamError(ctx, "Sosana poll request completed with error status", statusCode, cred, modelID, rawBody,
				"url", sosanaPollURL(cred.BaseURL, task.UID),
				"request_id", logCtx.RequestID)
			return sosanaAttemptResult{body: maskedUpstreamErrorBody(statusCode), statusCode: statusCode}
		}
	}
}

func (p *Proxy) sosanaTaskBody(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	task sosanaBananaTaskResponse,
	rawBody []byte,
	statusCode int,
	logCtx *RequestLogContext,
) ([]byte, int, bool) {
	switch task.Status {
	case sosanaStatusCompleted:
		body, statusCode := p.sosanaCompletedImageBody(ctx, cred, modelID, task, rawBody, logCtx)
		return body, statusCode, true
	case sosanaStatusFailed:
		p.logUpstreamError(ctx, "Sosana task failed", http.StatusBadGateway, cred, modelID, rawBody,
			"url", sosanaPollURL(cred.BaseURL, task.UID),
			"request_id", logCtx.RequestID)
		return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
	case sosanaStatusModerated:
		p.logUpstreamError(ctx, "Sosana task moderated", http.StatusBadRequest, cred, modelID, rawBody,
			"url", sosanaPollURL(cred.BaseURL, task.UID),
			"request_id", logCtx.RequestID)
		return maskedContentPolicyBody(), http.StatusBadRequest, true
	case sosanaStatusProcessing:
		if task.UID == "" {
			p.logUpstreamError(ctx, "Sosana processing task missing uid", http.StatusBadGateway, cred, modelID, rawBody,
				"url", cred.BaseURL,
				"request_id", logCtx.RequestID)
			return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
		}
		return nil, statusCode, false
	default:
		p.logUpstreamError(ctx, "Sosana task returned unknown status", http.StatusBadGateway, cred, modelID, rawBody,
			"url", sosanaPollURL(cred.BaseURL, task.UID),
			"request_id", logCtx.RequestID,
			"status", task.Status)
		return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway, true
	}
}

func (p *Proxy) sosanaCompletedImageBody(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	task sosanaBananaTaskResponse,
	rawBody []byte,
	logCtx *RequestLogContext,
) ([]byte, int) {
	image, contentType, statusCode, err := p.downloadSosanaResultImage(ctx, cred, modelID, task, rawBody, logCtx)
	if err != nil {
		return maskedUpstreamErrorBody(statusCode), statusCode
	}
	body, err := buildSosanaOpenAIImageResponse(task, image)
	if err != nil {
		p.logUpstreamError(ctx, "Sosana completed task could not be converted", http.StatusBadGateway, cred, modelID, nil,
			"request_id", logCtx.RequestID,
			"error", err)
		return maskedUpstreamErrorBody(http.StatusBadGateway), http.StatusBadGateway
	}
	p.logger.DebugContext(ctx, "Downloaded Sosana result image",
		"credential", cred.Name,
		"model", modelID,
		"result_host", sosanaResultHost(task),
		"image_bytes", len(image),
		"content_type", contentType,
		"request_id", logCtx.RequestID)
	return body, http.StatusOK
}

func (p *Proxy) downloadSosanaResultImage(
	ctx context.Context,
	cred *config.CredentialConfig,
	modelID string,
	task sosanaBananaTaskResponse,
	rawTaskBody []byte,
	logCtx *RequestLogContext,
) ([]byte, string, int, error) {
	resultURL := ""
	if task.ResultFileURL != nil {
		resultURL = strings.TrimSpace(*task.ResultFileURL)
	}
	parsed, err := parseSosanaResultURL(resultURL)
	if err != nil {
		p.logUpstreamError(ctx, "Sosana completed task returned invalid result URL", http.StatusBadGateway, cred, modelID, rawTaskBody,
			"request_id", logCtx.RequestID,
			"error", err)
		return nil, "", http.StatusBadGateway, err
	}
	if err := validateSosanaResultURL(ctx, parsed); err != nil {
		p.logUpstreamError(ctx, "Sosana completed task returned unsafe result URL", http.StatusBadGateway, cred, modelID, rawTaskBody,
			"result_host", parsed.Hostname(),
			"request_id", logCtx.RequestID,
			"error", err)
		return nil, "", http.StatusBadGateway, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, resultURL, nil)
	if err != nil {
		p.logUpstreamError(ctx, "Failed to build Sosana result image request", http.StatusBadGateway, cred, modelID, nil,
			"result_host", parsed.Hostname(),
			"request_id", logCtx.RequestID,
			"error", err)
		return nil, "", http.StatusBadGateway, err
	}
	req.Header.Set("Accept", "image/*")

	resp, err := p.doSosanaResultImageRequest(req)
	if err != nil {
		statusCode := http.StatusBadGateway
		if isTimeoutError(err) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			statusCode = http.StatusRequestTimeout
		}
		p.logUpstreamError(context.Background(), "Sosana result image download failed", statusCode, cred, modelID, nil,
			"result_host", parsed.Hostname(),
			"request_id", logCtx.RequestID,
			"error", err)
		return nil, "", statusCode, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.WarnContext(ctx, "Failed to close Sosana result image body", "error", closeErr)
		}
	}()

	contentType := strings.TrimSpace(resp.Header.Get("Content-Type"))
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		errorBody := readTextBodyForSosanaResultLog(resp.Body, contentType)
		p.logUpstreamError(ctx, "Sosana result image download returned error status", http.StatusBadGateway, cred, modelID, errorBody,
			"upstream_status", resp.StatusCode,
			"result_host", parsed.Hostname(),
			"request_id", logCtx.RequestID)
		return nil, "", http.StatusBadGateway, errors.New("sosana result image download returned error status")
	}

	image, err := readLimitedSosanaResultImage(resp.Body)
	if err != nil {
		p.logUpstreamError(ctx, "Failed to read Sosana result image", http.StatusBadGateway, cred, modelID, nil,
			"result_host", parsed.Hostname(),
			"request_id", logCtx.RequestID,
			"error", err)
		return nil, "", http.StatusBadGateway, err
	}
	sniffedType := http.DetectContentType(image)
	if !isPNGContentType(contentType) && !isPNGContentType(sniffedType) {
		responseBody := textBodyPrefixForSosanaResultLog(image, contentType, sniffedType)
		p.logUpstreamError(ctx, "Sosana result URL returned non-PNG content", http.StatusBadGateway, cred, modelID, responseBody,
			"result_host", parsed.Hostname(),
			"content_type", contentType,
			"sniffed_content_type", sniffedType,
			"request_id", logCtx.RequestID)
		return nil, "", http.StatusBadGateway, errors.New("sosana result URL returned non-PNG content")
	}
	if !isPNGContentType(contentType) {
		contentType = sniffedType
	}
	return image, contentType, http.StatusOK, nil
}

func (p *Proxy) doSosanaResultImageRequest(req *http.Request) (*http.Response, error) {
	client := *p.client
	client.Transport = sosanaResultImageTransport()
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return client.Do(req)
}

func sosanaResultImageTransport() http.RoundTripper {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DisableKeepAlives = true
	transport.DialContext = dialSosanaResultAddress
	return transport
}

func dialSosanaResultAddress(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}

	dialer := &net.Dialer{}
	if allowPrivateSosanaResultHostForTests(host) {
		return dialer.DialContext(ctx, network, address)
	}
	if ip := net.ParseIP(host); ip != nil {
		if isUnsafeSosanaResultIP(ip) {
			return nil, errors.New("sosana result_file_url resolves to a private address")
		}
		return dialer.DialContext(ctx, network, address)
	}

	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, err
	}
	if len(addrs) == 0 {
		return nil, errors.New("sosana result_file_url host has no addresses")
	}
	for _, addr := range addrs {
		if isUnsafeSosanaResultIP(addr.IP) {
			return nil, errors.New("sosana result_file_url resolves to a private address")
		}
	}

	var dialErr error
	for _, addr := range addrs {
		conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(addr.IP.String(), port))
		if err == nil {
			return conn, nil
		}
		dialErr = err
	}
	if dialErr != nil {
		return nil, dialErr
	}
	return nil, errors.New("sosana result_file_url host has no dialable addresses")
}

func allowPrivateSosanaResultHostForTests(host string) bool {
	if allowPrivateSosanaResultURLForTests == nil {
		return false
	}
	return allowPrivateSosanaResultURLForTests(&url.URL{Scheme: "http", Host: host})
}

func parseSosanaResultURL(raw string) (*url.URL, error) {
	if raw == "" {
		return nil, errors.New("sosana task completed without result_file_url")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return nil, err
	}
	if parsed.Hostname() == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("sosana result_file_url must be an http or https URL")
	}
	return parsed, nil
}

func validateSosanaResultURL(ctx context.Context, parsed *url.URL) error {
	if allowPrivateSosanaResultURLForTests != nil && allowPrivateSosanaResultURLForTests(parsed) {
		return nil
	}
	if parsed.Scheme != "https" {
		return errors.New("sosana result_file_url must use https")
	}

	host := parsed.Hostname()
	if strings.EqualFold(host, "localhost") {
		return errors.New("sosana result_file_url host is not allowed")
	}
	if !isAllowedSosanaResultHost(host) {
		return errors.New("sosana result_file_url host is not allowed")
	}
	if ip := net.ParseIP(host); ip != nil {
		if isUnsafeSosanaResultIP(ip) {
			return errors.New("sosana result_file_url resolves to a private address")
		}
		return nil
	}

	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return err
	}
	if len(addrs) == 0 {
		return errors.New("sosana result_file_url host has no addresses")
	}
	for _, addr := range addrs {
		if isUnsafeSosanaResultIP(addr.IP) {
			return errors.New("sosana result_file_url resolves to a private address")
		}
	}
	return nil
}

func isAllowedSosanaResultHost(host string) bool {
	host = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(host)), ".")
	for _, suffix := range []string{
		"sosana.blog",
		"sosana.art",
		"storage.yandexcloud.net",
	} {
		if host == suffix || strings.HasSuffix(host, "."+suffix) {
			return true
		}
	}
	return false
}

func isUnsafeSosanaResultIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if v4 := ip.To4(); v4 != nil && v4[0] == 100 && v4[1]&0xc0 == 64 {
		return true
	}
	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsMulticast() ||
		ip.IsUnspecified()
}

func sosanaResultHost(task sosanaBananaTaskResponse) string {
	if task.ResultFileURL == nil {
		return ""
	}
	parsed, err := url.Parse(strings.TrimSpace(*task.ResultFileURL))
	if err != nil {
		return ""
	}
	return parsed.Hostname()
}

func readLimitedSosanaResultImage(body io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, maxSosanaResultImageBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxSosanaResultImageBytes {
		return nil, ErrResponseBodyTooLarge
	}
	if len(data) == 0 {
		return nil, errors.New("sosana result image body is empty")
	}
	return data, nil
}

func readTextBodyForSosanaResultLog(body io.Reader, contentType string) []byte {
	if !isTextContentType(contentType) {
		return nil
	}
	data, err := io.ReadAll(io.LimitReader(body, maxSosanaResultErrorBytes))
	if err != nil {
		return nil
	}
	return data
}

func textBodyPrefixForSosanaResultLog(body []byte, contentTypes ...string) []byte {
	for _, contentType := range contentTypes {
		if isTextContentType(contentType) {
			if int64(len(body)) > maxSosanaResultErrorBytes {
				return body[:maxSosanaResultErrorBytes]
			}
			return body
		}
	}
	return nil
}

func isPNGContentType(contentType string) bool {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	return contentType == "image/png" || strings.HasPrefix(contentType, "image/png;")
}

func isTextContentType(contentType string) bool {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	return strings.HasPrefix(contentType, "text/") ||
		strings.Contains(contentType, "json") ||
		strings.Contains(contentType, "xml")
}

func (p *Proxy) doSosanaTaskRequest(ctx context.Context, method, url string, cred *config.CredentialConfig, body []byte) (sosanaBananaTaskResponse, []byte, int, error) {
	var reader *bytes.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	} else {
		reader = bytes.NewReader(nil)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return sosanaBananaTaskResponse{}, nil, http.StatusInternalServerError, err
	}
	req.Header.Set("Authorization", "Bearer "+cred.APIKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return sosanaBananaTaskResponse{}, nil, http.StatusBadGateway, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.WarnContext(ctx, "Failed to close Sosana response body", "error", closeErr)
		}
	}()

	rawBody, err := p.readLimitedResponseBody(resp.Body)
	if err != nil {
		return sosanaBananaTaskResponse{}, nil, http.StatusBadGateway, err
	}
	var task sosanaBananaTaskResponse
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

func buildSosanaImageGenerationRequest(openAIBody []byte, modelID string) ([]byte, string, error) {
	if reason := unsupportedSosanaRequest("/v1/images/generations", openAIBody, "application/json"); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}

	var req sosanaOpenAIImageRequest
	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, "", fmt.Errorf("failed to parse OpenAI image request: %w", err)
	}
	if err := validateSosanaImageCount(req.N); err != nil {
		return nil, "", err
	}
	if err := validateSosanaResponseFormat(req.ResponseFormat); err != nil {
		return nil, "", err
	}
	if err := validateSosanaOutputFormat(req.OutputFormat); err != nil {
		return nil, "", err
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return nil, "", fmt.Errorf("image generation request missing prompt")
	}
	imageSize, err := sosanaImageSize(req.ImageSize, req.Size)
	if err != nil {
		return nil, "", err
	}
	concreteModel := sosanaProviderModel(modelID, req.Model, imageSize)
	if reason := unsupportedSosanaModel(concreteModel); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}
	body, err := json.Marshal(sosanaBananaCreateRequest{
		Prompt:             prompt,
		Model:              concreteModel,
		AspectRatio:        sosanaAspectRatio(req.AspectRatio, req.Ratio, req.Size),
		ImageSize:          imageSize,
		PromptOptimization: false,
	})
	if err != nil {
		return nil, "", err
	}
	return body, concreteModel, nil
}

func buildSosanaImageEditRequest(openAIBody []byte, contentType string, modelID string) ([]byte, string, error) {
	if reason := unsupportedSosanaRequest("/v1/images/edits", openAIBody, contentType); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}

	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, "", fmt.Errorf("failed to parse image edit content type: %w", err)
	}
	if !strings.HasPrefix(mediaType, "multipart/form-data") {
		return nil, "", fmt.Errorf("image edits require multipart/form-data content type")
	}
	boundary := params["boundary"]
	if boundary == "" {
		return nil, "", fmt.Errorf("missing multipart boundary in content type")
	}

	fields := make(map[string]string)
	imageURLs := make([]string, 0, 1)
	reader := multipart.NewReader(bytes.NewReader(openAIBody), boundary)
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", fmt.Errorf("failed to read multipart image edit payload: %w", err)
		}

		formName := part.FormName()
		if formName == "" {
			continue
		}
		data, err := readLimitedSosanaMultipartPart(part, maxSosanaMultipartImageBytes)
		if err != nil {
			return nil, "", err
		}
		if part.FileName() == "" {
			fields[formName] = strings.TrimSpace(string(data))
			continue
		}
		if formName == "mask" {
			return nil, "", fmt.Errorf("image edits do not support mask")
		}
		if formName != "image" && formName != "images" && formName != "image[]" {
			continue
		}
		mimeType := detectSosanaImageMIMEType(part.Header.Get("Content-Type"), data)
		if mimeType != "image/png" {
			return nil, "", fmt.Errorf("image edits support PNG images only")
		}
		imageURLs = append(imageURLs, "data:"+mimeType+";base64,"+base64.StdEncoding.EncodeToString(data))
	}

	if len(imageURLs) > maxSosanaInputImages {
		return nil, "", fmt.Errorf("image edits support up to %d input images", maxSosanaInputImages)
	}
	if err := validateSosanaImageCountString(fields["n"]); err != nil {
		return nil, "", err
	}
	if err := validateSosanaResponseFormat(fields["response_format"]); err != nil {
		return nil, "", err
	}
	if err := validateSosanaOutputFormat(fields["output_format"]); err != nil {
		return nil, "", err
	}
	prompt := strings.TrimSpace(fields["prompt"])
	if prompt == "" {
		return nil, "", fmt.Errorf("image edit request missing prompt field")
	}
	if len(imageURLs) == 0 {
		return nil, "", fmt.Errorf("image edit request missing image")
	}
	imageSize, err := sosanaImageSize(fields["image_size"], fields["size"])
	if err != nil {
		return nil, "", err
	}
	concreteModel := sosanaProviderModel(modelID, fields["model"], imageSize)
	if reason := unsupportedSosanaModel(concreteModel); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}
	body, err := json.Marshal(sosanaBananaCreateRequest{
		Prompt:             prompt,
		ImageURLs:          imageURLs,
		Model:              concreteModel,
		AspectRatio:        sosanaAspectRatio(fields["aspect_ratio"], fields["ratio"], fields["size"]),
		ImageSize:          imageSize,
		PromptOptimization: false,
	})
	if err != nil {
		return nil, "", err
	}
	return body, concreteModel, nil
}

func buildSosanaOpenAIImageResponse(task sosanaBananaTaskResponse, image []byte) ([]byte, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("image task completed without image bytes")
	}
	resp := openai.OpenAIImageResponse{
		Created: sosanaCreatedAtUnix(task.CreatedAt),
		Data: []openai.OpenAIImageData{
			{
				B64JSON:       base64.StdEncoding.EncodeToString(image),
				RevisedPrompt: strings.TrimSpace(task.OptimizedPrompt),
			},
		},
	}
	return json.Marshal(resp)
}

var unsupportedSosanaImageFields = []string{
	"tools",
	"tool_choice",
	"google_search",
	"thinking_level",
	"thinking_budget",
	"thinking_config",
	"thinking",
	"reasoning_effort",
	"generation_config",
	"temperature",
	"top_p",
	"top_k",
	"seed",
	"max_tokens",
	"stop",
	"stream",
	"messages",
	"extra_body",
	"image",
	"images",
	"image_urls",
	"reference_images",
}

func unsupportedSosanaRequest(path string, body []byte, contentType string) string {
	switch {
	case strings.Contains(path, "/images/generations"):
		return unsupportedSosanaGenerationRequest(body)
	case strings.Contains(path, "/images/edits"):
		return unsupportedSosanaEditRequest(body, contentType)
	default:
		return "endpoint is unsupported"
	}
}

func unsupportedSosanaModel(modelID string) string {
	if supportedSosanaModel(modelID) {
		return ""
	}
	return "model is unsupported"
}

func unsupportedSosanaGenerationRequest(body []byte) string {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return ""
	}
	return unsupportedSosanaImageFieldsInJSON(raw)
}

func unsupportedSosanaEditRequest(body []byte, contentType string) string {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(mediaType, "multipart/form-data") {
		return ""
	}
	boundary := params["boundary"]
	if boundary == "" {
		return ""
	}

	fields := make(map[string]string)
	imageCount := 0
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, err := reader.NextPart()
		if err != nil {
			break
		}

		formName := part.FormName()
		if formName == "" {
			continue
		}
		data, err := readLimitedSosanaMultipartPart(part, maxSosanaMultipartImageBytes)
		if err != nil {
			return err.Error()
		}
		if part.FileName() == "" {
			fields[formName] = strings.TrimSpace(string(data))
			continue
		}
		if formName == "mask" {
			return "mask is unsupported"
		}
		if formName != "image" && formName != "images" && formName != "image[]" {
			continue
		}
		imageCount++
		if detectSosanaImageMIMEType(part.Header.Get("Content-Type"), data) != "image/png" {
			return "only PNG input images are supported"
		}
	}
	if imageCount > maxSosanaInputImages {
		return "too many input images"
	}
	return unsupportedSosanaImageFieldsInForm(fields)
}

func unsupportedSosanaImageFieldsInJSON(raw map[string]json.RawMessage) string {
	if reason := unsupportedSosanaJSONImageCount(raw["n"]); reason != "" {
		return reason
	}
	if reason := unsupportedSosanaJSONResponseFormat(raw["response_format"]); reason != "" {
		return reason
	}
	if reason := unsupportedSosanaJSONOutputFormat(raw["output_format"]); reason != "" {
		return reason
	}
	if reason := unsupportedSosanaJSONImageSize(raw["image_size"]); reason != "" {
		return reason
	}
	if reason := unsupportedSosanaJSONExactSize(raw["size"]); reason != "" {
		return reason
	}
	for _, field := range []string{"quality", "style", "background", "moderation"} {
		if hasSosanaJSONValue(raw[field]) {
			return field + " is unsupported"
		}
	}
	if hasSosanaJSONValue(raw["output_compression"]) {
		return "output_compression is unsupported"
	}
	for _, field := range unsupportedSosanaImageFields {
		if hasSosanaJSONValue(raw[field]) {
			return field + " is unsupported"
		}
	}
	return ""
}

func unsupportedSosanaImageFieldsInForm(fields map[string]string) string {
	if err := validateSosanaImageCountString(fields["n"]); err != nil {
		return err.Error()
	}
	if err := validateSosanaResponseFormat(fields["response_format"]); err != nil {
		return err.Error()
	}
	if err := validateSosanaOutputFormat(fields["output_format"]); err != nil {
		return err.Error()
	}
	if reason := unsupportedSosanaFormImageSize(fields["image_size"]); reason != "" {
		return reason
	}
	if reason := unsupportedSosanaFormExactSize(fields["size"]); reason != "" {
		return reason
	}
	for _, field := range []string{"quality", "style", "background", "moderation"} {
		if strings.TrimSpace(fields[field]) != "" {
			return field + " is unsupported"
		}
	}
	if _, ok := fields["output_compression"]; ok {
		return "output_compression is unsupported"
	}
	for _, field := range unsupportedSosanaImageFields {
		if strings.TrimSpace(fields[field]) != "" {
			return field + " is unsupported"
		}
	}
	return ""
}

func unsupportedSosanaJSONImageCount(raw json.RawMessage) string {
	if !hasSosanaJSONValue(raw) {
		return ""
	}
	var n int
	if err := json.Unmarshal(raw, &n); err == nil {
		if n == 1 {
			return ""
		}
		return "image requests support n=1 only"
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		if f == 1 {
			return ""
		}
		return "image requests support n=1 only"
	}
	return "invalid image count"
}

func unsupportedSosanaJSONResponseFormat(raw json.RawMessage) string {
	if !hasSosanaJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "response_format is unsupported"
	}
	if strings.EqualFold(strings.TrimSpace(value), "b64_json") || strings.TrimSpace(value) == "" {
		return ""
	}
	if strings.EqualFold(strings.TrimSpace(value), "url") {
		return "response_format=url is unsupported for this image model"
	}
	return "response_format is unsupported"
}

func unsupportedSosanaJSONOutputFormat(raw json.RawMessage) string {
	if !hasSosanaJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "output_format is unsupported"
	}
	if sosanaOutputFormatAllowed(value) {
		return ""
	}
	return "output_format is unsupported"
}

func unsupportedSosanaJSONImageSize(raw json.RawMessage) string {
	if !hasSosanaJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "image_size is unsupported"
	}
	if _, ok := normalizeSosanaImageSize(value); ok {
		return ""
	}
	return "image_size is unsupported"
}

func unsupportedSosanaJSONExactSize(raw json.RawMessage) string {
	if !hasSosanaJSONValue(raw) {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "size is unsupported"
	}
	if _, ok := sosanaImageSizeFromExactSize(value); ok {
		return ""
	}
	return "size is unsupported"
}

func unsupportedSosanaFormImageSize(raw string) string {
	if strings.TrimSpace(raw) == "" {
		return ""
	}
	if _, ok := normalizeSosanaImageSize(raw); ok {
		return ""
	}
	return "image_size is unsupported"
}

func unsupportedSosanaFormExactSize(raw string) string {
	if strings.TrimSpace(raw) == "" {
		return ""
	}
	if _, ok := sosanaImageSizeFromExactSize(raw); ok {
		return ""
	}
	return "size is unsupported"
}

func supportedSosanaModel(modelID string) bool {
	model := strings.ToLower(strings.TrimSpace(modelID))
	if model == "" || model == "google/gemini-3.1-flash-image-preview" {
		return true
	}
	if model == "banana-2-{image_size}-compliant" {
		return true
	}
	switch model {
	case "banana-2-1k-compliant", "banana-2-2k-compliant", "banana-2-4k-compliant":
		return true
	default:
		return false
	}
}

func validateSosanaOutputFormat(raw string) error {
	if sosanaOutputFormatAllowed(raw) {
		return nil
	}
	return fmt.Errorf("output_format is unsupported")
}

func sosanaOutputFormatAllowed(raw string) bool {
	value := strings.ToLower(strings.TrimSpace(raw))
	return value == "" || value == "png"
}

func hasSosanaJSONValue(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && !bytes.Equal(raw, []byte("null"))
}

func sosanaProviderModel(modelID, requestModel, imageSize string) string {
	model := strings.TrimSpace(modelID)
	if model == "" {
		model = strings.TrimSpace(requestModel)
	}
	if model == "" {
		return ""
	}
	if strings.EqualFold(model, "google/gemini-3.1-flash-image-preview") {
		return "banana-2-" + strings.ToLower(imageSize) + "-compliant"
	}
	return strings.ReplaceAll(model, "{image_size}", strings.ToLower(imageSize))
}

func sosanaCreatedAtUnix(value string) int64 {
	if value == "" {
		return converterutil.GetCurrentTimestamp()
	}
	if ts, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return ts.Unix()
	}
	return converterutil.GetCurrentTimestamp()
}

func sosanaCreateURL(baseURL string) string {
	return strings.TrimSuffix(baseURL, "/") + "/api/banana/create-async"
}

func sosanaPollURL(baseURL, uid string) string {
	return strings.TrimSuffix(baseURL, "/") + "/api/banana/" + uid
}

func sosanaSizeToAspectRatio(size string) string {
	if spec, ok := sosanaImageSpecFromExactSize(size); ok {
		return spec.aspectRatio
	}
	return "auto"
}

func sosanaAspectRatio(explicit, ratio, size string) string {
	if value := strings.TrimSpace(explicit); value != "" {
		return value
	}
	if value := strings.TrimSpace(ratio); value != "" {
		return value
	}
	return sosanaSizeToAspectRatio(size)
}

func sosanaImageSize(explicit, size string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		if value, ok := normalizeSosanaImageSize(explicit); ok {
			return value, nil
		}
		return "", fmt.Errorf("image_size is unsupported")
	}
	if value, ok := sosanaImageSizeFromExactSize(size); ok {
		return value, nil
	}
	return "", fmt.Errorf("size is unsupported")
}

func normalizeSosanaImageSize(raw string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "auto", "1k":
		return "1K", true
	case "2k":
		return "2K", true
	case "4k":
		return "4K", true
	default:
		return "", false
	}
}

func sosanaImageSizeFromExactSize(size string) (string, bool) {
	if spec, ok := sosanaImageSpecFromExactSize(size); ok {
		return spec.imageSize, true
	}
	return "", false
}

type sosanaExactImageSpec struct {
	imageSize   string
	aspectRatio string
}

func sosanaImageSpecFromExactSize(size string) (sosanaExactImageSpec, bool) {
	switch strings.TrimSpace(size) {
	case "", "auto":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "auto"}, true

	case "1024x1024":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "1:1"}, true
	case "512x2048":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "1:4"}, true
	case "384x3072":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "1:8"}, true
	case "848x1264":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "2:3"}, true
	case "1264x848":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "3:2"}, true
	case "896x1200":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "3:4"}, true
	case "2048x512":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "4:1"}, true
	case "1200x896":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "4:3"}, true
	case "928x1152":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "4:5"}, true
	case "1152x928":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "5:4"}, true
	case "3072x384":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "8:1"}, true
	case "768x1376":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "9:16"}, true
	case "1376x768":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "16:9"}, true
	case "1584x672":
		return sosanaExactImageSpec{imageSize: "1K", aspectRatio: "21:9"}, true

	case "2048x2048":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "1:1"}, true
	case "1024x4096":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "1:4"}, true
	case "768x6144":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "1:8"}, true
	case "1696x2528":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "2:3"}, true
	case "2528x1696":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "3:2"}, true
	case "1792x2400":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "3:4"}, true
	case "4096x1024":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "4:1"}, true
	case "2400x1792":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "4:3"}, true
	case "1856x2304":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "4:5"}, true
	case "2304x1856":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "5:4"}, true
	case "6144x768":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "8:1"}, true
	case "1536x2752":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "9:16"}, true
	case "2752x1536":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "16:9"}, true
	case "3168x1344":
		return sosanaExactImageSpec{imageSize: "2K", aspectRatio: "21:9"}, true

	case "4096x4096":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "1:1"}, true
	case "2048x8192":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "1:4"}, true
	case "1536x12288":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "1:8"}, true
	case "3392x5056":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "2:3"}, true
	case "5056x3392":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "3:2"}, true
	case "3584x4800":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "3:4"}, true
	case "8192x2048":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "4:1"}, true
	case "4800x3584":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "4:3"}, true
	case "3712x4608":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "4:5"}, true
	case "4608x3712":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "5:4"}, true
	case "12288x1536":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "8:1"}, true
	case "3072x5504":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "9:16"}, true
	case "5504x3072":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "16:9"}, true
	case "6336x2688":
		return sosanaExactImageSpec{imageSize: "4K", aspectRatio: "21:9"}, true
	default:
		return sosanaExactImageSpec{}, false
	}
}

func validateSosanaImageCount(n *int) error {
	if n == nil || *n == 1 {
		return nil
	}
	return fmt.Errorf("image requests support n=1 only")
}

func validateSosanaResponseFormat(format string) error {
	if strings.EqualFold(strings.TrimSpace(format), "url") {
		return fmt.Errorf("response_format=url is unsupported for this image model")
	}
	return nil
}

func validateSosanaImageCountString(raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return fmt.Errorf("invalid image count: %w", err)
	}
	if n != 1 {
		return fmt.Errorf("image requests support n=1 only")
	}
	return nil
}

func readLimitedSosanaMultipartPart(r io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, fmt.Errorf("failed to read multipart part: %w", err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("multipart image exceeds %d bytes", limit)
	}
	return data, nil
}

func detectSosanaImageMIMEType(header string, data []byte) string {
	header = strings.ToLower(strings.TrimSpace(header))
	if strings.HasPrefix(header, "image/") {
		mediaType, _, err := mime.ParseMediaType(header)
		if err == nil {
			return mediaType
		}
		return strings.TrimSpace(strings.Split(header, ";")[0])
	}
	detected := http.DetectContentType(data)
	if strings.HasPrefix(detected, "image/") {
		return detected
	}
	return "application/octet-stream"
}
