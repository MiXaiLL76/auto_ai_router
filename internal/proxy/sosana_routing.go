package proxy

import (
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/sosana"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const unsupportedImageProviderRequestMessage = "request parameters are not supported by available image providers"
const unsupportedProviderEndpointMessage = "request endpoint is not supported by available providers"

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

	nextCred, nextReq, routed := p.nextPrimaryAfterUnsupportedSosana(r, prepared, modelID, *cred, reason)
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
	reason string,
) (*config.CredentialConfig, credentialPreparedRequest, bool) {
	triedCreds := GetTried(r.Context())
	triedCreds[currentCred.Name] = true

	for attempts := 0; attempts < 128; attempts++ {
		candidate, err := p.balancer.NextForModelExcluding(modelID, triedCreds)
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
