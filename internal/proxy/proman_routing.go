package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const unsupportedProManRequestMessage = "request parameters are not supported by available providers"

func (p *Proxy) applyProManCompatibilityRouting(
	w http.ResponseWriter,
	r *http.Request,
	prepared *orchestratedRequest,
	modelID string,
	cred **config.CredentialConfig,
	body *[]byte,
	proxyBody *[]byte,
	realModelID *string,
	logCtx *RequestLogContext,
	start time.Time,
) bool {
	if (*cred).Type == config.ProviderTypeProxy || !isProManCredential(*cred) {
		return true
	}

	reason := unsupportedProManRequest(*body, modelID)
	if reason == "" {
		return true
	}

	nextCred, nextReq, routed := p.nextPrimaryAfterUnsupportedProMan(r, prepared, modelID, *cred, logCtx.Scope, reason)
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
		logCtx.Credential = nextCred
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

	p.logger.DebugContext(r.Context(), "No fallback handled unsupported ProMan request",
		"credential", (*cred).Name,
		"model", modelID,
		"reason", reason,
		"fallback_reason", fallbackReason)
	logCtx.Credential = *cred
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusBadRequest
	logCtx.ErrorMsg = unsupportedProManRequestMessage
	WriteErrorBadRequest(w, unsupportedProManRequestMessage)
	return false
}

func (p *Proxy) nextPrimaryAfterUnsupportedProMan(
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
			p.logger.DebugContext(r.Context(), "No compatible primary credential available for unsupported ProMan request",
				"model", modelID,
				"credential", currentCred.Name,
				"reason", reason,
				"error", err)
			return nil, credentialPreparedRequest{}, false
		}
		triedCreds[candidate.Name] = true
		if isProManCredential(candidate) {
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
			p.logger.WarnContext(r.Context(), "Failed to prepare alternate primary request after ProMan compatibility skip",
				"credential", candidate.Name,
				"provider", string(candidate.Type),
				"model", modelID,
				"reason", reason,
				"error", prepErr)
			continue
		}

		p.logger.InfoContext(r.Context(), "Skipping incompatible ProMan credential for unsupported request",
			"credential", currentCred.Name,
			"next_credential", candidate.Name,
			"model", modelID,
			"reason", reason)
		return candidate, nextReq, true
	}

	p.logger.WarnContext(r.Context(), "ProMan compatibility skip exhausted primary credential scan",
		"credential", currentCred.Name,
		"model", modelID,
		"reason", reason)
	return nil, credentialPreparedRequest{}, false
}

type proManModelCapabilities struct {
	blockAssistantPrefill bool
	blockLegacyThinking   bool
}

var proManCapabilitiesByModel = map[string]proManModelCapabilities{
	"claude-fable-5": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   true,
	},
	"claude-haiku-4.5": {
		blockAssistantPrefill: false,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.5": {
		blockAssistantPrefill: false,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.6": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.7": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.8": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-sonnet-4.5": {
		blockAssistantPrefill: false,
		blockLegacyThinking:   false,
	},
	"claude-sonnet-4.6": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-sonnet-5": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   true,
	},
}

func unsupportedProManRequest(body []byte, selectedModel string) string {
	if len(bytes.TrimSpace(body)) == 0 {
		return ""
	}

	var root any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&root); err != nil {
		return ""
	}

	obj, ok := root.(map[string]any)
	if !ok {
		return ""
	}
	model := proManCanonicalModel(selectedModel)
	if model == "" {
		model = proManCanonicalModel(stringValue(obj["model"]))
	}
	caps, hasCaps := proManCapabilitiesByModel[model]
	if hasProManRecursiveType(root, "server_tool_use") {
		return "server_tool_use"
	}
	if hasCaps {
		if caps.blockLegacyThinking && hasProManUnsupportedThinking(obj) {
			return "thinking"
		}
		if caps.blockAssistantPrefill && hasProManAssistantPrefill(obj) {
			return "assistant_prefill"
		}
	}
	return ""
}

func hasProManUnsupportedThinking(obj map[string]any) bool {
	for _, container := range []map[string]any{obj, mapValue(obj["extra_body"])} {
		if container == nil {
			continue
		}
		if thinking, ok := container["thinking"]; ok {
			if hasEnabledThinkingConfig(thinking) {
				return true
			}
		}
		if nonNoneString(container["reasoning_effort"]) {
			return true
		}
	}

	if reasoning, ok := obj["reasoning"].(map[string]any); ok {
		return nonNoneString(reasoning["effort"])
	}
	return false
}

func hasEnabledThinkingConfig(value any) bool {
	thinking, ok := value.(map[string]any)
	if !ok {
		return true
	}
	typ := strings.ToLower(strings.TrimSpace(stringValue(thinking["type"])))
	if typ == "disabled" {
		return false
	}
	return true
}

func hasProManRecursiveType(value any, targetType string) bool {
	switch typed := value.(type) {
	case map[string]any:
		if typ, _ := typed["type"].(string); strings.EqualFold(strings.TrimSpace(typ), targetType) {
			return true
		}
		for _, nested := range typed {
			if hasProManRecursiveType(nested, targetType) {
				return true
			}
		}
	case []any:
		for _, nested := range typed {
			if hasProManRecursiveType(nested, targetType) {
				return true
			}
		}
	}
	return false
}

func hasProManAssistantPrefill(obj map[string]any) bool {
	messages, ok := obj["messages"].([]any)
	if !ok || len(messages) == 0 {
		if input, ok := obj["input"].([]any); ok {
			messages = input
		}
	}
	if len(messages) == 0 {
		return false
	}
	last, ok := messages[len(messages)-1].(map[string]any)
	if !ok {
		return false
	}
	role := strings.TrimSpace(strings.ToLower(stringValue(last["role"])))
	return role == "assistant" || role == "model"
}

func proManCanonicalModel(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	normalized = strings.ReplaceAll(normalized, "_", "-")
	normalized = strings.ReplaceAll(normalized, ".", "-")
	for _, prefix := range []string{"anthropic/", "anthropic-", "claude/"} {
		normalized = strings.TrimPrefix(normalized, prefix)
	}

	switch {
	case strings.Contains(normalized, "fable-5"):
		return "claude-fable-5"
	case strings.Contains(normalized, "haiku-4-5"):
		return "claude-haiku-4.5"
	case strings.Contains(normalized, "opus-4-5"):
		return "claude-opus-4.5"
	case strings.Contains(normalized, "opus-4-6"):
		return "claude-opus-4.6"
	case strings.Contains(normalized, "opus-4-7"):
		return "claude-opus-4.7"
	case strings.Contains(normalized, "opus-4-8"):
		return "claude-opus-4.8"
	case strings.Contains(normalized, "sonnet-4-5"):
		return "claude-sonnet-4.5"
	case strings.Contains(normalized, "sonnet-4-6"):
		return "claude-sonnet-4.6"
	case strings.Contains(normalized, "sonnet-5"):
		return "claude-sonnet-5"
	default:
		return ""
	}
}

func mapValue(value any) map[string]any {
	obj, _ := value.(map[string]any)
	return obj
}

func stringValue(value any) string {
	s, _ := value.(string)
	return s
}

func nonNoneString(value any) bool {
	s, ok := value.(string)
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "none", "disable", "disabled":
		return false
	default:
		return true
	}
}
