package proxy

import (
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

const (
	// HeaderAIRProxyClient marks AIR-to-AIR internal proxy requests. AIR strips
	// it from inbound requests before forwarding to real providers.
	HeaderAIRProxyClient = "Air-Proxy-Client"

	// HeaderLegacyAIRProxyClient is kept for one rolling-upgrade window.
	HeaderLegacyAIRProxyClient = "X-Aar-Proxy-Client"

	// HeaderAIRUsageAudioTokens describes the cached-audio contract of the
	// response body. AIR sets it so downstream AIR proxies do not need a
	// credential-level guess about whether audio_tokens includes cached audio.
	HeaderAIRUsageAudioTokens = "Air-Usage-Audio-Tokens"

	airUsageAudioTokensExcludeCached = "exclude-cached"
	airUsageAudioTokensIncludeCached = "include-cached"
)

type audioUsageContract int

const (
	audioUsageContractUnknown audioUsageContract = iota
	audioUsageContractIncludesCached
	audioUsageContractExcludesCached
)

func markAudioUsageExcludesCached(headers http.Header) {
	headers.Set(HeaderAIRUsageAudioTokens, airUsageAudioTokensExcludeCached)
}

func markAudioUsageIncludesCached(headers http.Header) {
	headers.Set(HeaderAIRUsageAudioTokens, airUsageAudioTokensIncludeCached)
}

func audioUsageContractFromHeaders(headers http.Header) audioUsageContract {
	if headers == nil {
		return audioUsageContractUnknown
	}
	for _, value := range headers.Values(HeaderAIRUsageAudioTokens) {
		for _, part := range strings.Split(value, ",") {
			switch {
			case strings.EqualFold(strings.TrimSpace(part), airUsageAudioTokensExcludeCached):
				return audioUsageContractExcludesCached
			case strings.EqualFold(strings.TrimSpace(part), airUsageAudioTokensIncludeCached):
				return audioUsageContractIncludesCached
			}
		}
	}
	return audioUsageContractUnknown
}

func tokenUsageExtractionOptionsForCredential(cred *config.CredentialConfig) converter.TokenUsageExtractionOptions {
	includesCachedAudio := cred == nil || cred.Type != config.ProviderTypeAIR
	return converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: includesCachedAudio}
}

func tokenUsageExtractionOptionsForResponse(cred *config.CredentialConfig, headers http.Header) converter.TokenUsageExtractionOptions {
	opts := tokenUsageExtractionOptionsForCredential(cred)
	if cred != nil && cred.IsProxyLike() {
		switch audioUsageContractFromHeaders(headers) {
		case audioUsageContractExcludesCached:
			opts.AudioInputIncludesCachedAudio = false
		case audioUsageContractIncludesCached:
			opts.AudioInputIncludesCachedAudio = true
		}
	}
	return opts
}

func markAudioUsageContract(headers http.Header, includesCachedAudio bool) {
	if includesCachedAudio {
		markAudioUsageIncludesCached(headers)
	} else {
		markAudioUsageExcludesCached(headers)
	}
}

// markAudioUsageContractForClient sets the AIR audio-usage-contract header
// like markAudioUsageContract, but — mirroring setCredentialResponseHeader —
// suppresses it in allowlist mode for clients that aren't trusted internal
// AIR/proxy callers. The header only has a documented consumer (a downstream
// AIR router in the chain, see docs/providers/air.md) and must not leak
// router-internal signaling to plain external clients once allowlist mode
// asks for exactly that.
func (p *Proxy) markAudioUsageContractForClient(w http.ResponseWriter, logCtx *RequestLogContext, includesCachedAudio bool) {
	if logCtx == nil || !logCtx.IsProxyRequest {
		w.Header().Del(HeaderAIRUsageAudioTokens)
		return
	}
	markAudioUsageContract(w.Header(), includesCachedAudio)
}

func (p *Proxy) markAudioUsageExcludesCachedForClient(w http.ResponseWriter, logCtx *RequestLogContext) {
	p.markAudioUsageContractForClient(w, logCtx, false)
}

func (p *Proxy) markAudioUsageIncludesCachedForClient(w http.ResponseWriter, logCtx *RequestLogContext) {
	p.markAudioUsageContractForClient(w, logCtx, true)
}

func directStreamingAudioUsageContract(prepared *orchestratedRequest, cred *config.CredentialConfig, statusCode int) audioUsageContract {
	if statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return audioUsageContractUnknown
	}
	if prepared == nil {
		return audioUsageContractUnknown
	}
	if prepared.nativeResponses || prepared.convertedResp {
		return audioUsageContractExcludesCached
	}
	if prepared.passthroughResponses {
		return audioUsageContractIncludesCached
	}
	if transformedProviderAudioUsageExcludesCached(cred) {
		return audioUsageContractExcludesCached
	}
	return audioUsageContractIncludesCached
}

func transformedProviderAudioUsageExcludesCached(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	switch cred.Type {
	case config.ProviderTypeVertexAI,
		config.ProviderTypeGemini,
		config.ProviderTypeAnthropic,
		config.ProviderTypeCometAPI,
		config.ProviderTypeBedrock:
		return true
	default:
		return false
	}
}
