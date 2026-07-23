package proxy

import (
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

const (
	// HeaderAARUsageAudioTokens describes the cached-audio contract of the
	// response body. AIR sets it only when audio_tokens in usage already exclude
	// cached_audio_tokens, so downstream AIR proxies must not subtract cached
	// audio again.
	HeaderAARUsageAudioTokens = "X-Aar-Usage-Audio-Tokens"

	aarUsageAudioTokensExcludeCached = "exclude-cached"
)

func markAudioUsageExcludesCached(headers http.Header) {
	headers.Set(HeaderAARUsageAudioTokens, aarUsageAudioTokensExcludeCached)
}

func audioUsageExcludesCached(headers http.Header) bool {
	if headers == nil {
		return false
	}
	for _, value := range headers.Values(HeaderAARUsageAudioTokens) {
		for _, part := range strings.Split(value, ",") {
			if strings.EqualFold(strings.TrimSpace(part), aarUsageAudioTokensExcludeCached) {
				return true
			}
		}
	}
	return false
}

func tokenUsageExtractionOptionsForCredential(cred *config.CredentialConfig) converter.TokenUsageExtractionOptions {
	includesCachedAudio := true
	if cred != nil && cred.Type == config.ProviderTypeProxy {
		// Legacy override for old upstream AIR deployments that do not emit the
		// usage-contract response header yet. New AIR-to-AIR chains should rely on
		// HeaderAARUsageAudioTokens instead of credential-level configuration.
		includesCachedAudio = cred.EffectiveProxyUsageFormat() != config.ProxyUsageFormatNormalized
	}
	return converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: includesCachedAudio}
}

func tokenUsageExtractionOptionsForResponse(cred *config.CredentialConfig, headers http.Header) converter.TokenUsageExtractionOptions {
	opts := tokenUsageExtractionOptionsForCredential(cred)
	if cred != nil && cred.Type == config.ProviderTypeProxy && audioUsageExcludesCached(headers) {
		opts.AudioInputIncludesCachedAudio = false
	}
	return opts
}

func directStreamingAudioUsageExcludesCached(prepared *orchestratedRequest, cred *config.CredentialConfig, statusCode int) bool {
	if statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return false
	}
	if prepared == nil {
		return false
	}
	if prepared.nativeResponses || prepared.convertedResp {
		return true
	}
	if prepared.passthroughResponses {
		return false
	}
	return transformedProviderAudioUsageExcludesCached(cred)
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
