package proxy

import (
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

const (
	// HeaderAARUsageAudioTokens describes the cached-audio contract of the
	// response body. AIR sets it so downstream AIR proxies do not need a
	// credential-level guess about whether audio_tokens includes cached audio.
	HeaderAARUsageAudioTokens = "X-Aar-Usage-Audio-Tokens"

	aarUsageAudioTokensExcludeCached = "exclude-cached"
	aarUsageAudioTokensIncludeCached = "include-cached"
)

type audioUsageContract int

const (
	audioUsageContractUnknown audioUsageContract = iota
	audioUsageContractIncludesCached
	audioUsageContractExcludesCached
)

func markAudioUsageExcludesCached(headers http.Header) {
	headers.Set(HeaderAARUsageAudioTokens, aarUsageAudioTokensExcludeCached)
}

func markAudioUsageIncludesCached(headers http.Header) {
	headers.Set(HeaderAARUsageAudioTokens, aarUsageAudioTokensIncludeCached)
}

func audioUsageContractFromHeaders(headers http.Header) audioUsageContract {
	if headers == nil {
		return audioUsageContractUnknown
	}
	for _, value := range headers.Values(HeaderAARUsageAudioTokens) {
		for _, part := range strings.Split(value, ",") {
			switch {
			case strings.EqualFold(strings.TrimSpace(part), aarUsageAudioTokensExcludeCached):
				return audioUsageContractExcludesCached
			case strings.EqualFold(strings.TrimSpace(part), aarUsageAudioTokensIncludeCached):
				return audioUsageContractIncludesCached
			}
		}
	}
	return audioUsageContractUnknown
}

func tokenUsageExtractionOptionsForCredential(_ *config.CredentialConfig) converter.TokenUsageExtractionOptions {
	return converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
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
