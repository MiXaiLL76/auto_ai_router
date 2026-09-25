package converterutil

// CachingTokensExtension is Requesty's naming for cache writes in
// prompt_tokens_details; prompt_tokens includes them, as with
// cache_creation_tokens. Embedding it keeps the fields on re-encode.
type CachingTokensExtension struct {
	CachingTokens       int                  `json:"caching_tokens,omitempty"`
	CachingTokenDetails *CachingTokenDetails `json:"caching_token_details,omitempty"`
}

type CachingTokenDetails struct {
	Caching5mTokens int `json:"caching_5m_tokens,omitempty"`
	Caching1hTokens int `json:"caching_1h_tokens,omitempty"`
}

// CachingWrite returns the total and its 5m / 1h split. Negative counters count
// as absent; a missing total is the sum of the split.
func (e CachingTokensExtension) CachingWrite() (total, fiveMinutes, oneHour int) {
	total = NonNegativeTokenCount(e.CachingTokens)
	if e.CachingTokenDetails != nil {
		fiveMinutes = NonNegativeTokenCount(e.CachingTokenDetails.Caching5mTokens)
		oneHour = NonNegativeTokenCount(e.CachingTokenDetails.Caching1hTokens)
	}
	if total == 0 {
		total = fiveMinutes + oneHour
	}
	return total, fiveMinutes, oneHour
}
