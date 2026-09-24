package converterutil

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCachingTokensExtension_CacheWrite(t *testing.T) {
	tests := []struct {
		name                  string
		details               string
		total, fiveMin, oneHr int
	}{
		{name: "absent", details: `{"cached_tokens":5}`},
		{name: "1h write", details: `{"caching_tokens":26310,"caching_token_details":{"caching_1h_tokens":26310,"caching_5m_tokens":0}}`, total: 26310, oneHr: 26310},
		{name: "5m write", details: `{"caching_tokens":26314,"caching_token_details":{"caching_1h_tokens":0,"caching_5m_tokens":26314}}`, total: 26314, fiveMin: 26314},
		{name: "aggregate derived from split", details: `{"caching_token_details":{"caching_1h_tokens":10,"caching_5m_tokens":20}}`, total: 30, fiveMin: 20, oneHr: 10},
		{name: "aggregate without split", details: `{"caching_tokens":40}`, total: 40},
		{name: "negative counters ignored", details: `{"caching_tokens":-1,"caching_token_details":{"caching_1h_tokens":-5,"caching_5m_tokens":7}}`, total: 7, fiveMin: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var details struct {
				CachedTokens int `json:"cached_tokens"`
				CachingTokensExtension
			}
			require.NoError(t, json.Unmarshal([]byte(tt.details), &details))
			total, fiveMin, oneHr := details.CacheWrite()
			assert.Equal(t, tt.total, total)
			assert.Equal(t, tt.fiveMin, fiveMin)
			assert.Equal(t, tt.oneHr, oneHr)
		})
	}
}
