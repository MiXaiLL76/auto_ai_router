package proxy

import (
	"testing"

	goccyjson "github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractSessionID_PromptCacheKeyBeforeClientID(t *testing.T) {
	var body map[string]goccyjson.RawMessage
	require.NoError(t, goccyjson.Unmarshal([]byte(`{"user":"u","safety_identifier":"s","prompt_cache_key":"conv-1"}`), &body))
	assert.Equal(t, "conv-1", extractSessionIDFromRawBody(body))
	assert.Equal(t, "u", extractClientIDFromRawBody(body))
}
