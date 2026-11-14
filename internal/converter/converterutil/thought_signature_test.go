package converterutil

import (
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToolCallIDSignature_RoundTrip(t *testing.T) {
	// Bytes chosen so that standard base64 would contain '+' and '/'.
	signature := []byte{0xfb, 0xff, 0xbf, 0x01, 0x02}
	id := EncodeToolCallIDWithSignature("call_abc", signature)

	assert.Regexp(t, `^[a-zA-Z0-9_-]+$`, id, "id must stay within the Anthropic tool_use.id charset")
	baseID, decoded := SplitToolCallIDSignature(id)
	assert.Equal(t, "call_abc", baseID)
	assert.Equal(t, signature, decoded)
}

func TestToolCallIDSignature_DecodesLiteLLMStandardBase64(t *testing.T) {
	signature := []byte{0xfb, 0xff, 0xbf, 0x01}
	id := "call_abc" + ThoughtSignatureSeparator + base64.StdEncoding.EncodeToString(signature)

	baseID, decoded := SplitToolCallIDSignature(id)
	assert.Equal(t, "call_abc", baseID)
	assert.Equal(t, signature, decoded)
}

func TestToolCallIDSignature_NoSignature(t *testing.T) {
	assert.Equal(t, "call_abc", EncodeToolCallIDWithSignature("call_abc", nil))

	for _, id := range []string{"", "call_abc", "call_abc" + ThoughtSignatureSeparator, "call_abc" + ThoughtSignatureSeparator + "!!!"} {
		baseID, decoded := SplitToolCallIDSignature(id)
		assert.Equal(t, id, baseID, "id %q must be returned unchanged", id)
		assert.Nil(t, decoded)
	}
}
