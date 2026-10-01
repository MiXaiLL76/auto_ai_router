package converterutil

import (
	"encoding/base64"
	"strings"
)

// ThoughtSignatureSeparator joins a tool call id and the Gemini thoughtSignature
// embedded into it. Same separator as LiteLLM (THOUGHT_SIGNATURE_SEPARATOR), so
// ids minted by either gateway decode on the other.
const ThoughtSignatureSeparator = "__thought__"

// SkipThoughtSignatureValidator is the dummy signature Gemini 3 accepts in place
// of a real one when the client did not round-trip it.
var SkipThoughtSignatureValidator = []byte("skip_thought_signature_validator")

// EncodeToolCallIDWithSignature appends the thoughtSignature to a tool call id.
// Most OpenAI-compatible clients drop provider_specific_fields when replaying an
// assistant turn but always echo the id back, so the id is the only place the
// signature reliably survives. Unpadded base64url keeps the id within the
// [a-zA-Z0-9_-] charset other providers (e.g. Anthropic tool_use.id) require.
func EncodeToolCallIDWithSignature(id string, signature []byte) string {
	if len(signature) == 0 {
		return id
	}
	return id + ThoughtSignatureSeparator + base64.RawURLEncoding.EncodeToString(signature)
}

// SplitToolCallIDSignature splits an id produced by EncodeToolCallIDWithSignature
// (or LiteLLM) into the base id and the decoded signature. An id without the
// separator, or with an undecodable suffix, is returned unchanged with a nil
// signature.
func SplitToolCallIDSignature(id string) (string, []byte) {
	idx := strings.Index(id, ThoughtSignatureSeparator)
	if idx < 0 {
		return id, nil
	}
	signature := decodeAnyBase64(id[idx+len(ThoughtSignatureSeparator):])
	if signature == nil {
		return id, nil
	}
	return id[:idx], signature
}

// decodeAnyBase64 accepts both the standard alphabet (LiteLLM, provider_specific_fields)
// and base64url (our own ids), padded or not.
func decodeAnyBase64(s string) []byte {
	if s == "" {
		return nil
	}
	for _, enc := range [...]*base64.Encoding{
		base64.RawURLEncoding, base64.URLEncoding, base64.StdEncoding, base64.RawStdEncoding,
	} {
		if decoded, err := enc.DecodeString(s); err == nil && len(decoded) > 0 {
			return decoded
		}
	}
	return nil
}
