package vertexresponses

import (
	"encoding/base64"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"google.golang.org/genai"
)

// A Gemini function call comes with an ID and, from Gemini 3 on, a thought
// signature, and both have to go back with the call on the next turn: Gemini 4
// pairs a function response with its call by ID, and a call without its signature
// is refused. The Responses API has a call_id and no field for a signature, so for
// such a model (see vertex.UsesFunctionCallIDs) the signature rides in the
// call_id, which clients return verbatim (with function_call_output too):
// "<id>__thought__<signature, base64url>". Another model's call_id is the bare ID.
const callIDSignatureSeparator = "__thought__"

// skipThoughtSignature is the stand-in Gemini accepts for the signature of a call
// that has none -- one written by the client, or by another model -- as the chat
// route sends it (vertex.convertToolCallsToGenaiParts).
var skipThoughtSignature = []byte("skip_thought_signature_validator")

// encodeCallID returns the call_id the client sees for a function call made by
// model, the model that answered: the call's ID, with the signature when model
// pairs a response with its call by ID.
func encodeCallID(call *genai.FunctionCall, signature []byte, model string) string {
	id := call.ID
	if id == "" {
		id = generateItemID("call_")
	}
	if len(signature) == 0 || !vertex.UsesFunctionCallIDs(model) {
		return id
	}
	return id + callIDSignatureSeparator + base64.RawURLEncoding.EncodeToString(signature)
}

// decodeCallID splits a call_id into the function call's ID and its thought
// signature (nil when the call_id carries none).
func decodeCallID(callID string) (string, []byte) {
	id, encoded, found := strings.Cut(callID, callIDSignatureSeparator)
	if !found || encoded == "" {
		return callID, nil
	}
	signature, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		if signature, err = base64.StdEncoding.DecodeString(encoded); err != nil {
			return callID, nil
		}
	}
	return id, signature
}
