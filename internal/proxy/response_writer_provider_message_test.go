package proxy

import (
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/upstreamerror"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A provider that packs the status of the failure, its own error class and
// code, and the explanation meant for the client into ONE message, and
// answers HTTP 200 with the failure in the body:
//
//	{"response":{"id":"resp_1","error":{"code":"InvalidParameter",
//	          "message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}
//
// The router read neither the status nor a code it knew, so the request was
// answered 500 or 502 -- "Request failed", which reads like the provider
// being down and invites a retry on another credential when every credential
// refuses it identically.
//
// These are the cases as production produced them.
func TestStatusCodeFromProviderBodyError_ProviderPrefixedMessage(t *testing.T) {
	body := []byte(`{"response":{"id":"resp_1","error":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}`)

	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok, "the error inside a 200 body must be recognized")
	assert.Equal(t, http.StatusBadRequest, status,
		"the status the provider stated is the status the client is answered with")
}

// TestMaskedUpstreamErrorBody_ProviderPrefixedMessage pins what the client
// actually sees: the provider's own explanation, and nothing else of the
// provider's -- not its error class, not its code, not the status prefix.
func TestMaskedUpstreamErrorBody_ProviderPrefixedMessage(t *testing.T) {
	body := []byte(`{"response":{"id":"resp_1","error":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}`)

	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok)

	masked := maskedUpstreamErrorBody(status, "req-1", body)
	assert.Contains(t, string(masked), "Agent capabilities are not enabled for the current model.",
		"the client must be told what the provider said")
	assert.NotContains(t, string(masked), "InternalError",
		"the provider's own error class is not the client's business")
	assert.NotContains(t, string(masked), "<400>",
		"the provider's status prefix is not the client's business")
}

// TestStatusCodeFromProviderStreamError_ProviderPrefixedMessage is the
// streaming counterpart: the same failure arriving as a terminal stream
// error event.
func TestStatusCodeFromProviderStreamError_ProviderPrefixedMessage(t *testing.T) {
	payload := `{"type":"response.failed","response":{"status":"failed","error":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}`

	extracted := extractStreamErrorEvent([]byte("data: " + payload + "\n\n"))
	require.NotEmpty(t, extracted, "the terminal error event must be extracted")

	assert.Equal(t, http.StatusBadRequest, statusCodeFromProviderStreamError(extracted))
}

// TestStatusCodeFromProviderBodyError_ConfiguredRule checks that a rule a
// deployment configures works end to end: the provider's words are matched,
// the router's answer is what the client is told, and the provider's own
// text does not reach it.
//
// This is the path a provider message takes when it has no status prefix at
// all -- the provider answers 200 with the failure in the body, and only the
// rule says what the failure is worth.
func TestStatusCodeFromProviderBodyError_ConfiguredRule(t *testing.T) {
	t.Cleanup(func() { upstreamerror.SetProviderMessageRules(nil) })

	upstreamerror.SetProviderMessageRules([]upstreamerror.ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "unsupported model",
		Code:         "invalid_model",
		Message:      "Invalid model",
		Status:       http.StatusBadRequest,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter","message":"Unsupported model: 'kimi-k2.7-code'"}}`)

	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, status,
		"the rule's status is the status the client is answered with")

	masked := maskedUpstreamErrorBody(status, "req-1", body)
	assert.Contains(t, string(masked), "Invalid model")
	assert.NotContains(t, string(masked), "kimi-k2.7-code",
		"the provider's own text must not reach the client")
}
