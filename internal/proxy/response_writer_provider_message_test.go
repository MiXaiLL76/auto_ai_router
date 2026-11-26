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

// TestStatusCodeFromProviderBodyError_NestedProviderError checks that a
// provider's error nested one level down is read: production carried the
// shape {"error":{"detail":{...}}}, and a reader that stopped at the first
// level found no code, no message and no status in it -- so the failure fell
// through to the generic 502 and the rule that names it went unused.
func TestStatusCodeFromProviderBodyError_NestedProviderError(t *testing.T) {
	body := []byte(`{"error":{"detail":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}`)

	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok, "the nested error must be recognized")
	assert.Equal(t, http.StatusBadRequest, status)
}

// TestMaskedUpstreamErrorBody_RuleForANon400Status checks that a rule
// asking for a 404 is honored with its own message: the status and the
// message come from ONE rule, so a rule that sets a status cannot lose the
// message that goes with it.
func TestMaskedUpstreamErrorBody_RuleForANon400Status(t *testing.T) {
	t.Cleanup(func() { upstreamerror.SetProviderMessageRules(nil) })

	upstreamerror.SetProviderMessageRules([]upstreamerror.ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "insufficient permission",
		Code:         "invalid_request",
		Message:      "Caller lacks permission",
		Status:       http.StatusNotFound,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter","message":"insufficient permission to call this resource"}}`)

	masked := maskedUpstreamErrorBody(http.StatusNotFound, "req-1", body)
	assert.Contains(t, string(masked), "Caller lacks permission",
		"the rule's message must reach the client")
}

// TestShouldRetry_RuleForbidsTheRetry checks that a rule saying the failure
// must not be replayed keeps the router from replaying it.
//
// The status alone does not say this: a 400 is in the router's retryable set
// (a provider account may lack a model another account serves), so without
// the rule's own answer the request is asked of every credential in the
// pool. Production measured it: 441 requests, 1 236 attempts, not one
// attempt succeeded.
func TestShouldRetry_RuleForbidsTheRetry(t *testing.T) {
	t.Cleanup(func() { upstreamerror.SetProviderMessageRules(nil) })

	notRetryable := false
	upstreamerror.SetProviderMessageRules([]upstreamerror.ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "agent capabilities are not enabled for the current model",
		Code:         "invalid_request",
		Message:      "Agent capabilities are not enabled for the current model.",
		Status:       http.StatusBadRequest,
		Retryable:    &notRetryable,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`)

	retry, _ := ShouldRetryWithFallback(http.StatusBadRequest, body)
	assert.False(t, retry, "the rule forbids the replay")
}

// TestShouldRetry_RuleAllowsTheRetry checks the other direction: a rule that
// says the failure MAY be replayed keeps it replayable even when the status
// alone would not say so.
func TestShouldRetry_RuleAllowsTheRetry(t *testing.T) {
	t.Cleanup(func() { upstreamerror.SetProviderMessageRules(nil) })

	retryable := true
	upstreamerror.SetProviderMessageRules([]upstreamerror.ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "the model is not served by this account",
		Code:         "invalid_model",
		Message:      "Invalid model",
		Status:       http.StatusBadRequest,
		Retryable:    &retryable,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter","message":"the model is not served by this account"}}`)

	retry, _ := ShouldRetryWithFallback(http.StatusBadRequest, body)
	assert.True(t, retry, "the rule allows the replay")
}

// TestShouldRetry_RuleForbidsTheRetryOnTheStreamingPath checks the same on
// the streaming path: a terminal error event is classified the same way, and
// the rule's answer must reach that path too.
func TestShouldRetry_RuleForbidsTheRetryOnTheStreamingPath(t *testing.T) {
	t.Cleanup(func() { upstreamerror.SetProviderMessageRules(nil) })

	notRetryable := false
	upstreamerror.SetProviderMessageRules([]upstreamerror.ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "agent capabilities are not enabled for the current model",
		Code:         "invalid_request",
		Message:      "Agent capabilities are not enabled for the current model.",
		Status:       http.StatusBadRequest,
		Retryable:    &notRetryable,
	}})

	payload := `{"type":"response.failed","response":{"status":"failed","error":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}`

	status := statusCodeFromProviderStreamError(payload)
	require.Equal(t, http.StatusBadRequest, status)

	retry, _ := ShouldRetryWithFallback(status, []byte(payload))
	assert.False(t, retry, "the rule forbids the replay on the streaming path too")
}
