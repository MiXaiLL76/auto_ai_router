package upstreamerror

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A provider that packs the status of the failure, its own error class and
// code, and the explanation meant for the client into ONE message, and
// answers HTTP 200 with the failure in the body:
//
//	{"error":{"code":"InvalidParameter",
//	          "message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}
//
// The router read neither the status nor a code it knew, so the request was
// answered 500 or 502 -- "Request failed", which reads like the provider
// being down and invites a retry on another credential when every credential
// refuses it identically.
//
// These are the cases as production produced them.
func TestClassifyBadRequest_ProviderPrefixedMessage(t *testing.T) {
	for _, tt := range []struct {
		name    string
		body    string
		message string
	}{
		{
			name:    "agent capabilities",
			body:    `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`,
			message: "Agent capabilities are not enabled for the current model.",
		},
		{
			name:    "pdf input",
			body:    `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: The current model does not support PDF file input."}}`,
			message: "The current model does not support PDF file input.",
		},
		{
			name:    "thinking budget",
			body:    `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: The thinking_budget parameter must be a positive integer and not greater than 131072"}}`,
			message: "The thinking_budget parameter must be a positive integer and not greater than 131072",
		},
		{
			// The same failure under a Responses-shaped body: the error sits
			// under "response", which is the shape Alibaba's native
			// Responses API answers with.
			name:    "responses shape",
			body:    `{"response":{"id":"resp_1","error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}}`,
			message: "Agent capabilities are not enabled for the current model.",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			classified := ClassifyBadRequest([]byte(tt.body))

			assert.Equal(t, "invalid_request", classified.Code)
			assert.Equal(t, tt.message, classified.Message)
			assert.Nil(t, classified.Param)

			// The provider's own words are not the client's business: the
			// code and the class are not repeated in the message.
			assert.NotContains(t, classified.Message, "InvalidParameter")
			assert.NotContains(t, classified.Message, "InternalError")
			assert.NotContains(t, classified.Message, "<400>")
		})
	}
}

// TestClassifyProviderErrorSignals_ProviderPrefixedStatus pins the status a
// provider states in its own message. It is taken at its word, and it
// outranks the keyword pass: the same message contains "internal" (which the
// keyword pass reads as a 500) while stating "<400>".
func TestClassifyProviderErrorSignals_ProviderPrefixedStatus(t *testing.T) {
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`
	signals, _ := providerErrorSignalsFromBody([]byte(body))
	require.NotEmpty(t, signals)

	assert.Equal(t, http.StatusBadRequest, ClassifyProviderErrorSignals(signals),
		"the status the provider stated is the status the client is answered with")
}

// TestClassifyBadRequest_ProviderPrefixedMessageIsIdempotent checks that the
// router's own answer survives a second classification pass: the LiteLLM
// compatibility mode re-classifies the body the router wrote, and an
// AIR-to-AIR chain hands one router what another already classified.
func TestClassifyBadRequest_ProviderPrefixedMessageIsIdempotent(t *testing.T) {
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`
	first := ClassifyBadRequest([]byte(body))

	routerBody, err := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": first.Message,
			"type":    "invalid_request_error",
			"param":   first.Param,
			"code":    first.Code,
		},
	})
	require.NoError(t, err)

	second := ClassifyBadRequest(routerBody)
	assert.Equal(t, first.Message, second.Message, "the message must survive a second pass")
	assert.Equal(t, first.Code, second.Code, "the code must survive a second pass")
}

// TestProviderMessageRule_OnlyAMatchIsRelayed is the safety guard. The
// router never hands a provider's words to the client as they are -- the
// provider's text is MATCHED, and the router's own answer is what reaches
// the client. An explanation that no rule names is not relayed at all.
//
// The texts below are the ones production carried: a request id, a URL, a
// host and a port, a path. A provider that starts writing an internal
// identifier into its explanations must not leak it into a client's response
// just by being new.
func TestProviderMessageRule_OnlyAMatchIsRelayed(t *testing.T) {
	for _, tt := range []struct {
		name string
		text string
	}{
		{"a path", "/home/user/app.py:42 failed"},
		{"a url", "see https://internal.example.com/docs for details"},
		{"a host and port", "connect to db-01.internal:5432 failed"},
		{"an email", "the account owner@example.com was not found"},
		{"a request id", "The parameter `size` specified in the request are not valid: image area must be at least 921600 pixels. Request id: 021795058711799a81c77d48bc6ef475723297888da3"},
		{"a long identifier", "request 8f3c2a1b4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f80 failed"},
		{"a stack frame", "goroutine 1 [running]: main.main()"},
		{"an unknown explanation", "Something new the provider says about the request."},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: ` + tt.text + `"}}`
			classified := ClassifyBadRequest([]byte(body))

			assert.NotEqual(t, "Agent capabilities are not enabled for the current model.", classified.Message)
			assert.NotContains(t, classified.Message, tt.text,
				"an explanation that is not matched must not reach the client")
			assert.NotContains(t, classified.Message, "InternalError")
			assert.NotContains(t, classified.Message, "<400>")
		})
	}
}

// TestProviderMessageRule_UnknownExplanationStillGetsTheStatus checks that
// an explanation the router does not know does not cost the client the
// diagnosis the provider DID state: the status is still the 400 the provider
// wrote, and only the wording falls back to the router's own.
func TestProviderMessageRule_UnknownExplanationStillGetsTheStatus(t *testing.T) {
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Something new the provider says about the request."}}`
	signals, _ := providerErrorSignalsFromBody([]byte(body))
	require.NotEmpty(t, signals)

	assert.Equal(t, http.StatusBadRequest, ClassifyProviderErrorSignals(signals))
}

// TestProviderMessageRule_OtherStatuses checks that the status a provider
// states is the one the client is answered with, for the statuses a provider
// states as a request's fault.
func TestProviderMessageRule_OtherStatuses(t *testing.T) {
	for _, tt := range []struct {
		status   string
		expected int
	}{
		{"400", http.StatusBadRequest},
		{"401", http.StatusUnauthorized},
		{"403", http.StatusForbidden},
		{"404", http.StatusNotFound},
		{"413", http.StatusRequestEntityTooLarge},
		{"429", http.StatusTooManyRequests},
	} {
		t.Run(tt.status, func(t *testing.T) {
			body := `{"error":{"code":"SomeError","message":"<` + tt.status + `> SomeError.Class: The request was refused."}}`
			signals, _ := providerErrorSignalsFromBody([]byte(body))
			status, ok := providerPrefixedStatus(signals)
			require.True(t, ok)
			assert.Equal(t, tt.expected, status)
		})
	}
}

// TestProviderMessageRule_NotPrefixed checks that a message without the
// provider's prefix states no status of its own: the ordinary keyword
// branches own those.
func TestProviderMessageRule_NotPrefixed(t *testing.T) {
	for _, message := range []string{
		"Rate limit exceeded",
		"Received Model Group=gpt-5 does not exist",
		"The model produced invalid JSON",
		"",
	} {
		t.Run(message, func(t *testing.T) {
			_, ok := providerPrefixedStatus([]string{message})
			assert.False(t, ok, "a message without the prefix states no status")
		})
	}
}

// TestProviderMessageRule_UnknownStatusIsNotTaken checks that a status the
// router does not relay is not taken as one: a provider's own numbering may
// differ from HTTP's.
func TestProviderMessageRule_UnknownStatusIsNotTaken(t *testing.T) {
	body := `{"error":{"code":"SomeError","message":"<799> SomeError.Class: Something happened."}}`
	signals, _ := providerErrorSignalsFromBody([]byte(body))
	_, ok := providerPrefixedStatus(signals)
	assert.False(t, ok, "a status the router does not relay must not be taken")
}

// TestProviderMessageRule_ProseIsNotAPrefixedMessage checks that a message
// that merely contains a colon is not read as a prefixed one: prose with a
// time in it ("failed at 12:30") and a URL must both be left to the ordinary
// keyword branches.
func TestProviderMessageRule_ProseIsNotAPrefixedMessage(t *testing.T) {
	for _, message := range []string{
		"failed at 12:30 while reading the request",
		"see https://example.com/docs: the request was refused",
		"InternalError: the request was refused",
	} {
		t.Run(message, func(t *testing.T) {
			_, ok := providerPrefixedStatus([]string{message})
			assert.False(t, ok, "prose must not be read as a prefixed message")
		})
	}
}

// TestProviderMessageRule_TextIsMatchedLoosely checks that a provider that
// changes only the capitalization or the spacing of an explanation it
// already wrote still gets the same answer.
func TestProviderMessageRule_TextIsMatchedLoosely(t *testing.T) {
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: AGENT CAPABILITIES ARE NOT ENABLED FOR THE CURRENT MODEL."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "Agent capabilities are not enabled for the current model.", classified.Message)
}

// TestProviderMessageRule_LongTextIsDropped checks that a provider's stack
// trace or a dump does not become the client's message.
func TestProviderMessageRule_LongTextIsDropped(t *testing.T) {
	long := strings.Repeat("a very long explanation ", 40)
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: ` + long + `"}}`
	classified := ClassifyBadRequest([]byte(body))
	assert.NotContains(t, classified.Message, "a very long explanation",
		"a provider's dump must not become the client's message")
}

// TestProviderMessageRule_ContentPolicyIsNotRelayed checks that a refusal
// is not relayed under the provider's own wording: the content-policy branch
// already names a refusal, and its name is what the retry policy and the
// client's own handling key off.
func TestProviderMessageRule_ContentPolicyIsNotRelayed(t *testing.T) {
	body := `{"error":{"code":"server_error","message":"<400> InternalError.Algo.DataInspectionFailed: Input text data may contain inappropriate content."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "content_policy_violation", classified.Code)
	assert.Equal(t, fixedErrorMessages["content_policy_violation"], classified.Message)
	assert.NotContains(t, classified.Message, "Input text data may contain inappropriate content.")
}

// TestSetProviderMessageRules_AddsARule checks that a deployment's own rule
// works: the provider's words are matched and the router's answer is what
// the client is told.
//
// The rules are global state, so the test puts the built-in ones back when
// it is done.
func TestSetProviderMessageRules_AddsARule(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	SetProviderMessageRules([]ProviderMessageRule{
		{
			ProviderCode: "InvalidParameter",
			Match:        "unsupported model",
			Code:         "invalid_model",
			Message:      "Invalid model",
			Status:       http.StatusBadRequest,
		},
	})

	body := `{"error":{"code":"InvalidParameter","message":"Unsupported model: 'kimi-k2.7-code'"}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "invalid_model", classified.Code)
	assert.Equal(t, "Invalid model", classified.Message)
	assert.NotContains(t, classified.Message, "kimi-k2.7-code",
		"the provider's own text must not reach the client")
}

// TestSetProviderMessageRules_RuleStatusIsTheStatus checks that a rule's own
// status is the one the client is answered with -- including when the
// provider stated nothing. A provider that answers 200 with a failure in the
// body would otherwise be a 502.
func TestSetProviderMessageRules_RuleStatusIsTheStatus(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	SetProviderMessageRules([]ProviderMessageRule{
		{
			ProviderCode: "InvalidParameter",
			Match:        "unsupported model",
			Code:         "invalid_model",
			Message:      "Invalid model",
			Status:       http.StatusBadRequest,
		},
	})

	body := `{"error":{"code":"InvalidParameter","message":"Unsupported model: 'kimi-k2.7-code'"}}`
	signals, _ := providerErrorSignalsFromBody([]byte(body))
	require.NotEmpty(t, signals)

	assert.Equal(t, http.StatusBadRequest, ClassifyProviderErrorSignals(signals),
		"the rule's status is the status the client is answered with")
}

// TestSetProviderMessageRules_ReplacesABuiltinRule checks that a deployment
// may reword what the router says about a failure it already knows: the
// rule with the same text replaces the built-in one instead of adding a
// second answer for the same failure.
func TestSetProviderMessageRules_ReplacesABuiltinRule(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	SetProviderMessageRules([]ProviderMessageRule{
		{
			Match:   "agent capabilities are not enabled for the current model.",
			Code:    "invalid_request",
			Message: "This model does not support agent capabilities.",
			Status:  http.StatusBadRequest,
		},
	})

	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "This model does not support agent capabilities.", classified.Message)
}

// TestSetProviderMessageRules_RuleWithoutAMatchDoesNotMatchEverything checks
// that a rule that names a code alone matches only that code: a failure
// with another code is not the failure the rule describes.
func TestSetProviderMessageRules_RuleWithoutAMatchDoesNotMatchEverything(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	SetProviderMessageRules([]ProviderMessageRule{
		{
			ProviderCode: "InvalidParameter",
			Code:         "invalid_request",
			Message:      "Invalid request parameter",
			Status:       http.StatusBadRequest,
		},
	})

	body := `{"error":{"code":"SomeOtherError","message":"Something went wrong"}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.NotEqual(t, "Invalid request parameter", classified.Message,
		"a rule for one code must not answer for another")
}

// TestSetProviderMessageRules_RuleWithTextOnlyMatchesThatText checks that a
// rule that names text alone matches only that text, whatever the code.
func TestSetProviderMessageRules_RuleWithTextOnlyMatchesThatText(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	SetProviderMessageRules([]ProviderMessageRule{
		{
			Match:   "unsupported model",
			Code:    "invalid_model",
			Message: "Invalid model",
			Status:  http.StatusBadRequest,
		},
	})

	body := `{"error":{"code":"SomeOtherError","message":"Unsupported model: 'kimi-k2.7-code'"}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "Invalid model", classified.Message)
}

// TestSetProviderMessageRules_MatchIsASubstring checks that a rule matches a
// provider's message that carries more than the rule's text: a provider
// that appends its own request id to the message still matches, and the
// request id is not relayed with it.
func TestSetProviderMessageRules_MatchIsASubstring(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	SetProviderMessageRules([]ProviderMessageRule{
		{
			Match:   "unsupported model",
			Code:    "invalid_model",
			Message: "Invalid model",
			Status:  http.StatusBadRequest,
		},
	})

	body := `{"error":{"code":"InvalidParameter","message":"Unsupported model: 'kimi-k2.7-code'. Request id: 021795058711799a81c77d48bc6ef475723297888da3"}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "Invalid model", classified.Message)
	assert.NotContains(t, classified.Message, "021795058711799a81c77d48bc6ef475723297888da3",
		"the provider's request id must not reach the client")
}
