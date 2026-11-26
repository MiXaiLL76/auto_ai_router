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
// code, and the explanation meant for the client into ONE message:
//
//	{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}
//
// The router read neither the status nor a code it knew, so the request was
// answered 500 or 502 -- "Request failed", which reads like the provider
// being down and invites a retry on another credential when every credential
// would refuse it identically.
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

// TestProviderPrefixedMessage_OnlyVettedExplanationsAreRelayed is the safety
// guard. The explanation after the colon is the provider's own text, and
// this package never hands a provider's words to the client as they are --
// only the explanations that were read and vetted one by one. Everything else
// falls back to the router's own wording.
//
// The texts below are the ones production carried: a request id, a URL, a
// host and a port, a path. A provider that starts writing an internal
// identifier into its explanations must not leak it into a client's response
// just by being new.
//
// The fallback is whatever the ordinary classification makes of the body --
// here "Invalid request parameter", because the provider's own code says
// "InvalidParameter". What matters is that none of the provider's own
// wording reaches the client.
func TestProviderPrefixedMessage_OnlyVettedExplanationsAreRelayed(t *testing.T) {
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
				"an explanation that is not vetted must not reach the client")
			assert.NotContains(t, classified.Message, "InternalError")
			assert.NotContains(t, classified.Message, "<400>")
		})
	}
}

// TestProviderPrefixedMessage_UnknownExplanationStillGetsTheStatus checks that
// an explanation the router does not know does not cost the client the
// diagnosis the provider DID state: the status is still the 400 the provider
// wrote, and only the wording falls back to the router's own.
func TestProviderPrefixedMessage_UnknownExplanationStillGetsTheStatus(t *testing.T) {
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Something new the provider says about the request."}}`
	signals, _ := providerErrorSignalsFromBody([]byte(body))
	require.NotEmpty(t, signals)

	assert.Equal(t, http.StatusBadRequest, ClassifyProviderErrorSignals(signals))
}

// TestProviderPrefixedMessage_OtherStatuses checks that the status a
// provider states is the one the client is answered with, for the statuses a
// provider states as a request's fault.
func TestProviderPrefixedMessage_OtherStatuses(t *testing.T) {
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

// TestProviderPrefixedMessage_NotPrefixed checks that a message without the
// provider's prefix is left alone: the ordinary keyword branches own those.
func TestProviderPrefixedMessage_NotPrefixed(t *testing.T) {
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

// TestProviderPrefixedMessage_UnknownStatusIsNotTaken checks that a status
// the router does not relay is not taken as one: a provider's own numbering
// may differ from HTTP's.
func TestProviderPrefixedMessage_UnknownStatusIsNotTaken(t *testing.T) {
	body := `{"error":{"code":"SomeError","message":"<799> SomeError.Class: Something happened."}}`
	signals, _ := providerErrorSignalsFromBody([]byte(body))
	_, ok := providerPrefixedStatus(signals)
	assert.False(t, ok, "a status the router does not relay must not be taken")
}

// TestProviderPrefixedMessage_StatusWithoutTheFullShape checks that the
// status is taken even when the rest of the provider's shape is missing: a
// provider that writes "<400> Request rejected" states the same thing as one
// that writes "<400> InternalError.Algo.InvalidParameter: ...", and the
// diagnosis must not be lost for a wording the provider changed.
func TestProviderPrefixedMessage_StatusWithoutTheFullShape(t *testing.T) {
	for _, tt := range []struct {
		name string
		body string
	}{
		{
			name: "no explanation at all",
			body: `{"error":{"code":"SomeError","message":"<400> Request rejected"}}`,
		},
		{
			name: "a code but no colon",
			body: `{"error":{"code":"<400> InvalidParameter","message":"Agent capabilities are not enabled for the current model."}}`,
		},
		{
			name: "the prefix in the code field",
			body: `{"error":{"code":"<400> InvalidParameter","message":"Agent capabilities are not enabled for the current model."}}`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			signals, _ := providerErrorSignalsFromBody([]byte(tt.body))
			require.NotEmpty(t, signals)

			status, ok := providerPrefixedStatus(signals)
			require.True(t, ok, "the status the provider stated must be taken")
			assert.Equal(t, http.StatusBadRequest, status)
		})
	}
}

// TestProviderPrefixedMessage_StatusWithoutTheFullShapeIsNotRelayed checks
// that a status taken without the full shape is not also RELAYED: the
// explanation that accompanies it was never vetted, so it does not reach the
// client.
func TestProviderPrefixedMessage_StatusWithoutTheFullShapeIsNotRelayed(t *testing.T) {
	body := `{"error":{"code":"<400> InvalidParameter","message":"Agent capabilities are not enabled for the current model."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.NotEqual(t, "Agent capabilities are not enabled for the current model.", classified.Message,
		"an explanation outside the full shape was never vetted")
}

// TestProviderPrefixedMessage_ProseIsNotAPrefixedMessage checks that a
// message that merely contains a colon is not read as a prefixed one: prose
// with a time in it ("failed at 12:30") and a URL must both be left to the
// ordinary keyword branches.
func TestProviderPrefixedMessage_ProseIsNotAPrefixedMessage(t *testing.T) {
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

// TestProviderPrefixedMessage_ExplanationIsMatchedLoosely checks that a
// provider that changes only the capitalization or the spacing of an
// explanation it already wrote still gets the same answer.
func TestProviderPrefixedMessage_ExplanationIsMatchedLoosely(t *testing.T) {
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: AGENT CAPABILITIES ARE NOT ENABLED FOR THE CURRENT MODEL."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "Agent capabilities are not enabled for the current model.", classified.Message)
}

// TestProviderPrefixedMessage_LongTextIsDropped checks that a provider's
// stack trace or a dump does not become the client's message.
func TestProviderPrefixedMessage_LongTextIsDropped(t *testing.T) {
	long := strings.Repeat("a very long explanation ", 40)
	body := `{"error":{"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: ` + long + `"}}`
	classified := ClassifyBadRequest([]byte(body))
	assert.NotContains(t, classified.Message, "a very long explanation",
		"a provider's dump must not become the client's message")
}

// TestProviderPrefixedMessage_ContentPolicyIsNotRelayed checks that a refusal
// is not relayed a second time under the provider's own wording: the
// content-policy branch already names a refusal, and its name is what the
// retry policy and the client's own handling key off.
func TestProviderPrefixedMessage_ContentPolicyIsNotRelayed(t *testing.T) {
	body := `{"error":{"code":"server_error","message":"<400> InternalError.Algo.DataInspectionFailed: Input text data may contain inappropriate content."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.Equal(t, "content_policy_violation", classified.Code)
	assert.Equal(t, fixedErrorMessages["content_policy_violation"], classified.Message)
	assert.NotContains(t, classified.Message, "Input text data may contain inappropriate content.")
}

// TestProviderPrefixedMessage_NonRequestStatusIsNotRelayed checks that only
// a request's own fault is answered with the provider's explanation: a
// status the router has a message of its own for (a rate limit, a timeout)
// keeps that message.
func TestProviderPrefixedMessage_NonRequestStatusIsNotRelayed(t *testing.T) {
	body := `{"error":{"code":"SomeError","message":"<429> SomeError.Class: Agent capabilities are not enabled for the current model."}}`
	classified := ClassifyBadRequest([]byte(body))

	assert.NotEqual(t, "Agent capabilities are not enabled for the current model.", classified.Message,
		"a non-400 status keeps the router's own wording")
}
