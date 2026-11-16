package upstreamerror

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func stringPtr(s string) *string {
	return &s
}

func equalStringPtr(a, b *string) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func TestClassifyBadRequest(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantMessage string
		wantCode    string
		wantParam   *string
		notContains []string
	}{
		{
			name:        "invalid tool choice",
			body:        `{"error":{"message":"Provider rejected tool_choice. Debug trace abc123","type":"invalid_request_error","param":"tool_choice","code":"bad_request"}}`,
			wantMessage: "Invalid tool_choice",
			wantCode:    "invalid_tool_choice",
			wantParam:   stringPtr("tool_choice"),
			notContains: []string{"Provider rejected", "Debug trace"},
		},
		{
			name:        "max tokens",
			body:        `{"error":{"message":"max_completion_tokens must be less than 8192","param":"max_completion_tokens","code":"invalid_request_error"}}`,
			wantMessage: "Invalid max_completion_tokens",
			wantCode:    "invalid_max_tokens",
			wantParam:   stringPtr("max_completion_tokens"),
			notContains: []string{"8192"},
		},
		{
			name:        "context length",
			body:        `{"error":{"message":"Input is too long for the context window","code":"context_length_exceeded"}}`,
			wantMessage: "This model's maximum context length is exceeded by the request",
			wantCode:    "context_length_exceeded",
			wantParam:   stringPtr("input"),
		},
		{
			// vLLM: mentions "output tokens" too, must not land in invalid_max_tokens.
			name:        "vllm context length with output tokens",
			body:        `{"error":{"code":400,"message":"This model's maximum context length is 409600 tokens. However, you requested 32000 output tokens and your prompt contains at least 377601 input tokens, for a total of at least 409601 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=377601)","param":"input_tokens","type":"BadRequestError"}}`,
			wantMessage: "This model's maximum context length is 409600 tokens. However, you requested 32000 output tokens and your prompt contains at least 377601 input tokens.",
			wantCode:    "context_length_exceeded",
			wantParam:   stringPtr("input_tokens"),
			notContains: []string{"Please reduce", "parameter="},
		},
		{
			name:        "vllm legacy context length",
			body:        `{"object":"error","message":"This model's maximum context length is 32768 tokens. However, you requested 40000 tokens (38000 in the messages, 2000 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","code":400}`,
			wantMessage: "This model's maximum context length is 32768 tokens. However, you requested 2000 output tokens and your prompt contains at least 38000 input tokens.",
			wantCode:    "context_length_exceeded",
			wantParam:   stringPtr("messages"),
		},
		{
			name:        "invalid model",
			body:        `{"error":{"message":"litellm.BadRequestError: Received Model Group=secret/internal Available Model Group Fallbacks=None"}}`,
			wantMessage: "Invalid model",
			wantCode:    "invalid_model",
			wantParam:   stringPtr("model"),
			notContains: []string{"litellm", "secret/internal", "Fallbacks"},
		},
		{
			name:        "invalid parameter",
			body:        `{"error":{"message":"unsupported parameter response_format for this model","param":"response_format"}}`,
			wantMessage: "Invalid request parameter",
			wantCode:    "invalid_parameter",
			wantParam:   stringPtr("response_format"),
			notContains: []string{"unsupported parameter", "this model"},
		},
		{
			// Reproduces a real production body: DeepInfra-style
			// "is not supported" phrasing (not "does not support") with an
			// underscore-style error code ("invalid_parameter_error", not a
			// literal "invalid parameter" substring) — both must still land
			// in the invalid_parameter bucket instead of the generic fallback.
			name:        "is not supported phrasing with underscore error code",
			body:        "{\"error\":{\"code\":\"invalid_parameter_error\",\"message\":\"The parameters `logprobs` is not supported.\",\"param\":null,\"type\":\"invalid_request_error\"},\"id\":\"chatcmpl-c851a2c6-8a54-9ca4-9561-b3ff6163be27\"}",
			wantMessage: "Invalid request parameter",
			wantCode:    "invalid_parameter",
			wantParam:   stringPtr("logprobs"),
			notContains: []string{"chatcmpl", "c851a2c6"},
		},
		{
			// Reproduces another real production body: a PascalCase
			// "InvalidParameter" type (lowercases to "invalidparameter",
			// no space — doesn't match the "invalid parameter" needle) with
			// no "param" field of its own, and a dynamic/nested offending
			// field path ("output[1].type") that a fixed param-name list can
			// never enumerate in advance. Must extract the path quoted right
			// after "Invalid " in the message instead of leaving param null.
			name:        "PascalCase InvalidParameter with quoted dynamic field path",
			body:        `{"error":{"message":"Invalid 'output[1].type': 'input_file'. Supported values are: 'input_text', 'input_image'.","type":"InvalidParameter"}}`,
			wantMessage: "Invalid request parameter",
			wantCode:    "invalid_parameter",
			wantParam:   stringPtr("output[1].type"),
			notContains: []string{"Supported values are"},
		},
		{
			// Reproduces two real production cases (temperature out of range,
			// negative n) where the provider's own JSON names the offending
			// param explicitly but phrases the message in a way none of the
			// switch's keyword signals recognize. The structured param field
			// is still trustworthy — it's not a text guess — so this should
			// surface "Invalid <param>" instead of falling all the way to the
			// fully generic "Invalid request".
			name:        "known param, unrecognized message phrasing",
			body:        `{"error":{"message":"5 is greater than the maximum of 2","param":"temperature","code":"decimal_above_max_value"}}`,
			wantMessage: "Invalid temperature",
			wantCode:    "invalid_field",
			wantParam:   stringPtr("temperature"),
			notContains: []string{"is greater than", "decimal_above_max_value"},
		},
		{
			// A real production body, and the single largest classification
			// loss in the fleet: 71,764 requests in one fortnight against one
			// provider came back as exactly this -- a machine-readable
			// reason nested under "detail", with no wording at all. The
			// "detail" object WAS walked, but "error_type" was not among
			// the keys read inside it, so the body decoded to no signal and
			// the client got the generic "Invalid request" for 42% of all
			// 400s. Guarded here so the key cannot be dropped again.
			//
			// The code is "max_tokens_exceeded" and not "invalid_max_tokens":
			// every one of those 71,764 requests was an agent-loop turn
			// carrying a tool_calls history, and NOT ONE of them set
			// max_tokens -- the string appeared only as the NAME of a
			// tool-schema property. So there is no client-side parameter
			// that was wrong; the generation simply ran into the output
			// limit. Reporting "Invalid max_tokens" would point the client
			// at a parameter it never sent and never has to change.
			name:        "error_type nested under detail",
			body:        `{"detail":{"error_type":"max_tokens_exceeded"}}`,
			wantMessage: "Max tokens exceeded",
			wantCode:    "max_tokens_exceeded",
		},
		{
			// The client's OWN max_tokens value being rejected stays a
			// separate thing: "the value you sent is not allowed, change
			// it" is a different failure with a different fix from "the
			// answer was cut off because it ran long". The two must not
			// collapse into one code.
			name:        "client max_tokens value rejected stays invalid_max_tokens",
			body:        `{"error":{"message":"max_tokens is too large: 128000. Maximum allowed is 4096.","param":"max_tokens"}}`,
			wantMessage: "Invalid max_tokens",
			wantCode:    "invalid_max_tokens",
			wantParam:   stringPtr("max_tokens"),
		},
		{
			// "max_tokens_exceeded" contains the substring "max_tokens", so
			// it would otherwise fall into the generic max_tokens branch --
			// which reports the client's own max_tokens FIELD as invalid
			// ("Invalid max_tokens" inferred from the request). The client's
			// parameter is fine here: the provider's own ceiling is lower
			// than what was asked for, and the message has to say so.
			name:        "max_tokens_exceeded is not a context-window overflow",
			body:        `{"error":{"message":"Your input exceeds the model's maximum context length. Please shorten your messages and try again."}}`,
			wantMessage: "This model's maximum context length is exceeded by the request",
			wantCode:    "context_length_exceeded",
			wantParam:   stringPtr("messages"),
		},
		{
			// A real production body: 2,864 requests in one fortnight, all
			// against one model, all from a client that asked for
			// temperature=0 while the provider pins that model to a single
			// value. The wording contains none of the switch's keyword
			// signals ("must be", "invalid parameter", ...), so the request
			// fell through to the generic "Invalid request" and the client
			// could not tell which parameter to change -- the one thing the
			// provider stated plainly in the first two words.
			name:        "parameter pinned to a single value",
			body:        `{"error":{"message":"invalid temperature: only 1 is allowed for this model","type":"invalid_request_error"}}`,
			wantMessage: "Invalid temperature",
			wantCode:    "invalid_field",
			wantParam:   stringPtr("temperature"),
			notContains: []string{"only 1 is allowed"},
		},
		{
			// Same wording, a fractional pinned value -- the provider's
			// limit is not always an integer.
			name:        "parameter pinned to a fractional value",
			body:        `{"error":{"message":"invalid temperature: only 0.6 is allowed for this model","type":"invalid_request_error"}}`,
			wantMessage: "Invalid temperature",
			wantCode:    "invalid_field",
			wantParam:   stringPtr("temperature"),
		},
		{
			// A real production body (Google Vertex / Gemini, 1,050 requests
			// in one fortnight): the provider states the client's value AND
			// the range it accepts. The client's next action is a concrete
			// number, so the message carries one -- "ask for 65536, not
			// 200000" -- which "Invalid max_output_tokens" alone would not
			// tell it. Only the NUMBERS are taken from the provider's text;
			// the wording around them is this package's own, so no provider
			// free text reaches the client.
			name:        "output token value outside the supported range",
			body:        `{"error":{"code":400,"message":"Unable to submit request because it has a maxOutputTokens value of 200000 but the supported range is from 1 (inclusive) to 65537 (exclusive). Update the value and try again.","status":"INVALID_ARGUMENT"}}`,
			wantMessage: "Invalid max_output_tokens: requested 200000, but the supported range is from 1 to 65537.",
			wantCode:    "invalid_max_tokens",
			wantParam:   stringPtr("max_output_tokens"),
			notContains: []string{"Unable to submit", "Update the value", "inclusive", "exclusive"},
		},
		{
			// The same wording with a different requested value -- the
			// message must report the value the client actually sent, not a
			// fixed one.
			name:        "output token value outside the range, other value",
			body:        `{"error":{"code":400,"message":"Unable to submit request because it has a maxOutputTokens value of 115000 but the supported range is from 1 (inclusive) to 65537 (exclusive). Update the value and try again.","status":"INVALID_ARGUMENT"}}`,
			wantMessage: "Invalid max_output_tokens: requested 115000, but the supported range is from 1 to 65537.",
			wantCode:    "invalid_max_tokens",
			wantParam:   stringPtr("max_output_tokens"),
		},
		{
			// A real production body (OpenRouter, 468 requests in one
			// fortnight, all of them carrying reasoning_effort=none): a
			// provider that serves only reasoning models refuses a request
			// that turns reasoning off. The wording matches none of the
			// switch's keyword signals, so the request fell through to the
			// generic "Invalid request" and the client was told nothing --
			// while the fix (enable reasoning) was a single field it had
			// set to "none".
			name:        "reasoning cannot be disabled",
			body:        `{"error":{"code":400,"message":"Reasoning is mandatory for this endpoint and cannot be disabled."}}`,
			wantMessage: "Reasoning is required and cannot be disabled",
			wantCode:    "reasoning_required",
			wantParam:   stringPtr("reasoning_effort"),
			notContains: []string{"mandatory for this endpoint"},
		},
		{
			// The same refusal in Anthropic's spelling. The parameter the
			// client sets is named in the CLIENT's spelling too -- it is
			// the field the client has to change.
			name:        "thinking cannot be disabled",
			body:        `{"error":{"message":"Thinking is mandatory for this endpoint and cannot be disabled."}}`,
			wantMessage: "Reasoning is required and cannot be disabled",
			wantCode:    "reasoning_required",
			wantParam:   stringPtr("reasoning_effort"),
		},
		{
			// A DIFFERENT refusal that mentions reasoning: the model does
			// not support the parameter at all. That is the opposite
			// problem -- the client asked for reasoning on a model that
			// refuses it -- and must not be reported as "reasoning is
			// required".
			name:        "reasoning not supported is not reasoning required",
			body:        `{"error":{"message":"reasoning_effort is not supported for this model","param":"reasoning_effort"}}`,
			wantMessage: "Invalid request parameter",
			wantCode:    "invalid_parameter",
			wantParam:   stringPtr("reasoning_effort"),
		},
		{
			// The provider's wording belongs to the PARAMETER it named, not
			// to any parameter: one production request used the very same
			// "the supported range is from N to K" phrasing for temperature.
			// Matching on the phrase alone would report that one as an
			// output-token problem.
			name:        "supported range wording for a different parameter",
			body:        `{"error":{"message":"it has a temperature value of 5 but the supported range is from 1 to 65537"}}`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
		},
		{
			// No parsable numbers: the parameter is still named, so the
			// client learns what to fix even though it does not learn the
			// range.
			name:        "output token range without parsable numbers",
			body:        `{"error":{"message":"maxOutputTokens is unsupported for this model"}}`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
		},
		{
			// The parameter name comes out of the provider's own free text,
			// so it must not be passed through unchecked: an unknown name
			// is not reported at all, rather than smuggled into the
			// client's "param" field.
			name:        "unknown parameter name is not echoed",
			body:        `{"error":{"message":"invalid your-secret-setting: only 1 is allowed for this model"}}`,
			wantMessage: "Invalid request parameter",
			wantCode:    "invalid_parameter",
			notContains: []string{"your-secret-setting"},
		},
		{
			// A real production body: 1,163 requests in one fortnight that
			// carried NO safety_violations list, so the "safety violation"
			// marker did not match them and the refusal was reported as a
			// generic "Invalid request" -- telling the client to fix a
			// request that was actually refused on policy grounds, with
			// nothing to fix.
			name:        "safety system refusal without a violations list",
			body:        `{"error":{"message":"Your request was rejected by the safety system. If you believe this is an error, contact us at help.openai.com and include the request ID req_abc123.","type":"invalid_request_error"}}`,
			wantMessage: "The prompt was blocked by the provider's content policy",
			wantCode:    "content_policy_violation",
			notContains: []string{"help.openai.com", "req_abc123"},
		},
		{
			name:        "plain text fallback",
			body:        `vendor stack id 012345`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
			notContains: []string{"vendor stack"},
		},
		{
			name:        "empty body",
			body:        "",
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
		},
		{
			// The router's own prompt-block event (vertex writePromptBlockedEvent):
			// a content refusal must classify as a content-policy 400, never the
			// generic default, and must not echo the provider's block reason text.
			name:        "content policy violation",
			body:        `{"error":{"type":"invalid_request_error","code":"content_policy_violation","message":"The prompt was blocked by the provider's content policy (reason: PROHIBITED_CONTENT)"}}`,
			wantMessage: "The prompt was blocked by the provider's content policy",
			wantCode:    "content_policy_violation",
			notContains: []string{"PROHIBITED_CONTENT"},
		},
		{
			name:        "provider content policy wording",
			body:        `{"error":{"message":"The request was blocked by our content policy.","code":"invalid_request"}}`,
			wantMessage: "The prompt was blocked by the provider's content policy",
			wantCode:    "content_policy_violation",
			notContains: []string{"blocked by our"},
		},
		{
			// A bare "blocked by the provider" must NOT match: an account
			// billing block is not a content refusal, and swallowing it here
			// would replace the real cause with a misleading content-policy
			// message (this is a regression guard for that false positive).
			name:        "account blocked for billing is not content policy",
			body:        `{"error":{"message":"Your account has been blocked by the provider for suspected billing fraud.","code":"account_blocked"}}`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
			notContains: []string{"billing fraud", "content policy"},
		},
		{
			name:        "geo restriction is not content policy",
			body:        `{"error":{"message":"This request was blocked by the provider due to an IP/region restriction.","code":"region_restricted"}}`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
			notContains: []string{"content policy", "region restriction"},
		},
		{
			// The production case: an aggregator forwarded the upstream
			// provider's refusal text verbatim, next to the upstream's own
			// NUMERIC code (502) and error_type ("provider_unavailable").
			// The refusal must classify as a content-policy 400, and the
			// provider's text must not reach the client.
			name:        "aggregator forwarded inappropriate content refusal",
			body:        `{"error":{"code":502,"message":"Upstream error from Alibaba: Output data may contain inappropriate content.","metadata":{"error_type":"provider_unavailable"}}}`,
			wantMessage: "The prompt was blocked by the provider's content policy",
			wantCode:    "content_policy_violation",
			notContains: []string{"Alibaba", "inappropriate content", "provider_unavailable"},
		},
		{
			// The same wording without a content-refusal phrase must stay a
			// generic invalid_request -- the refusal detection must not turn
			// every provider failure into a content-policy answer.
			name:        "provider failure with numeric code is not content policy",
			body:        `{"error":{"code":502,"message":"Upstream error from Darkbloom: inference generation failed","metadata":{"error_type":"provider_unavailable"}}}`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
			notContains: []string{"content policy", "Darkbloom"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyBadRequest([]byte(tt.body))

			assert.Equal(t, tt.wantMessage, got.Message)
			assert.Equal(t, tt.wantCode, got.Code)
			assert.True(t, equalStringPtr(got.Param, tt.wantParam), "param = %v, want %v", got.Param, tt.wantParam)

			serialized, err := json.Marshal(got)
			require.NoError(t, err)
			for _, s := range tt.notContains {
				assert.NotContains(t, string(serialized), s)
			}
		})
	}
}

func TestExtractQuotedInvalidField(t *testing.T) {
	tests := []struct {
		name    string
		signals []string
		want    *string
	}{
		{
			name:    "dynamic nested path",
			signals: []string{"Invalid 'output[1].type': 'input_file'. Supported values are: 'input_text', 'input_image'."},
			want:    stringPtr("output[1].type"),
		},
		{
			name:    "case-insensitive marker",
			signals: []string{"invalid 'tool_choice.type': unsupported value"},
			want:    stringPtr("tool_choice.type"),
		},
		{
			name:    "picks first matching signal",
			signals: []string{"generic failure", "Invalid 'foo': bar"},
			want:    stringPtr("foo"),
		},
		{
			name:    "no marker present",
			signals: []string{"something else went wrong"},
			want:    nil,
		},
		{
			name:    "marker with no closing quote",
			signals: []string{"Invalid 'unterminated"},
			want:    nil,
		},
		{
			name:    "empty quoted field",
			signals: []string{"Invalid '': something"},
			want:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractQuotedInvalidField(tt.signals)
			if !equalStringPtr(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

// TestClassifyBadRequest_NeverEchoesRawSensitiveText guards the whole point
// of this package: even when a raw, untrusted provider body contains
// sensitive detail (credential names, internal URLs), ClassifyBadRequest
// must never let it reach the output — every branch either returns a fixed
// message or a narrow, character-restricted param/field name.
func TestClassifyBadRequest_NeverEchoesRawSensitiveText(t *testing.T) {
	body := `{"error":{"message":"Alibaba credential prod-qwen throttled; see https://provider.internal/quota","type":"provider_error","code":"quota"}}`
	got := ClassifyBadRequest([]byte(body))

	serialized, err := json.Marshal(got)
	require.NoError(t, err)
	for _, s := range []string{"Alibaba", "prod-qwen", "provider.internal"} {
		assert.NotContains(t, string(serialized), s)
	}
}

// TestClassifyBadRequest_IsIdempotent guards the property litellm
// compatibility mode's normalizeError depends on: Transform only ever sees
// the router's own already-classified body (see
// responseCompatibilityWriter.Close), so it re-runs ClassifyBadRequest on
// that body rather than trusting it blindly. Every branch's own canonical
// output must re-classify back into itself — message and code identical,
// param preserved — or a fix like PR #187 silently regresses back to the
// generic default despite having classified correctly the first time (this
// is exactly what happened before "invalid_parameter" was added as a
// needle: the "invalid_parameter" code round-tripped to the generic
// default because "Invalid request parameter" doesn't itself contain
// "invalid parameter" as a substring — "request" sits in between).
func TestClassifyBadRequest_IsIdempotent(t *testing.T) {
	rawBodies := []string{
		`{"error":{"message":"Provider rejected tool_choice.","param":"tool_choice"}}`,
		`{"error":{"message":"max_completion_tokens must be less than 8192","param":"max_completion_tokens"}}`,
		`{"error":{"message":"Input is too long for the context window"}}`,
		`{"error":{"message":"This model's maximum context length is 409600 tokens. However, you requested 32000 output tokens and your prompt contains at least 377601 input tokens.","param":"input_tokens"}}`,
		`{"error":{"message":"litellm.BadRequestError: Received Model Group=x"}}`,
		`{"error":{"message":"The parameters logprobs is not supported.","code":"invalid_parameter_error"}}`,
		`{"error":{"message":"Invalid 'output[1].type': 'input_file'.","type":"InvalidParameter"}}`,
		`{"error":{"message":"5 is greater than the maximum of 2","param":"temperature","code":"decimal_above_max_value"}}`,
		`{"detail":{"error_type":"max_tokens_exceeded"}}`,
		`{"error":{"code":400,"message":"Unable to submit request because it has a maxOutputTokens value of 200000 but the supported range is from 1 (inclusive) to 65537 (exclusive). Update the value and try again.","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"Reasoning is mandatory for this endpoint and cannot be disabled."}}`,
		`{"error":{"message":"invalid temperature: only 1 is allowed for this model","type":"invalid_request_error"}}`,
		`{"error":{"message":"Your request was rejected by the safety system. contact us at help.openai.com","type":"invalid_request_error"}}`,
		`vendor stack id 012345`,
	}

	for _, raw := range rawBodies {
		t.Run(raw, func(t *testing.T) {
			first := ClassifyBadRequest([]byte(raw))

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

			assert.Equal(t, first.Message, second.Message, "message must survive a second classification pass")
			assert.Equal(t, first.Code, second.Code, "code must survive a second classification pass")
			assert.True(t, equalStringPtr(first.Param, second.Param), "param must survive a second classification pass: first=%v second=%v", first.Param, second.Param)
		})
	}
}
