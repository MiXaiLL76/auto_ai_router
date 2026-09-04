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
			wantMessage: "Context length exceeded",
			wantCode:    "context_length_exceeded",
			wantParam:   stringPtr("input"),
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
		`{"error":{"message":"litellm.BadRequestError: Received Model Group=x"}}`,
		`{"error":{"message":"The parameters logprobs is not supported.","code":"invalid_parameter_error"}}`,
		`{"error":{"message":"Invalid 'output[1].type': 'input_file'.","type":"InvalidParameter"}}`,
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
