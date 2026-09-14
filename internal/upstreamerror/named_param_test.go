package upstreamerror

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Providers that name the offending parameter in prose (no structured "param")
// must not have it guessed from unrelated words later in the message, such as
// "image size" or "not supported by the model".
func TestClassifyBadRequest_ParameterNamedInMessage(t *testing.T) {
	for _, tt := range []struct {
		name, body, code, param string
	}{
		{
			name:  "parameter with backticks and image wording",
			body:  "{\"error\":{\"code\":\"InvalidParameter\",\"message\":\"The parameter `size` specified in the request is not valid: image size must be at least 3686400 pixels. Request id: 0217\",\"param\":\"\",\"type\":\"BadRequest\"}}",
			code:  "invalid_parameter",
			param: "size",
		},
		{
			name:  "plain parameter name and model wording",
			body:  `{"error":{"code":"InvalidParameter","message":"The parameter size specified in the request is not valid: 1K is not supported by the model. Request id: 0217","type":"BadRequest"}}`,
			code:  "invalid_parameter",
			param: "size",
		},
		{
			name:  "specified parameter with dotted path",
			body:  `{"error":{"code":"InvalidParameter","message":"The specified parameter image_config.aspect_ratio is invalid.","type":"BadRequest"}}`,
			code:  "invalid_parameter",
			param: "image_config.aspect_ratio",
		},
		{
			name:  "parameter with assigned value",
			body:  `{"error":{"code":"InvalidParameter.UnsupportedParameter","message":"The parameter service_tier=flex specified in the request is not supported by this model version.","type":"BadRequest"}}`,
			code:  "invalid_parameter",
			param: "service_tier",
		},
		{
			name:  "argument not supported",
			body:  `{"code":"400","error":"Argument not supported: size"}`,
			code:  "invalid_parameter",
			param: "size",
		},
		{
			name:  "plural parameters names nothing",
			body:  `{"error":{"code":"InvalidParameter","message":"One or more parameters specified in the request are not valid. Request ID: 0217","type":"BadRequest"}}`,
			code:  "invalid_parameter",
			param: "",
		},
		{
			name:  "generic word after parameter is not a name",
			body:  `{"error":{"message":"Invalid parameter: value must be positive"}}`,
			code:  "invalid_parameter",
			param: "",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(tt.body)
			for range 3 {
				got := ClassifyBadRequest(body)
				assert.Equal(t, "Invalid request parameter", got.Message)
				assert.Equal(t, tt.code, got.Code)
				if tt.param == "" {
					assert.Nil(t, got.Param)
				} else {
					require.NotNil(t, got.Param)
					assert.Equal(t, tt.param, *got.Param)
				}
				var err error
				body, err = json.Marshal(map[string]any{"error": map[string]any{"message": got.Message, "code": got.Code, "param": got.Param}})
				require.NoError(t, err)
				assert.NotContains(t, string(body), "Request id")
				assert.NotContains(t, string(body), "3686400")
			}
		})
	}
}

func TestExtractNamedParameterField(t *testing.T) {
	for _, tt := range []struct {
		signal string
		want   string
	}{
		{"The parameter `size` specified in the request is not valid", "size"},
		{"The parameter 'n' specified in the request is not valid", "n"},
		{"The required parameter prompt is missing.", "prompt"},
		{"The parameter(s) watermark specified in the request is/are not valid.", "watermark"},
		{"The specified parameter image_config.aspect_ratio is invalid.", "image_config.aspect_ratio"},
		{"The parameter service_tier=flex specified in the request is not supported", "service_tier"},
		{"Invalid parameter: response_format", "response_format"},
		{"Argument not supported: quality", "quality"},
		{"InvalidParameter", ""},
		{"Invalid parameter value", ""},
		{"Invalid parameter: value must be positive", ""},
		{"parameters are invalid", ""},
		{"The parameter", ""},
		{"The specified parameter size.", ""},
		{"Error in parameter parsing", ""},
		{"Failed to read parameter payload from the request", ""},
	} {
		t.Run(tt.signal, func(t *testing.T) {
			got := extractNamedParameterField([]string{tt.signal})
			if tt.want == "" {
				assert.Nil(t, got)
				return
			}
			require.NotNil(t, got)
			assert.Equal(t, tt.want, *got)
		})
	}
}

// Plain-text bodies are used as signals verbatim and long messages are cut at a
// byte limit, so a signal can carry invalid UTF-8 before the parameter name.
func TestExtractNamedParameterField_InvalidUTF8(t *testing.T) {
	for _, signal := range []string{
		"\xff\xfe\xfd The parameter size specified in the request is not valid",
		"\xe2\x82 truncated rune, then parameter `n` is invalid",
		"The parameter \xff",
	} {
		t.Run(signal, func(t *testing.T) {
			require.NotPanics(t, func() { extractNamedParameterField([]string{signal}) })
		})
	}
	got := extractNamedParameterField([]string{"\xff\xfe\xfd The parameter size specified in the request is not valid"})
	require.NotNil(t, got)
	assert.Equal(t, "size", *got)

	body := []byte("\xff\xfe\xfd The parameter size specified in the request is not valid")
	require.NotPanics(t, func() { ClassifyBadRequest(body) })
}

// The keyword fallback matches parameter names as whole words only. A plural
// in prose ("reasoning models", "these models") describes the request, not the
// rejected field, so it must not surface as param "model"; a message that is
// about the model itself is classified by the model branch instead.
func TestClassifyBadRequest_PluralProseIsNotAParameter(t *testing.T) {
	for _, tt := range []struct {
		name, body, message, param string
	}{
		{
			name:    "plural models in a parameter message",
			body:    `{"error":{"message":"This value is not supported for reasoning models."}}`,
			message: "Invalid request parameter",
		},
		{
			name:    "plural models in a context length message",
			body:    `{"error":{"message":"Context length exceeded for these models"}}`,
			message: "Context length exceeded",
		},
		{
			name:    "unsupported models is a model error",
			body:    `{"error":{"message":"unsupported models requested"}}`,
			message: "Invalid model",
			param:   "model",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyBadRequest([]byte(tt.body))
			assert.Equal(t, tt.message, got.Message)
			if tt.param == "" {
				assert.Nil(t, got.Param)
				return
			}
			require.NotNil(t, got.Param)
			assert.Equal(t, tt.param, *got.Param)
		})
	}
}
