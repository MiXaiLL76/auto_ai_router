package converterutil

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRequestJSONValidationError_TypeMismatch covers the case this helper exists for: a
// client sends a field as the wrong JSON type (e.g. max_tokens as a string instead of a
// number). The resulting error must classify as a *RequestValidationError naming the
// offending field, not an opaque error a caller might let fall through to a 500.
func TestRequestJSONValidationError_TypeMismatch(t *testing.T) {
	type req struct {
		MaxTokens *int `json:"max_tokens,omitempty"`
	}
	var r req
	unmarshalErr := json.Unmarshal([]byte(`{"max_tokens":"five"}`), &r)
	require.Error(t, unmarshalErr)

	err := RequestJSONValidationError(unmarshalErr)

	var validationErr *RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "max_tokens", validationErr.Param)
	assert.Equal(t, "invalid_type", validationErr.Code)
	assert.Equal(t, "Invalid parameter type", validationErr.Message)
}

// TestRequestJSONValidationError_MalformedJSON covers a syntax error (not a type
// mismatch on a known field) -- still a validation error, just without a specific param.
func TestRequestJSONValidationError_MalformedJSON(t *testing.T) {
	var r struct{}
	unmarshalErr := json.Unmarshal([]byte(`{not valid json`), &r)
	require.Error(t, unmarshalErr)

	err := RequestJSONValidationError(unmarshalErr)

	var validationErr *RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "", validationErr.Param)
	assert.Equal(t, "invalid_json", validationErr.Code)
}
