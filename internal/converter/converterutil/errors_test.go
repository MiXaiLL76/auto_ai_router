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

// TestRequestJSONValidationError_PreservesCauseForLogging covers the Cause field: a
// caller logging this error (e.g. via errors.Unwrap) must still be able to reach the
// original json.UnmarshalTypeError -- full nested field path, actual Go type names --
// even though Error() and the client-facing message never include it.
func TestRequestJSONValidationError_PreservesCauseForLogging(t *testing.T) {
	type req struct {
		MaxTokens *int `json:"max_tokens,omitempty"`
	}
	var r req
	unmarshalErr := json.Unmarshal([]byte(`{"max_tokens":"five"}`), &r)
	require.Error(t, unmarshalErr)

	err := RequestJSONValidationError(unmarshalErr)

	var validationErr *RequestValidationError
	require.True(t, errors.As(err, &validationErr))

	cause := errors.Unwrap(err)
	require.NotNil(t, cause, "Cause must survive as an unwrappable error for logging")
	var typeErr *json.UnmarshalTypeError
	require.True(t, errors.As(cause, &typeErr))
	assert.Equal(t, "max_tokens", typeErr.Field)

	// The client-facing text must never include the raw Go error -- only the
	// structured Param/Code/Message the wire response is built from.
	assert.NotContains(t, err.Error(), "cannot unmarshal")
}

// TestRequestValidationError_UnwrapNilSafe covers calling Unwrap on a nil
// *RequestValidationError (e.g. after a failed errors.As leaves the pointer unset) --
// logging code that unconditionally calls .Unwrap() on it must not panic.
func TestRequestValidationError_UnwrapNilSafe(t *testing.T) {
	var validationErr *RequestValidationError
	assert.NotPanics(t, func() {
		assert.Nil(t, validationErr.Unwrap())
	})
}

// TestNewRequestValidationError_NoCause covers the plain constructor (no underlying Go
// error involved, e.g. "model is required"): Cause/Unwrap must stay nil, not some
// synthesized value.
func TestNewRequestValidationError_NoCause(t *testing.T) {
	err := NewRequestValidationError("model", "Missing required parameter")
	assert.Nil(t, errors.Unwrap(err))
}
