// Package converterutil holds helpers shared by the provider converters.
package converterutil

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

// RequestValidationError marks malformed or unsupported client payload content.
// Proxy layers should map it to 4xx without treating it as an AIR/internal failure.
// StatusCode is 0 for the common case ("caller decides", historically always
// mapped to 400) or a specific status (e.g. 413) when the error itself dictates
// which 4xx applies, regardless of which proxy call site catches it.
//
// Cause, when set, is the underlying error that led to this classification (e.g. the raw
// json.UnmarshalTypeError from RequestJSONValidationError) -- log-only detail, such as the
// full nested field path for an error inside a nested struct. Error() deliberately never
// includes it: Error()'s text can reach the client as-is (see proxy writeValidationError,
// which falls back to it whenever Code is unset), and the underlying Go error text isn't
// meant for a client response. Use errors.Unwrap / the Cause field directly for logging.
type RequestValidationError struct {
	Param      string
	Code       string
	Message    string
	StatusCode int
	Cause      error
}

func (e *RequestValidationError) Error() string {
	if e == nil {
		return ""
	}
	if e.Param == "" {
		return e.Message
	}
	if e.Message == "" {
		return e.Param
	}
	return fmt.Sprintf("%s: %s", e.Param, e.Message)
}

// Unwrap exposes Cause to errors.Is/errors.As/errors.Unwrap chains -- e.g. a caller logging
// this error can pull the original json.UnmarshalTypeError (full nested field path, actual
// wrong-type description) back out with errors.Unwrap without it ever reaching the client.
func (e *RequestValidationError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

func NewRequestValidationError(param, message string) error {
	return &RequestValidationError{Param: param, Message: message}
}

// NewRequestEntityTooLargeError marks a payload that exceeds a provider-imposed
// size limit (e.g. an inline base64 image/file). Proxy layers should map it to
// 413 Request Entity Too Large instead of the default 400.
func NewRequestEntityTooLargeError(param, message string) error {
	return &RequestValidationError{Param: param, Message: message, StatusCode: http.StatusRequestEntityTooLarge}
}

// RequestJSONValidationError classifies a json.Unmarshal error against the client's
// OpenAI-format request body into a RequestValidationError, so a malformed field (e.g.
// max_tokens sent as a string) reaches the client as a 4xx naming the offending param --
// the same shape the openai/* passthrough route already gets for free from the real
// OpenAI API -- instead of an opaque 500 from whatever provider-specific converter tried
// to json.Unmarshal the value next. Mirrors the pattern already used for image params in
// vertex/images.go; callers doing their own json.Unmarshal of the client body should wrap
// the error through this instead of a plain fmt.Errorf.
func RequestJSONValidationError(err error) error {
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return &RequestValidationError{Param: typeErr.Field, Message: "Invalid parameter type", Code: "invalid_type", Cause: typeErr}
	}
	return &RequestValidationError{Message: "Invalid JSON", Code: "invalid_json", Cause: err}
}
