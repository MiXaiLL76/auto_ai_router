// Package converterutil holds helpers shared by the provider converters.
package converterutil

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
)

// RequestValidationError marks malformed or unsupported client payload content.
// Proxy layers should map it to 4xx without treating it as an AIR/internal failure.
// StatusCode is 0 for the common case ("caller decides", historically always
// mapped to 400) or a specific status (e.g. 413) when the error itself dictates
// which 4xx applies, regardless of which proxy call site catches it.
//
// Cause (log-only, e.g. RequestJSONValidationError's raw json.UnmarshalTypeError) never
// reaches Error() -- that text can reach the client as-is -- so callers should just log
// this error normally; LogValue below folds Cause in automatically when present.
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

// LogValue folds Cause into this error's own slog output automatically, so a call site
// logging it the normal way ("error", err) gets the cause for free when there is one,
// without every such call site needing its own "cause", err.Unwrap() -- which, copy-pasted
// across several call sites, logged a noisy cause=<nil> for the common case (a plain
// NewRequestValidationError never has one).
func (e *RequestValidationError) LogValue() slog.Value {
	if e == nil {
		return slog.StringValue("<nil>")
	}
	if e.Cause == nil {
		return slog.StringValue(e.Error())
	}
	return slog.GroupValue(
		slog.String("msg", e.Error()),
		slog.Any("cause", e.Cause),
	)
}

func NewRequestValidationError(param, message string) error {
	return &RequestValidationError{Param: param, Message: message}
}

// NewInvalidTypeError marks a field the client sent as the wrong JSON type (present, just
// not the type this route expects for it) -- the "invalid_type" counterpart to
// RequestJSONValidationError's json.UnmarshalTypeError case, for callers that classify a
// field by hand (e.g. from a generic map[string]interface{} type assertion) instead of
// through a typed struct's json.Unmarshal.
func NewInvalidTypeError(param string) error {
	return &RequestValidationError{Param: param, Message: "Invalid parameter type", Code: "invalid_type"}
}

// NewInvalidValueError marks a field of the right type but an unacceptable value (e.g. an
// empty required array, a negative count) -- the "invalid_value" counterpart to
// NewInvalidTypeError above.
func NewInvalidValueError(param string) error {
	return &RequestValidationError{Param: param, Message: "Invalid parameter value", Code: "invalid_value"}
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
		vErr := NewInvalidTypeError(typeErr.Field).(*RequestValidationError)
		vErr.Cause = typeErr
		return vErr
	}
	return &RequestValidationError{Message: "Invalid JSON", Code: "invalid_json", Cause: err}
}
