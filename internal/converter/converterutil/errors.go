// Package converterutil holds helpers shared by the provider converters.
package converterutil

import (
	"fmt"
	"net/http"
)

// RequestValidationError marks malformed or unsupported client payload content.
// Proxy layers should map it to 4xx without treating it as an AIR/internal failure.
// StatusCode is 0 for the common case ("caller decides", historically always
// mapped to 400) or a specific status (e.g. 413) when the error itself dictates
// which 4xx applies, regardless of which proxy call site catches it.
type RequestValidationError struct {
	Param      string
	Message    string
	StatusCode int
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

func NewRequestValidationError(param, message string) error {
	return &RequestValidationError{Param: param, Message: message}
}

// NewRequestEntityTooLargeError marks a payload that exceeds a provider-imposed
// size limit (e.g. an inline base64 image/file). Proxy layers should map it to
// 413 Request Entity Too Large instead of the default 400.
func NewRequestEntityTooLargeError(param, message string) error {
	return &RequestValidationError{Param: param, Message: message, StatusCode: http.StatusRequestEntityTooLarge}
}
