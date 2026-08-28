// Package converterutil holds helpers shared by the provider converters.
package converterutil

import "fmt"

// RequestValidationError marks malformed or unsupported client payload content.
// Proxy layers should map it to 4xx without treating it as an AIR/internal failure.
type RequestValidationError struct {
	Param   string
	Message string
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
