package testhelpers

import (
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// RequireValidationError asserts err is a *converterutil.RequestValidationError with the
// given param and code. Tests that need the error object itself (e.g. to check
// validationErr.Message) should errors.As it themselves, as the *_MissingReportsMissing
// tests do.
func RequireValidationError(t testing.TB, err error, param, code string) {
	t.Helper()
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, param, validationErr.Param)
	assert.Equal(t, code, validationErr.Code)
}
