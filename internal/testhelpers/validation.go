package testhelpers

import (
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// RequireValidationError asserts err is a *converterutil.RequestValidationError with the
// given param and code, and returns it for further assertions (e.g. on Message).
func RequireValidationError(t testing.TB, err error, param, code string) *converterutil.RequestValidationError {
	t.Helper()
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, param, validationErr.Param)
	assert.Equal(t, code, validationErr.Code)
	return validationErr
}
