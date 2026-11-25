package proxy

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestImageErrorReachesClientAsClientError is the regression test for the
// production incident: a client whose image URL could not be fetched got a
// generic 500 "Failed to convert request", which looked like a router bug
// and was retried by clients that retry 5xx. It must be a classified 4xx
// with a fixed message and no URL in it.
func TestImageErrorReachesClientAsClientError(t *testing.T) {
	imgErr := converterutil.NewImageError(converterutil.ImageErrorFetchFailed,
		"messages.content.image_url.url")

	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(imgErr, &validationErr),
		"the proxy classifies converter errors by looking for *RequestValidationError")

	status := statusForValidationError(validationErr)
	assert.Equal(t, http.StatusBadRequest, status)

	w := httptest.NewRecorder()
	writeValidationError(w, validationErr, imgErr.Error())

	assert.Equal(t, http.StatusBadRequest, w.Code)

	var resp APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "Failed to fetch image URL", resp.Error.Message)
	assert.Equal(t, "invalid_request_error", resp.Error.Type)
	require.NotNil(t, resp.Error.Code)
	assert.Equal(t, "image_fetch_failed", *resp.Error.Code)
	require.NotNil(t, resp.Error.Param)
	assert.Equal(t, "messages.content.image_url.url", *resp.Error.Param)
}

// TestImageErrorTooLargeReachesClientAs413 covers the size-cap case, which
// must not be reported as a generic 400 either: a client sending an
// oversized image needs to know it is a size problem.
func TestImageErrorTooLargeReachesClientAs413(t *testing.T) {
	imgErr := converterutil.NewImageError(converterutil.ImageErrorTooLarge,
		"messages.content.image_url.url")

	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(imgErr, &validationErr))
	assert.Equal(t, http.StatusRequestEntityTooLarge, statusForValidationError(validationErr))

	w := httptest.NewRecorder()
	writeValidationError(w, validationErr, imgErr.Error())

	assert.Equal(t, http.StatusRequestEntityTooLarge, w.Code)

	var resp APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.NotNil(t, resp.Error.Code)
	assert.Equal(t, "image_too_large", *resp.Error.Code)
	assert.Equal(t, "invalid_request_error", resp.Error.Type)
}

// TestImageErrorBodyLeaksNoURL is the leak guard at the HTTP boundary: the
// bytes written to the client must not contain the image URL, its host or
// any upstream status. Image URLs are frequently presigned links carrying a
// token, so echoing one back would disclose a credential.
func TestImageErrorBodyLeaksNoURL(t *testing.T) {
	secretURL := "https://cdn.example.com/photo.jpg?X-Amz-Signature=SECRETSIGNATURE"

	imgErr := converterutil.NewImageError(converterutil.ImageErrorFetchFailed,
		"messages.content.image_url.url")
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(imgErr, &validationErr))

	w := httptest.NewRecorder()
	writeValidationError(w, validationErr, imgErr.Error())

	body := w.Body.String()
	assert.NotContains(t, body, "SECRETSIGNATURE")
	assert.NotContains(t, body, "cdn.example.com")
	assert.NotContains(t, body, secretURL)
	assert.NotContains(t, body, "http://")
	assert.NotContains(t, body, "https://")
}

// TestImageErrorIsNotTreatedAsInternalError checks the image error does not
// fall into the "fatal conversion error" branch, which would answer 500 and
// report the failure against the credential's health.
func TestImageErrorIsNotTreatedAsInternalError(t *testing.T) {
	imgErr := converterutil.NewImageError(converterutil.ImageErrorURLNotAllowed,
		"messages.content.image_url.url")

	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(imgErr, &validationErr),
		"image errors must be recognized as client errors, not internal failures")

	status := statusForValidationError(validationErr)
	assert.NotEqual(t, http.StatusInternalServerError, status)
	assert.GreaterOrEqual(t, status, 400)
	assert.Less(t, status, 500)
}

// TestImageErrorMessageMatchesUpstreamClassificationStyle keeps the image
// error wording consistent with how the router already reports bad
// requests: a short fixed phrase, no provider detail.
func TestImageErrorMessageMatchesUpstreamClassificationStyle(t *testing.T) {
	for _, reason := range []converterutil.ImageErrorReason{
		converterutil.ImageErrorFetchFailed,
		converterutil.ImageErrorInvalidData,
		converterutil.ImageErrorUnsupportedMediaType,
		converterutil.ImageErrorURLNotAllowed,
		converterutil.ImageErrorTooLarge,
	} {
		msg := converterutil.ImageErrorMessage(reason)
		assert.NotEmpty(t, msg)
		assert.False(t, strings.HasSuffix(msg, "."), "message %q should be a label, not a sentence", msg)
		assert.Less(t, len(msg), 80, "message %q is too long for a fixed error label", msg)
	}
}
