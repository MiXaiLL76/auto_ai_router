package converterutil

import (
	"errors"
	"net/http"

	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestImageErrorContract pins the client-facing contract of every image
// failure kind: a fixed message, a fixed code and a status that is always a
// 4xx. The message and code are what a client sees, so they must never
// carry a URL, a host or any upstream detail.
func TestImageErrorContract(t *testing.T) {
	for _, tt := range []struct {
		reason  ImageErrorReason
		message string
		code    string
		status  int
	}{
		{ImageErrorFetchFailed, "Failed to fetch image URL", "image_fetch_failed", http.StatusBadRequest},
		{ImageErrorInvalidData, "Image URL did not return a valid image", "invalid_image_data", http.StatusBadRequest},
		{ImageErrorUnsupportedMediaType, "Unsupported image media type", "unsupported_image_media_type", http.StatusBadRequest},
		{ImageErrorURLNotAllowed, "Image URL is not allowed", "image_url_not_allowed", http.StatusBadRequest},
		{ImageErrorTooLarge, "Image is too large", "image_too_large", http.StatusRequestEntityTooLarge},
	} {
		t.Run(tt.code, func(t *testing.T) {
			err := NewImageError(tt.reason, "messages.content.image_url.url")
			require.Error(t, err)

			var imageErr *ImageError
			require.True(t, errors.As(err, &imageErr), "must be an *ImageError")
			assert.Equal(t, tt.reason, imageErr.Reason)
			assert.Equal(t, tt.message, imageErr.Message)
			assert.Equal(t, tt.code, imageErr.Code)
			assert.Equal(t, tt.status, imageErr.StatusCode)

			// The proxy classifies converter errors by looking for a
			// *RequestValidationError. Without Unwrap the image error would
			// fall into the "fatal conversion error" branch and become a
			// generic 500 instead of a 4xx.
			var validationErr *RequestValidationError
			require.True(t, errors.As(err, &validationErr),
				"an ImageError must unwrap to *RequestValidationError")
			assert.Equal(t, tt.message, validationErr.Message)
			assert.Equal(t, tt.code, validationErr.Code)
			assert.Equal(t, tt.status, validationErr.StatusCode)
			assert.Equal(t, "messages.content.image_url.url", validationErr.Param)
		})
	}
}

// TestImageErrorNeverLeaksDetails is the regression guard for the leak that
// motivated this type: a failing image URL used to be interpolated straight
// into an error message, so a presigned URL (token included) could reach the
// client. The message, the code and the param must stay free of any URL.
func TestImageErrorNeverLeaksDetails(t *testing.T) {
	secretURL := "https://cdn.example.com/photo.jpg?X-Amz-Signature=SECRETSIGNATURE&X-Amz-Credential=AKIA"

	for _, reason := range []ImageErrorReason{
		ImageErrorFetchFailed,
		ImageErrorInvalidData,
		ImageErrorUnsupportedMediaType,
		ImageErrorURLNotAllowed,
		ImageErrorTooLarge,
	} {
		err := NewImageError(reason, "messages.content.image_url.url")

		var imageErr *ImageError
		require.True(t, errors.As(err, &imageErr))

		for name, value := range map[string]string{
			"message": imageErr.Message,
			"code":    imageErr.Code,
			"param":   imageErr.Param,
			"error()": imageErr.Error(),
		} {
			assert.NotContains(t, value, "SECRETSIGNATURE", "%s must not contain the signed URL", name)
			assert.NotContains(t, value, "cdn.example.com", "%s must not contain the host", name)
			assert.NotContains(t, value, "http://", "%s must not contain a URL scheme", name)
			assert.NotContains(t, value, "https://", "%s must not contain a URL scheme", name)
			// The URL must not be echoed back anywhere in the error either.
			assert.NotContains(t, value, secretURL, "%s must not contain the URL", name)
		}
	}
}

// TestDescribeImageErrorKeepsDetailInternal checks the counterpart: the
// router's own log line may name the URL and the cause — that is where the
// diagnostic detail belongs — while the client-facing message stays clean.
func TestDescribeImageErrorKeepsDetailInternal(t *testing.T) {
	imgURL := "https://cdn.example.com/photo.jpg?X-Amz-Signature=SECRETSIGNATURE"
	cause := errors.New("dial tcp 10.0.0.5:443: connect: connection refused")

	described := DescribeImageError(NewImageError(ImageErrorFetchFailed, "p"), imgURL, cause)
	assert.Contains(t, described, "image_fetch_failed")
	assert.Contains(t, described, imgURL)
	assert.Contains(t, described, "connection refused")
}

// TestImageErrorUnknownReasonFallsBack makes sure an unmapped reason still
// produces a usable client error rather than an empty message or a zero
// status (which would default to 200 in some paths).
func TestImageErrorUnknownReasonFallsBack(t *testing.T) {
	err := NewImageError(ImageErrorReason(999), "messages.content.image_url.url")

	var validationErr *RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "Failed to fetch image URL", validationErr.Message)
	assert.Equal(t, "image_fetch_failed", validationErr.Code)
	assert.Equal(t, http.StatusBadRequest, validationErr.StatusCode)
}

// TestImageErrorCodeAndMessageHelpers keeps the exported accessors in sync
// with the error itself, since callers use them to build log messages.
func TestImageErrorCodeAndMessageHelpers(t *testing.T) {
	assert.Equal(t, "image_fetch_failed", ImageErrorCode(ImageErrorFetchFailed))
	assert.Equal(t, "Failed to fetch image URL", ImageErrorMessage(ImageErrorFetchFailed))
	assert.Equal(t, "image_too_large", ImageErrorCode(ImageErrorTooLarge))
	assert.Equal(t, "Image is too large", ImageErrorMessage(ImageErrorTooLarge))
	assert.Equal(t, "image_fetch_failed", ImageErrorCode(ImageErrorReason(999)))
	assert.Equal(t, "Failed to fetch image URL", ImageErrorMessage(ImageErrorReason(999)))
}

// TestImageErrorStatusIsAlwaysClientError guards the reason this type
// exists: image failures used to be reported as 500 "Failed to convert
// request", which clients retry as if the router were broken.
func TestImageErrorStatusIsAlwaysClientError(t *testing.T) {
	for reason := ImageErrorFetchFailed; reason <= ImageErrorTooLarge; reason++ {
		err := NewImageError(reason, "p")
		var validationErr *RequestValidationError
		require.True(t, errors.As(err, &validationErr))
		assert.GreaterOrEqual(t, validationErr.StatusCode, 400)
		assert.Less(t, validationErr.StatusCode, 500)
		assert.Contains(t, validationErr.Code, "image",
			"code %q should identify the image failure", validationErr.Code)
	}
}
