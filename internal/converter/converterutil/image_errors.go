package converterutil

import (
	"errors"
	"fmt"
	"net/http"
	"sort"
)

// ImageErrorReason enumerates the ways an image attachment can fail to
// reach the provider. It is deliberately a closed set: every value maps to
// one fixed client-facing message and one fixed error code, so nothing about
// the failed fetch (URL, host, upstream status, network error text) can leak
// into a response body by accident.
type ImageErrorReason int

const (
	// ImageErrorFetchFailed covers every transport-level failure: DNS, TLS,
	// connection refused, timeout, truncated body. The remote host may be
	// reachable but broken, slow, or gone — from the client's point of view
	// the request carries an attachment that cannot be resolved.
	//
	// TODO: this reason conflates a permanent fault of the request with a
	// transient fault of the network. A URL that no longer resolves and a
	// one-off DNS glitch both land here, and because the error unwraps to
	// *RequestValidationError the request fails permanently on the first
	// attempt — a transient blip on a healthy image host never gets the
	// retry it deserves.
	//
	// The fix is to separate the reasons (a permanent "the URL does not
	// resolve" from a transient "the fetch failed") and route the transient
	// one through the retryable-fault mechanism the proxy already has for
	// provider-side failures (RetryReasonNetErr / shouldRetry). It is left
	// as one reason for now because splitting it changes the client-facing
	// contract, and the overwhelming majority of production failures here
	// are the permanent kind — a URL that never worked.
	ImageErrorFetchFailed ImageErrorReason = iota
	// ImageErrorInvalidData covers a response that arrived but is not a
	// usable image: an HTML error page, a non-image content type, or bytes
	// that decode to no known image format.
	ImageErrorInvalidData
	// ImageErrorUnsupportedMediaType covers a well-formed image in a format
	// the target provider does not accept (e.g. SVG or BMP for Anthropic).
	ImageErrorUnsupportedMediaType
	// ImageErrorURLNotAllowed covers URLs the router refuses to fetch:
	// non-HTTP schemes, and hosts that resolve to private or link-local
	// address space (SSRF guard).
	ImageErrorURLNotAllowed
	// ImageErrorTooLarge covers an image whose payload exceeds the router's
	// download limit.
	ImageErrorTooLarge
	// ImageErrorTooManyImages covers a request carrying more image
	// attachments than the router is willing to download for one request.
	ImageErrorTooManyImages
)

// imageErrorDetails maps each reason to its fixed client-facing contract.
// Message and Code never contain a URL, a host, an upstream HTTP status, or
// any provider-specific text: the client only learns what to fix in its own
// request, and where to fix it (Param).
var imageErrorDetails = map[ImageErrorReason]struct {
	message string
	code    string
	status  int
}{
	ImageErrorFetchFailed: {
		message: "Failed to fetch image URL",
		code:    "image_fetch_failed",
		status:  http.StatusBadRequest,
	},
	ImageErrorInvalidData: {
		message: "Image URL did not return a valid image",
		// "invalid_image_data", not "invalid_image": the upstream classifier
		// already owns "invalid_image" for a provider's own "Invalid image
		// data" verdict. Reusing the code would make the two indistinguishable
		// when a router body is re-classified, and the router's message would
		// be rewritten to the provider's wording.
		code:   "invalid_image_data",
		status: http.StatusBadRequest,
	},
	ImageErrorUnsupportedMediaType: {
		message: "Unsupported image media type",
		code:    "unsupported_image_media_type",
		status:  http.StatusBadRequest,
	},
	ImageErrorURLNotAllowed: {
		message: "Image URL is not allowed",
		code:    "image_url_not_allowed",
		status:  http.StatusBadRequest,
	},
	ImageErrorTooLarge: {
		message: "Image is too large",
		code:    "image_too_large",
		status:  http.StatusRequestEntityTooLarge,
	},
	ImageErrorTooManyImages: {
		message: "Too many images in the request",
		code:    "too_many_images",
		status:  http.StatusBadRequest,
	},
}

// ImageError marks a client-supplied image attachment that could not be
// turned into something the provider accepts.
//
// It is a RequestValidationError rather than a plain error on purpose: the
// proxy maps RequestValidationError to a 4xx and echoes its Message/Code to
// the client, while a plain error falls through to the "fatal conversion
// error" branch and becomes a generic 500. Before this type existed, a
// client whose image URL was unreachable produced a 500 "Failed to convert
// request" — indistinguishable from a router bug, and retried by clients
// that retry 5xx.
//
// The Error() string is the fixed message, optionally prefixed with Param.
// It deliberately carries no detail about the failed fetch: the full URL,
// the resolved host and the upstream status belong in the router's internal
// log, not in a response body that may be surfaced to end users.
//
// TODO: the "this is the request's fault, not the credential's" decision is
// made here by the error's Go type (the proxy asks for a
// *RequestValidationError), which is a mechanism of its own. The codebase
// already has another one for the same decision: the marker-based
// CredentialScopedMarkers in internal/upstreamerror, consumed by
// internal/proxy/retry.go.
//
// Two mechanisms for one decision means an engineer auditing "why was this
// credential not penalised" or "why did this not retry" has to know about
// both. Folding the image errors into the marker convention -- or making the
// marker convention understand request-scoped faults -- would be the
// consistent shape. It is left as it is for now: the marker convention works
// on response bodies from a provider, while an image error is produced
// before the provider is contacted at all, and the two do not have a common
// representation yet.
type ImageError struct {
	RequestValidationError
	Reason ImageErrorReason
}

// Unwrap exposes the embedded RequestValidationError so that errors.As finds
// it: the proxy classifies every converter-reported client error by asking
// for a *RequestValidationError, and an ImageError must land in that branch
// (4xx with a client-facing message) rather than the generic fatal-conversion
// branch (500).
func (e *ImageError) Unwrap() error {
	if e == nil {
		return nil
	}
	return &e.RequestValidationError
}

// NewImageError builds an ImageError for the given reason. The param names
// the request field to fix (e.g. "messages.content.image_url.url"); it is
// echoed to the client as the "param" field so the client knows where the
// problem is without being told anything about the target host.
func NewImageError(reason ImageErrorReason, param string) error {
	details, ok := imageErrorDetails[reason]
	if !ok {
		// Unknown reason: fall back to the generic fetch failure rather than
		// leaking an empty message or a zero status.
		details = imageErrorDetails[ImageErrorFetchFailed]
	}
	return &ImageError{
		RequestValidationError: RequestValidationError{
			Param:      param,
			Code:       details.code,
			Message:    details.message,
			StatusCode: details.status,
		},
		Reason: reason,
	}
}

// ImageErrorMessage returns the fixed client-facing message for a reason.
// Exposed so callers that need the same wording (e.g. for logCtx.ErrorMsg)
// do not re-derive it from the error's own Error() string.
func ImageErrorMessage(reason ImageErrorReason) string {
	details, ok := imageErrorDetails[reason]
	if !ok {
		details = imageErrorDetails[ImageErrorFetchFailed]
	}
	return details.message
}

// ImageErrorCode returns the fixed client-facing error code for a reason.
func ImageErrorCode(reason ImageErrorReason) string {
	details, ok := imageErrorDetails[reason]
	if !ok {
		details = imageErrorDetails[ImageErrorFetchFailed]
	}
	return details.code
}

// ImageErrorCodes lists every image error code the router can emit.
//
// The upstream classifier needs the same code-to-message mapping to recognize
// a router-written body on a re-classification pass (LiteLLM compatibility
// mode, an AIR-to-AIR chain). Exposing the codes from here keeps that
// mapping in one place: the two used to be separate literals, and a wording
// change in one but not the other silently degraded the router's own body to
// the generic message.
func ImageErrorCodes() []string {
	codes := make([]string, 0, len(imageErrorDetails))
	for reason := range imageErrorDetails {
		codes = append(codes, imageErrorDetails[reason].code)
	}
	sort.Strings(codes)
	return codes
}

// ImageErrorMessageForCode returns the fixed message for an image error code,
// and whether the code is one this package owns.
func ImageErrorMessageForCode(code string) (string, bool) {
	for _, details := range imageErrorDetails {
		if details.code == code {
			return details.message, true
		}
	}
	return "", false
}

// DescribeImageError renders an image error for the router's internal log.
// Unlike the client-facing message this may name the failed URL and the
// underlying cause — internal logs are where that diagnostic detail belongs.
// Keep the result out of anything the client can read.
func DescribeImageError(err error, imgURL string, cause error) string {
	var imageErr *ImageError
	if !errors.As(err, &imageErr) {
		return fmt.Sprintf("image attachment failed: %v", err)
	}
	msg := fmt.Sprintf("image attachment failed (%s): %s", ImageErrorCode(imageErr.Reason), ImageErrorMessage(imageErr.Reason))
	if imgURL != "" {
		msg += fmt.Sprintf(" url=%q", imgURL)
	}
	if cause != nil {
		msg += fmt.Sprintf(" cause=%v", cause)
	}
	return msg
}
