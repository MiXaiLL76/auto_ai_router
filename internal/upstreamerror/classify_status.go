package upstreamerror

import (
	"net/http"
	"strings"
)

const (
	accountBlockedCode         = "account_blocked"
	accountBlockedMessage      = "Inference is blocked on this account"
	sessionBlockedCode         = "session_blocked"
	sessionBlockedMessage      = "This session is blocked by content policy"
	upstreamFetchFailedCode    = "upstream_fetch_failed"
	upstreamFetchFailedMessage = "The upstream could not fetch the content it was asked to fetch"

	// sessionBlockedSignal is the provider's own machine-readable code for a
	// credential/session-level block. Shared between the 403 case below and
	// ClassifyBadRequest's 400 case (classify.go) -- production carries this
	// exact signal at both statuses -- so both report the identical
	// diagnosis rather than one falling through to the generic
	// content-policy bucket because the message also happens to say
	// "...content policy" in plain English.
	sessionBlockedSignal = "session_blocked_by_cyber_policy"
)

// CredentialScopedMarkers are provider signals that read as a content-policy
// refusal in plain English (so they would otherwise match
// ContentPolicyMarkers) but are actually about the CREDENTIAL or SESSION that
// served the request, not the request itself: a different credential answers
// the same request fine. The retry policy (config.DefaultNonRetryableMarkers
// is upstreamerror.ContentPolicyMarkers) must not suppress a retry for these
// just because their wording overlaps with a genuine content-policy marker --
// see internal/proxy/retry.go, checked before the content-policy markers.
var CredentialScopedMarkers = []string{sessionBlockedSignal}

// StatusError is the classification of an upstream error response.
type StatusError = BadRequest

// ClassifyStatus returns the client-facing classification for an upstream error.
// It always classifies HTTP 400 and reports whether it recognized a diagnosis
// for other statuses.
func ClassifyStatus(statusCode int, rawBody []byte) (StatusError, bool) {
	switch statusCode {
	case http.StatusBadRequest:
		return ClassifyBadRequest(rawBody), true
	case http.StatusForbidden, http.StatusNotFound:
	default:
		return StatusError{}, false
	}

	if classified, ok := ExtractRouterClassification(rawBody); ok && classified.Param == nil {
		switch statusCode {
		case http.StatusForbidden:
			if classified.Code == accountBlockedCode || classified.Code == sessionBlockedCode {
				return classified, true
			}
		case http.StatusNotFound:
			if classified.Code == upstreamFetchFailedCode {
				return classified, true
			}
		}
	}

	signals, _ := providerErrorSignalsFromBody(rawBody)
	joined := strings.ToLower(strings.Join(signals, " "))

	switch statusCode {
	case http.StatusForbidden:
		switch {
		case hasSignal(joined, "inference is blocked on this account"):
			return StatusError{Message: accountBlockedMessage, Code: accountBlockedCode}, true
		case hasSignal(joined, sessionBlockedSignal):
			return StatusError{Message: sessionBlockedMessage, Code: sessionBlockedCode}, true
		}
	case http.StatusNotFound:
		// The failing URL can come from the provider's own configuration.
		if hasSignal(joined, "cannot fetch content from the provided url") {
			return StatusError{Message: upstreamFetchFailedMessage, Code: upstreamFetchFailedCode}, true
		}
	}
	return StatusError{}, false
}
