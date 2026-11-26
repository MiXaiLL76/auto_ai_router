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
		// Any other 4xx is still classified: a provider_messages rule may
		// ask for one of them, and the rule's message is what the client
		// is told. The statuses the router has a message of its own for
		// (403, 404) keep their own branches below.
		if statusCode < http.StatusBadRequest || statusCode >= http.StatusInternalServerError {
			return StatusError{}, false
		}
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

	// A provider message the router has a rule for is answered with the
	// router's own wording, whatever the status the upstream used.
	//
	// The rules may ask for ANY 4xx (the config says so, and the operator
	// wrote them), and the status the rule asks for is the one the client is
	// answered with (see ClassifyProviderErrorInfo). So a rule that says
	// "404" must be honored on a 404, and a rule that says "400" must be
	// honored even when the upstream answered 403 or 404 -- the upstream's
	// status says how it refused the request, and the rule says what the
	// refusal is.
	//
	// Before this, a rule was only consulted for a 400: the status path and
	// the message path were two independent lookups, and the message path
	// only ran for a 400. A rule asking for a 404 therefore set the STATUS
	// correctly and lost its MESSAGE -- the client was answered with a
	// status it had no message for, and got the generic "Request failed".
	info := providerErrorInfoFromBody(rawBody)
	if classified, ok := ClassifyProviderMessage(info); ok {
		return StatusError{Message: classified.Message, Code: classified.Code}, true
	}

	joined := strings.ToLower(strings.Join(info.Signals, " "))

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
