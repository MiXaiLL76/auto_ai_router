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
)

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
		case hasSignal(joined, "session_blocked_by_cyber_policy"):
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
