package upstreamerror

import (
	"net/http"
	"strings"
)

// This file classifies the 4xx statuses that are NOT 400.
//
// 400 has its own classifier (ClassifyBadRequest) because it is by far the
// most common failure and carries the most information -- the provider
// names the parameter it rejected. The other 4xx statuses were reported to
// the client as a single generic message ("Request failed") with a generic
// code ("api_error"), whatever the provider actually said.
//
// That cost real diagnostics. Measured over production raw_bodies: 17,903
// requests in a fortnight came back as "Inference is blocked on this
// account" -- an account-level block from the provider -- and every one of
// them reached the client as "Request failed", with nothing to act on.

// StatusError is the classification of a non-400 4xx response.
type StatusError struct {
	Message string
	Code    string
	Param   *string
}

// ClassifyStatus classifies a non-400 4xx response body.
//
// Like ClassifyBadRequest it only ever returns a fixed message plus a
// narrowly-scoped parameter name -- never the provider's own free text.
//
// Returns false when the body carries nothing this package recognizes, in
// which case the caller keeps its generic message.
func ClassifyStatus(statusCode int, rawBody []byte) (StatusError, bool) {
	// A body this package's classifier already produced is returned as it
	// is, the same as in ClassifyBadRequest. LiteLLM compatibility mode
	// re-classifies the body the router already wrote, and without this the
	// second pass would see the router's own generic message ("Request
	// failed") instead of the provider's words, and lose the classification
	// the native mode had already made.
	if classified, ok := ExtractRouterClassification(rawBody); ok {
		return StatusError(classified), true
	}

	signals, providerParam := providerErrorSignalsFromBody(rawBody)
	if len(signals) == 0 {
		return StatusError{}, false
	}
	joined := strings.ToLower(strings.Join(signals, " "))

	switch statusCode {
	case http.StatusForbidden:
		// A provider that blocks inference for the whole ACCOUNT (not for
		// one request) says so in its own words: "Inference is blocked on
		// this account. Please contact support@openrouter.ai." The client's
		// next action is to fix the account -- a different credential, a
		// support ticket -- and "Request failed" gives it nothing.
		//
		// The marker is exact: in production it appears in 17,903 responses
		// and in NO other status, so it cannot shadow a different failure.
		if hasSignal(joined, "inference is blocked on this account", "inference is blocked on this account.") {
			return StatusError{
				Message: "Inference is blocked on this account",
				Code:    "account_blocked",
			}, true
		}
		// A provider that blocks a SESSION on policy grounds: "This session
		// is blocked by content policy". The content-policy markers do not
		// cover it ("content policy" alone would also match the request-side
		// marker), and the provider names it with its own code
		// ("session_blocked_by_cyber_policy"), so the code is what this
		// branch keys on.
		if hasSignal(joined, "session_blocked_by_cyber_policy") {
			return StatusError{
				Message: "This session is blocked by content policy",
				Code:    "session_blocked",
			}, true
		}
	case http.StatusNotFound:
		// A provider that could not fetch a URL it was asked to fetch.
		//
		// The URL is NOT reported as a rejected parameter, because it is
		// not one: in production none of these requests carry a URL at
		// all -- the body has only model, messages and (in 2 cases) tools.
		// The URL comes from the provider's OWN grounding configuration, so
		// the client has nothing to correct and no field to change.
		// Saying "Invalid url" with param "url" would point the client at
		// a field its own request never contained.
		//
		// What the client CAN do is try the request again on another
		// credential or later, so the message says that rather than
		// blaming the request.
		if hasSignal(joined, "cannot fetch content from the provided url") {
			return StatusError{
				Message: "The upstream could not fetch the content it was asked to fetch",
				Code:    "upstream_fetch_failed",
			}, true
		}
	}
	_ = providerParam
	return StatusError{}, false
}
