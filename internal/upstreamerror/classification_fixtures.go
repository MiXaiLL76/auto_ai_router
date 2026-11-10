package upstreamerror

import "net/http"

// ClassificationFixture is one (provider error body, expected client status)
// pair, reproduced from a real production body where possible.
type ClassificationFixture struct {
	Name string
	// Body is the raw JSON of a provider's error value -- what would sit
	// under a response's "error" key, or be the whole terminal stream event.
	Body       string
	WantStatus int
}

// ClassificationFixtures is the single table both the proxy's own
// classification (statusCodeFromProviderBodyError / statusCodeFromProviderStreamError)
// and the LiteLLM compatibility layer's (errorStatus) are tested against.
//
// Before ExtractErrorSignals/ClassifyProviderErrorSignals were shared, the
// two layers re-implemented this independently and drifted more than once:
// first the content-policy marker list (four separate copies, three gained a
// wording and one did not), then the field set a refusal could be phrased in
// (the compatibility layer read fewer fields than the proxy). Both drifts
// were found by hand, after the fact. A shared fixture table means a new
// case added here is checked against both consumers automatically -- if a
// future change to either one stops matching it, the test for THAT package
// fails, not a production incident two weeks later.
var ClassificationFixtures = []ClassificationFixture{
	{
		Name:       "content policy refusal via message, numeric upstream code",
		Body:       `{"code":502,"message":"Output data may contain inappropriate content.","metadata":{"error_type":"provider_unavailable"}}`,
		WantStatus: http.StatusBadRequest,
	},
	{
		Name:       "content policy refusal via metadata error_type only",
		Body:       `{"message":"refused","metadata":{"error_type":"content_policy_violation"}}`,
		WantStatus: http.StatusBadRequest,
	},
	{
		Name:       "content policy refusal via status field",
		Body:       `{"status":"content_filter","message":"refused"}`,
		WantStatus: http.StatusBadRequest,
	},
	{
		Name:       "data_inspection_failed with a message that names no content",
		Body:       `{"code":"data_inspection_failed","type":"data_inspection_failed","message":"Provider returned error"}`,
		WantStatus: http.StatusBadRequest,
	},
	{
		Name:       "provider failure with the same numeric code and error_type stays a 5xx",
		Body:       `{"code":502,"message":"Upstream error from Darkbloom: inference generation failed","metadata":{"error_type":"provider_unavailable"}}`,
		WantStatus: http.StatusBadGateway,
	},
	{
		Name:       "numeric code taken at face value when nothing else classifies",
		Body:       `{"code":400,"message":"Video inputs are not supported by this model."}`,
		WantStatus: http.StatusBadRequest,
	},
	{
		Name:       "a number that is not an http status stays 502",
		Body:       `{"code":42,"message":"unclear"}`,
		WantStatus: http.StatusBadGateway,
	},
	{
		Name:       "rate limit by keyword, no numeric code",
		Body:       `{"message":"exceeded token rate limit","type":"rate_limit_exceeded"}`,
		WantStatus: http.StatusTooManyRequests,
	},
	{
		Name:       "a rate limit outranks a content-policy wording beside it",
		Body:       `{"message":"content_filter","type":"rate_limit_exceeded"}`,
		WantStatus: http.StatusTooManyRequests,
	},
	{
		Name:       "timeout by keyword",
		Body:       `{"message":"upstream request timeout"}`,
		WantStatus: http.StatusRequestTimeout,
	},
	{
		Name:       "overloaded maps to 503",
		Body:       `{"message":"model_overloaded"}`,
		WantStatus: http.StatusServiceUnavailable,
	},
	{
		Name:       "billing block is not a content refusal",
		Body:       `{"message":"Your account has been blocked by the provider for suspected billing fraud."}`,
		WantStatus: http.StatusBadGateway,
	},
	{
		Name:       "geo/IP restriction is not a content refusal",
		Body:       `{"message":"This request was blocked by the provider due to an IP/region restriction."}`,
		WantStatus: http.StatusBadGateway,
	},
}
