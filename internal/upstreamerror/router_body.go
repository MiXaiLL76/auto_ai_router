package upstreamerror

import (
	"encoding/json"
	"regexp"
)

// This file recognizes a body that this package's own classifier has already
// produced.
//
// Why it exists. LiteLLM compatibility mode re-runs the classifier over the
// body the router ALREADY wrote to the client (see normalizeError), and an
// AIR-to-AIR proxy chain hands one router the classification another router
// already made. Both used to depend on every branch of the classifier
// recognizing its OWN canonical output -- which is why the classifier grew a
// second regex for one branch's message, and why a branch whose message did
// not round-trip (a range with no lower bound) silently degraded to the
// generic one on the second pass.
//
// Recognizing the shape instead makes the classifier idempotent BY
// CONSTRUCTION: a body that is already classified is returned as it is, and
// no branch has to know its own canonical text.
//
// The recognition is deliberately strict. A body that is NOT ours must still
// be classified from scratch -- and a mis-recognized body would put the
// provider's own message text on the wire, which is exactly what this
// package exists to prevent. So every part of the body is checked: the type
// has to be one this package emits, the code has to be one this package
// emits, the message has to match the EXACT shape this package renders for
// that code, and the parameter has to be one this package could have
// produced. Where a message carries a provider-derived value (a token count,
// a range) only DIGITS may vary -- nothing else.

// routerClassifiedMessages maps each code this package emits to the exact message
// it emits for it, for the codes whose message carries no provider-derived
// value.
var routerClassifiedMessages = map[string]string{ //nolint:gosec // G101: these are the classifier's own client-facing messages, not credentials
	"invalid_request":            "Invalid request",
	"invalid_model":              "Invalid model",
	"invalid_tool_choice":        "Invalid tool_choice",
	"max_tokens_exceeded":        "Max tokens exceeded",
	"invalid_parameter":          "Invalid request parameter",
	"content_policy_violation":   "The prompt was blocked by the provider's content policy",
	"invalid_image_size":         "Invalid image size",
	"invalid_image":              "Invalid image data",
	"missing_required_parameter": "Missing required parameter",
	"invalid_type":               "Invalid parameter type",
	"invalid_json":               "Invalid JSON",
	"invalid_multipart":          "Invalid multipart form data",
	"invalid_value":              "Invalid parameter value",
	"reasoning_required":         "Reasoning is required and cannot be disabled",
	"account_blocked":            "Inference is blocked on this account",
	"session_blocked":            "This session is blocked by content policy",
	"upstream_fetch_failed":      "The upstream could not fetch the content it was asked to fetch",
}

// routerErrorTypes is the set of "type" values this package's classifier
// emits (see errorTypeForStatus in the proxy package and the LiteLLM
// compatibility layer). A body whose type is not one of these was not
// produced by this package.
var routerErrorTypes = map[string]struct{}{
	"invalid_request_error": {},
	"authentication_error":  {},
	"insufficient_quota":    {},
	"permission_denied":     {},
	"not_found_error":       {},
	"timeout_error":         {},
	"rate_limit_error":      {},
	"api_error":             {},
	"server_error":          {},
}

// routerContextLengthMessageRe matches the message contextLengthMessage
// renders, with DIGITS as the only variable part. The three shapes are the
// three the function can produce: limit only, limit plus input tokens, and
// limit plus both.
//
// Nothing but digits may appear in the variable positions, so a provider
// cannot hide free text behind a number ("... is 409600 tokens of secret").
var routerContextLengthMessageRe = regexp.MustCompile(
	`^This model's maximum context length is \d+ tokens\.` +
		`(?: However, you requested \d+ output tokens and your prompt contains at least \d+ input tokens\.` +
		`| However, your prompt contains at least \d+ input tokens\.)?$`)

// routerUnsupportedParameterMessageRe matches the "Unsupported parameter:
// <param>" message, where the parameter name is fixed by the body's own param
// field.
func routerUnsupportedParameterMessageRe(param string) *regexp.Regexp {
	return regexp.MustCompile(`^Unsupported parameter: ` + regexp.QuoteMeta(param) + `$`)
}

// routerInvalidFieldMessageRe matches the "Invalid <param>" message, where
// the parameter name is fixed by the body's own param field.
//
// Two shapes: the bare one, and the one that carries the value the provider
// pins the parameter to ("Invalid temperature: only 1 is allowed for this
// model"). In the second, only the NUMBER may differ from what the router
// wrote -- anything else appended to it means the body is not ours and must
// be classified from scratch.
func routerInvalidFieldMessageRe(param string) *regexp.Regexp {
	return regexp.MustCompile(`^Invalid ` + regexp.QuoteMeta(param) +
		`(?:: only \d+(?:\.\d+)? is allowed for this model)?$`)
}

// routerInvalidMaxTokensMessageRe matches the "Invalid <param>" message the
// max-tokens branch renders, plus the variant that carries the provider's
// range -- where only DIGITS may vary.
//
// The range variant is "Invalid <param>: requested N, but the supported
// range is from L to U" (or "up to U" when the provider stated no lower
// bound); both renderings are accepted, and in both only the numbers may
// differ from what the router wrote.
func routerInvalidMaxTokensMessageRe(param string) *regexp.Regexp {
	return regexp.MustCompile(`^Invalid ` + regexp.QuoteMeta(param) +
		`(?:: requested \d+, but the supported range is (?:from \d+ to |up to )\d+\.)?$`)
}

// ExtractRouterClassification returns the classification this package's own
// classifier already applied to body, if body IS such a body.
//
// Returns false for anything else -- including an empty body, a body that is
// not JSON, and a provider body that merely resembles ours. The caller then
// classifies it from scratch.
func ExtractRouterClassification(body []byte) (BadRequest, bool) {
	if len(body) == 0 {
		return BadRequest{}, false
	}
	var envelope struct {
		Error *struct {
			Message *string `json:"message"`
			Type    *string `json:"type"`
			Param   *string `json:"param"`
			Code    *string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Error == nil {
		return BadRequest{}, false
	}
	e := envelope.Error
	// All of message, type and code are always present in a body this
	// package writes (APIError's fields carry no omitempty); param may be
	// null, but the key itself is always there.
	if e.Message == nil || e.Type == nil || e.Code == nil {
		return BadRequest{}, false
	}
	message, code, typ := *e.Message, *e.Code, *e.Type
	if message == "" || code == "" {
		return BadRequest{}, false
	}
	if _, ok := routerErrorTypes[typ]; !ok {
		return BadRequest{}, false
	}
	if !isRouterMessage(code, message, e.Param) {
		return BadRequest{}, false
	}
	// The param goes through the same cleaning as one read off a provider
	// body (cleanProviderErrorParam), so a body that reached this path
	// cannot put an arbitrary string into the client's "param" field just
	// by carrying one.
	var param *string
	if e.Param != nil {
		if cleaned := cleanProviderErrorParam(*e.Param); cleaned != "" {
			param = &cleaned
		}
	}
	return BadRequest{Message: message, Code: code, Param: param}, true
}

// isRouterMessage reports whether message is the message this package emits
// for code.
//
// For the fixed codes it is an exact match. For the codes whose message
// carries a provider-derived value it is a shape match, and the shape admits
// DIGITS only -- so a provider body that borrows one of our codes and
// appends free text to the message is rejected and classified from scratch
// like any other provider body.
func isRouterMessage(code, message string, param *string) bool {
	if fixed, ok := routerClassifiedMessages[code]; ok {
		return message == fixed
	}
	switch code {
	case "unsupported_parameter":
		// "Unsupported parameter: <param>" -- the parameter name is the
		// body's own param field, so only that exact rendering is ours.
		if param == nil || *param == "" {
			return false
		}
		return routerUnsupportedParameterMessageRe(*param).MatchString(message)
	case "context_length_exceeded":
		return routerContextLengthMessageRe.MatchString(message)
	case "invalid_field":
		if param == nil || *param == "" {
			return false
		}
		return routerInvalidFieldMessageRe(*param).MatchString(message)
	case "invalid_max_tokens":
		if param == nil || *param == "" {
			return false
		}
		return routerInvalidMaxTokensMessageRe(*param).MatchString(message)
	}
	return false
}
