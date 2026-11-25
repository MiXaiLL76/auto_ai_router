package upstreamerror

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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

// fixedErrorMessages maps each code this package emits to the exact message
// it emits for it, for the codes whose message carries no provider-derived
// value.
var fixedErrorMessages = map[string]string{ //nolint:gosec // G101: these are the classifier's own client-facing messages, not credentials
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
	accountBlockedCode:           accountBlockedMessage,
	sessionBlockedCode:           sessionBlockedMessage,
	upstreamFetchFailedCode:      upstreamFetchFailedMessage,
}

// fixedImageErrorMessages holds the router's own image-fetch failures.
//
// A client-supplied image URL that the router cannot resolve is a fault of
// the request, not of the provider: these codes are what the router answers
// with, and the classifier has to recognize them so a router body survives a
// re-classification pass (litellm compatibility mode, an AIR-to-AIR proxy
// chain) instead of degrading to the generic message.
//
// Kept in its own map rather than in fixedErrorMessages so the two concerns
// stay separate: "invalid_image" there is the classifier's verdict on a
// provider's own "Invalid image data", and the router's "not an image"
// verdict uses its own code so the two never collide.
//
// The messages themselves come from converterutil, which owns the codes the
// router writes into a response body. They used to be spelled out here as
// well, and the two copies drifted apart: a wording change in one left the
// other behind, and the router's own body then stopped being recognized on
// re-classification -- the exact drift this file exists to prevent.
var fixedImageErrorMessages = buildFixedImageErrorMessages()

func buildFixedImageErrorMessages() map[string]string {
	messages := make(map[string]string)
	for _, code := range converterutil.ImageErrorCodes() {
		if message, ok := converterutil.ImageErrorMessageForCode(code); ok {
			messages[code] = message
		}
	}
	return messages
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

var (
	routerPinnedParameterSuffix  = regexp.MustCompile("^" + pinnedValuePrefix + `\d+(?:\.\d+)?` + pinnedValueSuffix + "$")
	routerOutputTokenRangeSuffix = regexp.MustCompile("^" + outputTokenRangePrefix + `\d+` + rangeMessageSeparator +
		`(?:from \d+(?:` + regexp.QuoteMeta(exclusiveBoundSuffix) + `)? to |up to )\d+(?:` + regexp.QuoteMeta(exclusiveBoundSuffix) + `)?\.$`)
)

// ExtractRouterClassification returns the classification this package's own
// classifier already applied to body, if body IS such a body.
//
// Returns false for anything else -- including an empty body, a body that is
// not JSON, and a provider body that merely resembles ours. The caller then
// classifies it from scratch.
func ExtractRouterClassification(body []byte) (BadRequest, bool) {
	return extractRouterClassification(body, "")
}

func extractRouterClassification(body []byte, expectedType string) (BadRequest, bool) {
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
	if expectedType != "" && typ != expectedType {
		return BadRequest{}, false
	}
	switch code {
	case accountBlockedCode:
		expectedType = "permission_denied"
	case sessionBlockedCode:
		// Unlike every other code here, this one is legitimately produced at
		// two different statuses -- ClassifyStatus's 403 case ("permission_denied")
		// and ClassifyBadRequest's 400 case ("invalid_request_error"), classify.go
		// -- because production carries the same provider signal
		// (session_blocked_by_cyber_policy) at both. Either type is ours;
		// anything else is not.
		if typ != "permission_denied" && typ != "invalid_request_error" {
			return BadRequest{}, false
		}
		expectedType = typ
	case upstreamFetchFailedCode:
		expectedType = "not_found_error"
	default:
		expectedType = "invalid_request_error"
	}
	if typ != expectedType {
		return BadRequest{}, false
	}
	var param *string
	if e.Param != nil {
		cleaned := cleanProviderErrorParam(*e.Param)
		if cleaned == "" || cleaned != *e.Param {
			return BadRequest{}, false
		}
		param = &cleaned
	}
	if !isRouterMessage(code, message, param) {
		return BadRequest{}, false
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
	if fixed, ok := fixedErrorMessages[code]; ok {
		return message == fixed
	}
	if fixed, ok := fixedImageErrorMessages[code]; ok {
		return message == fixed
	}
	if code == "context_length_exceeded" {
		return message == contextLengthGenericMessage || routerContextLengthMessageRe.MatchString(message)
	}
	if param == nil {
		return false
	}
	if code == "unsupported_parameter" {
		_, known := requestParameterNames[*param]
		return known && message == unsupportedParameterPrefix+*param
	}

	suffix, ok := strings.CutPrefix(message, invalidFieldPrefix+*param)
	if !ok {
		return false
	}
	switch code {
	case "invalid_field":
		_, pinnable := pinnableParameters[*param]
		return suffix == "" || pinnable && routerPinnedParameterSuffix.MatchString(suffix)
	case "invalid_max_tokens":
		return isOutputTokenLimitParam(*param) &&
			(suffix == "" || *param == "max_output_tokens" && routerOutputTokenRangeSuffix.MatchString(suffix))
	default:
		return false
	}
}
