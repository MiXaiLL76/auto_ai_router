package upstreamerror

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

// azureContentFilterBodies are the shapes Azure uses to refuse a request on
// safety grounds: a 400 whose message names no content at all -- "The model
// was unable to complete inference due to an internal error." -- and whose
// own code is "invalid_prompt" (or "content_filter", or
// "ResponsibleAIPolicyViolation"). Reproduced from production request
// cf33c6845d7f7f3e051a4c1e95a9e889 (2026-11-10, model gpt-5-mini, four
// Azure credentials).
//
// The refusal is invisible to a classifier that looks for content-policy
// wording: there is none in the message, and "unable to complete inference"
// reads like a provider-side failure. Without the Azure markers the refusal
// was classified as a generic error, stayed retryable across the whole
// credential pool -- every credential refused it the same way -- and reached
// the client as a 502 after 7 attempts and 95 seconds.
var azureContentFilterBodies = []string{
	`{"message":"The model was unable to complete inference due to an internal error."}`,
	`{"code":"invalid_prompt","message":"The model was unable to complete inference due to an internal error."}`,
	`{"code":"invalid_prompt","message":"The model was unable to complete inference due to an internal error.","type":"invalid_request_error"}`,
	`{"code":"content_filter","message":"The model was unable to complete inference due to an internal error."}`,
	`{"code":"ResponsibleAIPolicyViolation","message":"The model was unable to complete inference due to an internal error."}`,
}

// TestAzureContentFilterIsARefusal pins that every shape of Azure's refusal
// is recognized as one.
func TestAzureContentFilterIsARefusal(t *testing.T) {
	for _, body := range azureContentFilterBodies {
		// The MESSAGE is what identifies the refusal: it carries Azure's
		// wording, and it is present in every shape the provider uses. The
		// codes are checked separately below -- "invalid_prompt" is NOT a
		// refusal marker on its own, because Azure uses it for ordinary
		// prompt problems too.
		assert.True(t, IsContentPolicySignal(fieldValue(body, "message")),
			"azure's content filter must be recognized as a refusal: %s", body)
	}
	// Azure's own content-filter codes are refusals on their own.
	assert.True(t, IsContentPolicySignal("content_filter"))
	assert.True(t, IsContentPolicySignal("ResponsibleAIPolicyViolation"))
	// ...but "invalid_prompt" is not: Azure uses it for ordinary prompt
	// problems too, so it must not turn every bad prompt into a refusal.
	assert.False(t, IsContentPolicySignal("invalid_prompt"),
		"invalid_prompt alone must not be treated as a content refusal")
}

// TestAzureContentFilterIsNotAGenericProviderFailure is the counterpart: a
// provider failure that merely mentions inference must NOT become a refusal.
// The marker list is deliberately specific -- "unable to complete inference"
// is Azure's wording, but a provider that says "inference generation failed"
// has not refused anything on safety grounds, and treating it as a refusal
// would stop a retry that would have succeeded.
func TestAzureContentFilterIsNotAGenericProviderFailure(t *testing.T) {
	for _, body := range []string{
		`{"code":"provider_unavailable","message":"inference generation failed"}`,
		`{"code":502,"message":"Upstream error from Darkbloom: inference generation failed","metadata":{"error_type":"provider_unavailable"}}`,
		`{"message":"Your account has been blocked by the provider for suspected billing fraud."}`,
		`{"message":"This request was blocked by the provider due to an IP/region restriction."}`,
	} {
		for _, field := range []string{"code", "message", "type"} {
			value := fieldValue(body, field)
			if value == "" {
				continue
			}
			assert.False(t, IsContentPolicySignal(value),
				"a provider failure must not be reported as a refusal: %s (%s=%q)", body, field, value)
		}
	}
}

// TestAzureContentFilterClientMessage pins that a refusal is reported to the
// client as the content-policy message, whatever wording the provider used.
func TestAzureContentFilterClientMessage(t *testing.T) {
	for _, body := range azureContentFilterBodies {
		classified := ClassifyBadRequest([]byte(body))
		assert.Equal(t, "The prompt was blocked by the provider's content policy", classified.Message,
			"a refusal must reach the client as the content-policy message: %s", body)
		assert.Equal(t, "content_policy_violation", classified.Code)
	}
}

// fieldValue pulls one field's string value out of a flat JSON object, so the
// test can check each field the classifier reads without depending on which
// of them a given branch exposes as an API.
func fieldValue(body, field string) string {
	var obj map[string]any
	if err := json.Unmarshal([]byte(body), &obj); err != nil {
		return ""
	}
	value, _ := obj[field].(string)
	return value
}
