package upstreamerror

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestContentPolicyMarkersClassifyEveryWording pins that EVERY wording in the
// shared marker list is recognized by the client-facing classifier.
//
// This is the guard against the drift that made the list four separate
// copies: three of them gained a wording while the classifier's own copy did
// not, so a refusal phrased that way was correctly classified as a
// non-retryable 400 yet reported to the client as the generic "Invalid
// request" instead of the content-policy message. The list is now shared, and
// this test fails if a wording is added to it without the classifier
// following.
func TestContentPolicyMarkersClassifyEveryWording(t *testing.T) {
	for _, marker := range ContentPolicyMarkers {
		t.Run(marker, func(t *testing.T) {
			body := []byte(`{"error":{"message":"Upstream refused: ` + marker + `"}}`)
			result := ClassifyBadRequest(body)
			assert.Equal(t, "content_policy_violation", result.Code,
				"every marker in ContentPolicyMarkers must classify as a content-policy refusal")
			assert.Equal(t, "The prompt was blocked by the provider's content policy", result.Message)
		})
	}
}

// TestIsContentPolicySignalMatchesClassifier is the cross-check that the two
// halves of the package agree: whatever IsContentPolicySignal accepts must
// also classify as a content-policy refusal, and vice versa. A wording
// recognized by one and not the other is exactly the inconsistency that let
// the drift go unnoticed.
func TestIsContentPolicySignalMatchesClassifier(t *testing.T) {
	phrasings := []string{
		"content policy",
		"content_policy",
		"content filter",
		"content_filter",
		"content moderation",
		"inappropriate content",
		"safety violation",
		"safety_violation",
		"prompt_blocked",
		"prompt blocked",
		"blocked by content",
		"content management policy",
		"policy violation",
		"prohibited_content",
		"model_armor",
		// Not refusals: a provider failure, a billing block, a geo
		// restriction, a parameter error.
		"inference generation failed",
		"Your account has been blocked by the provider for suspected billing fraud.",
		"blocked by the provider due to an IP/region restriction",
		"invalid parameter: max_tokens",
		"service unavailable",
	}

	for _, phrasing := range phrasings {
		t.Run(phrasing, func(t *testing.T) {
			signal := IsContentPolicySignal(phrasing)
			classified := ClassifyBadRequest([]byte(`{"error":{"message":"`+phrasing+`"}}`)).Code == "content_policy_violation"
			assert.Equal(t, signal, classified,
				"IsContentPolicySignal and ClassifyBadRequest must agree on what a content refusal is")
		})
	}
}

// TestHasContentPolicySignal checks the slice form used by the proxy's status
// classification.
func TestHasContentPolicySignal(t *testing.T) {
	assert.True(t, HasContentPolicySignal([]string{"provider_unavailable", "Output data may contain inappropriate content."}),
		"a refusal must be recognized among provider-side signals")
	assert.False(t, HasContentPolicySignal([]string{"provider_unavailable", "inference generation failed"}),
		"a provider failure must not be mistaken for a refusal")
	assert.False(t, HasContentPolicySignal(nil))
}

// TestContentPolicyMarkersRejectFalsePositives guards the markers against
// becoming so broad that they swallow unrelated errors: a bare "safety" would
// match a safety_identifier parameter error, and a bare "blocked by the
// provider" would swallow a billing block.
func TestContentPolicyMarkersRejectFalsePositives(t *testing.T) {
	for _, body := range []string{
		`{"error":{"message":"Your account has been blocked by the provider for suspected billing fraud.","code":"account_blocked"}}`,
		`{"error":{"message":"This request was blocked by the provider due to an IP/region restriction.","code":"region_restricted"}}`,
		`{"error":{"message":"safety_identifier must be a string","code":"invalid_parameter"}}`,
	} {
		result := ClassifyBadRequest([]byte(body))
		require.NotEqual(t, "content_policy_violation", result.Code,
			"an unrelated error must not be reported as a content refusal: %s", body)
	}
}
