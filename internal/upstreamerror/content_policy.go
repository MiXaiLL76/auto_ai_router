package upstreamerror

import "strings"

// ContentPolicyMarkers are the texts a provider uses when it refuses content
// on safety/policy grounds. A provider may phrase the refusal in the message
// ("Output data may contain inappropriate content."), in the error type
// ("content_filter"), in a machine-readable field ("prompt_blocked"), or in
// the prose of its own wording ("blocked by content") -- all of them mean the
// same thing.
//
// This is the single source of truth for that set. Four places need to know
// it:
//
//   - the proxy's status classification (a refusal is a 400, not a 5xx);
//   - the LiteLLM compatibility layer's own status mapping (it otherwise takes
//     the provider's numeric error code as the status verbatim);
//   - the retry policy (every credential refuses the same content, so the
//     request must not be replayed across the pool);
//   - this package's classifier (what the client is told the refusal was).
//
// They were four separate copies until the drift showed up: three of them
// gained a wording and the fourth did not, so a refusal phrased that way was
// classified as a 400 and treated as non-retryable, yet the client was told
// the generic "Invalid request" instead of the content-policy message. One
// list, imported everywhere, cannot drift.
//
// The markers are deliberately specific. A bare "safety" or "blocklist" would
// also match unrelated validation messages (a safety_identifier parameter
// error); a bare "blocked by the provider" would swallow billing/geo/IP
// blocks ("Your account has been blocked by the provider for suspected
// fraud"), replacing the real cause with a misleading content-policy message.
//
// "data_inspection_failed" is Alibaba's own code for the same refusal (it
// accompanies "Input/Output data may contain inappropriate content.", and
// occasionally arrives with a message that names no content at all -- "Provider
// returned error"). It is by far the most frequent content-refusal marker in
// production: ~164k of the refusals logged in a single month carry it, in the
// "code"/"type" field rather than the message.
var ContentPolicyMarkers = []string{
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
	"data_inspection_failed",
}

// IsContentPolicySignal reports whether a provider signal describes a
// content-policy refusal rather than a provider or credential failure.
//
// It matches the same markers as the classifier (plus the two the classifier
// already knew about before the list was introduced), so a refusal is
// classified as a content-policy 400 exactly when it is also reported as
// non-retryable and given the content-policy message.
func IsContentPolicySignal(signal string) bool {
	signal = strings.ToLower(strings.TrimSpace(signal))
	if signal == "" {
		return false
	}
	for _, marker := range ContentPolicyMarkers {
		if strings.Contains(signal, marker) {
			return true
		}
	}
	// "prohibited_content" and "model_armor" are the two remaining
	// content-policy phrasings this package recognized before the list
	// existed; they stay recognized here so the status classification and
	// the client-facing message keep agreeing.
	return strings.Contains(signal, "prohibited_content") || strings.Contains(signal, "model_armor")
}

// HasContentPolicySignal reports whether any of the given provider signals
// describes a content-policy refusal.
func HasContentPolicySignal(signals []string) bool {
	for _, signal := range signals {
		if IsContentPolicySignal(signal) {
			return true
		}
	}
	return false
}
