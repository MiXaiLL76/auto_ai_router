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
	// Azure's content filter. It reports the refusal under a code of its own
	// and a message that names no content at all -- "The model was unable to
	// complete inference due to an internal error." -- so without these
	// markers the refusal was classified as a generic error, stayed retryable
	// across the whole credential pool, and reached the client as a 502.
	//
	// The wording is specific enough to stand on its own: a provider-side
	// failure says "inference generation failed", not "unable to complete
	// inference". Azure's own codes are listed too, for the shapes that
	// carry no message at all.
	//
	// "invalid_prompt" is deliberately NOT here: Azure uses it for ordinary
	// prompt problems too, so treating it as a refusal would turn every bad
	// prompt into a content-policy one.
	"unable to complete inference",
	"responsibleaipolicyviolation",
	"responsible ai policy violation",
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

// ContentPolicyErrorFields are the keys of a provider's error object that may
// carry the reason for the failure.
//
// "code" and "status" are included alongside the obvious "type"/"message"/
// "reason": a provider may put the refusal in any of them ("status":
// "content_filter", "code": "content_policy_violation"), and a check that
// reads only the message-bearing fields misses those. They are read as TEXT
// only -- a numeric "code" describes the upstream's failure and says nothing
// about the cause.
var ContentPolicyErrorFields = []string{"code", "type", "message", "status", "reason"}

// ContentPolicyErrorMetadataFields are the keys of a provider's error
// object's "metadata" sub-object that may carry the reason. Aggregators put
// the machine-readable one ("error_type": "provider_unavailable") there,
// next to the human-readable message.
var ContentPolicyErrorMetadataFields = []string{"error_type", "type", "reason"}

// IsContentPolicyErrorObject reports whether a provider's error object
// describes a content-policy refusal.
//
// It reads the same fields wherever the error object is inspected -- the
// proxy's own classification and the LiteLLM compatibility layer's -- so the
// two cannot disagree about what a refusal is. They used to: the proxy read
// "code" and "status" as well, while the compatibility layer read only the
// message-bearing fields, so a refusal the proxy recognized was passed
// through as a 5xx when the same body went through the compatibility layer.
//
// obj must be the decoded error object (a map), as the compatibility layer
// has it.
func IsContentPolicyErrorObject(obj map[string]any) bool {
	if obj == nil {
		return false
	}
	for _, key := range ContentPolicyErrorFields {
		if s, ok := obj[key].(string); ok && IsContentPolicySignal(s) {
			return true
		}
	}
	if meta, ok := obj["metadata"].(map[string]any); ok {
		for _, key := range ContentPolicyErrorMetadataFields {
			if s, ok := meta[key].(string); ok && IsContentPolicySignal(s) {
				return true
			}
		}
	}
	return false
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
