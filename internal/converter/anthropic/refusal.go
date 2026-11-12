package anthropic

// RefusalMessage exposes the upstream explanation and category through the
// standard OpenAI refusal field. Missing details must still produce an explicit
// refusal; an empty answer alone is never evidence of a refusal.
func RefusalMessage(stopReason string, details *StopDetails) string {
	if stopReason != "refusal" {
		return ""
	}
	message := "Request refused by Anthropic."
	if details != nil {
		if details.Explanation != "" {
			message = details.Explanation
		}
		if details.Category != "" {
			message += " (category: " + details.Category + ")"
		}
	}
	return message
}
