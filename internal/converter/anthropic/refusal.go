package anthropic

import "github.com/mixaill76/auto_ai_router/internal/converter/publicrefusal"

// RefusalMessage exposes a safe explanation of the upstream category through the
// standard OpenAI refusal field. Missing details must still produce an explicit
// refusal; an empty answer alone is never evidence of a refusal.
func RefusalMessage(stopReason string, details *StopDetails) string {
	if stopReason != "refusal" {
		return ""
	}
	category := ""
	if details != nil {
		category = details.Category
	}
	return publicrefusal.Message(category)
}
