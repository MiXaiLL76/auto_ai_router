package proxy

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGeminiStreamSearchGroundingBilling merges a streamed Gemini answer's usage
// chunks the way the stream handler does (each over the previous ones). Gemini
// reports the tool-use (search) context from the first chunk on but the grounding
// metadata only with the last, so on a model priced without its search context the
// early chunks must not leave the search results' image tokens behind in the
// merged usage; any other model keeps the search context billed.
func TestGeminiStreamSearchGroundingBilling(t *testing.T) {
	usage := func(candidates, total int) string {
		return fmt.Sprintf(`"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":900,`+
			`"toolUsePromptTokensDetails":[{"modality":"TEXT","tokenCount":300},{"modality":"IMAGE","tokenCount":600}],`+
			`"candidatesTokenCount":%d,"totalTokenCount":%d}`, candidates, total)
	}
	stream := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"Here"}]}}],` + usage(10, 1010) + "}\n\n" +
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":" it is"}]},"finishReason":"STOP",` +
		`"groundingMetadata":{"webSearchQueries":["timareta butterfly"],"imageSearchQueries":["timareta butterfly"]}}],` +
		usage(50, 1050) + "}\n\n"

	tests := []struct {
		model        string
		promptTokens int
		imageTokens  int
	}{
		{model: "gemini-nano-banana-2.1", promptTokens: 100, imageTokens: 0},
		{model: "gemini-2.5-flash", promptTokens: 1000, imageTokens: 600},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			var out bytes.Buffer
			require.NoError(t, vertex.TransformVertexStreamToOpenAI(strings.NewReader(stream), tt.model, &out))

			var merged *converter.TokenUsage
			for _, frame := range strings.SplitAfter(out.String(), "\n\n") {
				payloads := splitSSEPayloads([]byte(frame), nil)
				if usage := extractTokenUsageFromPayloads(payloads, converter.TokenUsageExtractionOptions{}); usage != nil {
					if merged == nil {
						merged = &converter.TokenUsage{}
					}
					merged.MergeNonZero(usage)
				}
			}
			require.NotNil(t, merged)
			assert.Equal(t, tt.promptTokens, merged.PromptTokens)
			assert.Equal(t, tt.imageTokens, merged.ImageTokens)
			assert.Equal(t, 50, merged.CompletionTokens)
			assert.Equal(t, 2, merged.WebSearchRequests, "one web and one image search query")
		})
	}
}
