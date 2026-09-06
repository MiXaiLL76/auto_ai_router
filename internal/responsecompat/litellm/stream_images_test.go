package litellm

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChatStreamPreservesImageOnlyDelta(t *testing.T) {
	for _, tc := range []struct {
		name   string
		size   int
		finish string
	}{
		{name: "image after text", size: 32, finish: "null"},
		{name: "image with finish", size: 32, finish: `"stop"`},
		{name: "image larger than one MiB", size: 2 * 1024 * 1024, finish: "null"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			imageURL := "data:image/png;base64," + strings.Repeat("A", tc.size)
			source := `data: {"id":"image-stream","model":"provider-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Here is the image"},"finish_reason":null}]}` + "\n\n" +
				`data: {"id":"image-stream","model":"provider-model","choices":[{"index":0,"delta":{"images":[{"type":"image_url","image_url":{"url":"` + imageURL + `"}}]},"finish_reason":` + tc.finish + `}]}` + "\n\n"
			if tc.finish == "null" {
				source += `data: {"id":"image-stream","model":"provider-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
			}
			source += `data: {"id":"image-stream","model":"provider-model","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":1290,"total_tokens":1304,"completion_tokens_details":{"image_tokens":1290}}}` + "\n\ndata: [DONE]\n\n"
			output, err := io.ReadAll(New().Stream(Context{
				Endpoint: "/v1/chat/completions", RequestedModel: "public-model", IncludeUsage: true,
			}, strings.NewReader(source)))
			require.NoError(t, err)
			frames := splitDataFrames(string(output))
			require.NotEmpty(t, frames)
			require.Equal(t, "[DONE]", frames[len(frames)-1])
			images, finishes, usages := 0, 0, 0
			for _, frame := range frames[:len(frames)-1] {
				var body map[string]any
				require.NoError(t, json.Unmarshal([]byte(frame), &body))
				require.Equal(t, "public-model", body["model"])
				if usage, ok := body["usage"].(map[string]any); ok {
					usages++
					require.Equal(t, float64(1304), usage["total_tokens"])
					require.Equal(t, float64(1290), usage["completion_tokens_details"].(map[string]any)["image_tokens"])
				}
				for _, raw := range body["choices"].([]any) {
					choice := raw.(map[string]any)
					delta := choice["delta"].(map[string]any)
					if values, ok := delta["images"].([]any); ok {
						images += len(values)
						require.Equal(t, imageURL, values[0].(map[string]any)["image_url"].(map[string]any)["url"])
						require.Nil(t, choice["finish_reason"])
					}
					if choice["finish_reason"] == "stop" {
						finishes++
					}
				}
			}
			require.Equal(t, 1, images)
			require.Equal(t, 1, finishes)
			require.Equal(t, 1, usages)
		})
	}
}
