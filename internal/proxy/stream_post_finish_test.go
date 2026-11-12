package proxy

import (
	"io"
	"strings"
	"testing"

	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
	"github.com/stretchr/testify/require"
)

func TestChatPostFinishDetectorUsesSanitizerParse(t *testing.T) {
	source := strings.NewReader(strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":"stop"},{"index":1,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}`,
		`data: {"choices":[{"index":1,"delta":{"content":"still running"},"finish_reason":null}]}`,
		`data: {"choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":null}]}`,
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":"late"},"finish_reason":null}]}`,
		`data: {"choices":[{"index":0,"delta":{"content":"another late delta"}}]}`,
		`data: {"choices":[{"index":1,"delta":{},"finish_reason":"length"}]}`,
		`data: {"choices":[{"index":1,"delta":{"content":"late answer"}}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n")
	var detector chatPostFinishDetector
	var violations []chatPostFinishViolation
	reader := promanutils.NewSanitizingSSEReaderWithObserver(source, "test-model", func(value any) {
		violations = append(violations, detector.observeJSON(value)...)
	})
	_, err := io.Copy(io.Discard, reader)
	require.NoError(t, err)
	require.Equal(t, []chatPostFinishViolation{
		{index: 0, finishReason: "stop", field: "reasoning_content"},
		{index: 1, finishReason: "length", field: "content"},
	}, violations)
}

func TestChatPostFinishDetectorReassemblesSplitSSELine(t *testing.T) {
	source := io.MultiReader(
		strings.NewReader("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"),
		strings.NewReader("data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_"),
		strings.NewReader("content\":\"late\"}}]}\n\n"),
	)
	var detector chatPostFinishDetector
	var violations []chatPostFinishViolation
	reader := promanutils.NewSanitizingSSEReaderWithObserver(source, "test-model", func(value any) {
		violations = append(violations, detector.observeJSON(value)...)
	})
	_, err := io.Copy(io.Discard, reader)
	require.NoError(t, err)
	require.Equal(t, []chatPostFinishViolation{{index: 0, finishReason: "stop", field: "reasoning_content"}}, violations)
}
