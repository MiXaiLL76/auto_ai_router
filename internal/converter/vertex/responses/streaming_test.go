package vertexresponses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func buildVertexSSEStream(events []map[string]interface{}) string {
	var sb strings.Builder
	for _, e := range events {
		b, _ := json.Marshal(e)
		sb.WriteString("data: ")
		sb.Write(b)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

func parseVertexSSEEvents(output string) []map[string]interface{} {
	var events []map[string]interface{}
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var e map[string]interface{}
		if json.Unmarshal([]byte(data), &e) == nil {
			events = append(events, e)
		}
	}
	return events
}

func TestTransformVertexStreamToResponses_MessageEventsIncludeRequiredFields(t *testing.T) {
	stream := buildVertexSSEStream([]map[string]interface{}{
		{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"role": "model",
						"parts": []map[string]interface{}{
							{"text": "hello"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]interface{}{
				"promptTokenCount":     5,
				"candidatesTokenCount": 3,
				"totalTokenCount":      8,
			},
		},
	})

	var out bytes.Buffer
	err := TransformVertexStreamToResponses(
		strings.NewReader(stream), &out, "gemini-test", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseVertexSSEEvents(out.String())
	require.NotEmpty(t, events)

	var messageItemID string
	for _, e := range events {
		_, hasSeq := e["sequence_number"]
		assert.True(t, hasSeq, "every event must include sequence_number: %#v", e)

		typ, _ := e["type"].(string)
		if typ == "response.output_item.added" {
			item, _ := e["item"].(map[string]interface{})
			if item != nil && item["type"] == "message" {
				messageItemID, _ = item["id"].(string)
			}
		}
	}
	require.NotEmpty(t, messageItemID)

	for _, e := range events {
		typ, _ := e["type"].(string)
		switch typ {
		case "response.content_part.added", "response.output_text.delta", "response.output_text.done", "response.content_part.done":
			itemID, _ := e["item_id"].(string)
			assert.Equal(t, messageItemID, itemID, "event %s must include matching item_id", typ)
		}
	}
}

func TestTransformVertexStreamToResponses_ImageGenerationCallAndUsage(t *testing.T) {
	stream := buildVertexSSEStream([]map[string]interface{}{
		{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"role": "model",
						"parts": []map[string]interface{}{
							{"inlineData": map[string]interface{}{"mimeType": "image/png", "data": "aW1hZ2U="}},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]interface{}{
				"promptTokenCount":     22,
				"candidatesTokenCount": 1120,
				"totalTokenCount":      1142,
				"candidatesTokensDetails": []map[string]interface{}{
					{"modality": "IMAGE", "tokenCount": 1120},
				},
			},
		},
	})

	var out bytes.Buffer
	err := TransformVertexStreamToResponses(
		strings.NewReader(stream), &out, "gemini-3.1-flash-image-preview", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseVertexSSEEvents(out.String())
	var completed map[string]interface{}
	var sawImageAdded, sawImageDone bool
	for _, event := range events {
		switch event["type"] {
		case "response.output_item.added":
			item, _ := event["item"].(map[string]interface{})
			sawImageAdded = sawImageAdded || item["type"] == "image_generation_call"
		case "response.output_item.done":
			item, _ := event["item"].(map[string]interface{})
			if item["type"] == "image_generation_call" {
				sawImageDone = true
				assert.Equal(t, "aW1hZ2U=", item["result"])
			}
		case "response.completed":
			completed, _ = event["response"].(map[string]interface{})
		}
	}
	assert.True(t, sawImageAdded)
	assert.True(t, sawImageDone)
	require.NotNil(t, completed)

	output := completed["output"].([]interface{})
	require.Len(t, output, 1)
	imageCall := output[0].(map[string]interface{})
	assert.Equal(t, "image_generation_call", imageCall["type"])
	assert.Equal(t, "aW1hZ2U=", imageCall["result"])

	usage := completed["usage"].(map[string]interface{})
	details := usage["output_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(1120), details["image_tokens"])
}
