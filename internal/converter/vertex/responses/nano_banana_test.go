package vertexresponses

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The Responses API route to gemini-nano-banana-2.1 must shape the request exactly
// as the Chat Completions route does; see vertex/nano_banana_test.go.
const nanoBanana21 = "gemini-nano-banana-2.1"

func nanoResponsesRequest(t *testing.T, params string) map[string]interface{} {
	t.Helper()
	body := fmt.Sprintf(`{"model":%q,"input":"Draw a banana"%s}`, nanoBanana21, params)
	out, err := ResponsesRequestToVertex([]byte(body), nanoBanana21)
	require.NoError(t, err)
	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &req))
	return req
}

func TestResponsesRequestToVertex_NanoBanana21Thinking(t *testing.T) {
	tests := []struct {
		name   string
		params string
		level  string
	}{
		{name: "default", params: "", level: "MEDIUM"},
		{name: "low becomes minimal", params: `,"reasoning":{"effort":"low"}`, level: "MINIMAL"},
		{name: "high", params: `,"reasoning":{"effort":"high"}`, level: "HIGH"},
		{name: "none is the lowest level", params: `,"reasoning":{"effort":"none"}`, level: "MINIMAL"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := nanoResponsesRequest(t, tt.params)["generationConfig"].(map[string]interface{})
			thinking := cfg["thinkingConfig"].(map[string]interface{})
			assert.Equal(t, tt.level, thinking["thinkingLevel"])
			assert.NotContains(t, thinking, "thinkingBudget")
		})
	}
}

func TestResponsesRequestToVertex_NanoBanana21DropsUnsupportedSamplingParams(t *testing.T) {
	cfg := nanoResponsesRequest(t, `,"temperature":0.7,"top_p":0.9`)["generationConfig"].(map[string]interface{})
	assert.NotContains(t, cfg, "temperature")
	assert.NotContains(t, cfg, "topP")
}

func TestResponsesRequestToVertex_GoogleSearchTypes(t *testing.T) {
	tests := []struct {
		name  string
		tools string
		want  string
	}{
		{
			name:  "image search",
			tools: `[{"type":"web_search","search_types":["image_search"]}]`,
			want:  `[{"googleSearch":{"searchTypes":{"imageSearch":{}}}}]`,
		},
		{
			name:  "web and image search",
			tools: `[{"type":"google_search","search_types":["web_search","image_search"]}]`,
			want:  `[{"googleSearch":{"searchTypes":{"webSearch":{},"imageSearch":{}}}}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tools, err := json.Marshal(nanoResponsesRequest(t, `,"tools":`+tt.tools)["tools"])
			require.NoError(t, err)
			assert.JSONEq(t, tt.want, string(tools))
		})
	}

	t.Run("unknown search type", func(t *testing.T) {
		body := fmt.Sprintf(`{"model":%q,"input":"x","tools":[{"type":"web_search","search_types":["video_search"]}]}`, nanoBanana21)
		_, err := ResponsesRequestToVertex([]byte(body), nanoBanana21)
		testhelpers.RequireValidationError(t, err, "tools[0].search_types", "invalid_value")
	})
}

func TestResponsesRequestToVertex_NanoBanana21ReferenceImageLimit(t *testing.T) {
	request := func(count int) []byte {
		content := []interface{}{map[string]interface{}{"type": "input_text", "text": "Combine these"}}
		for i := 0; i < count; i++ {
			data := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("image-%02d", i)))
			content = append(content, map[string]interface{}{"type": "input_image", "image_url": "data:image/png;base64," + data})
		}
		body, err := json.Marshal(map[string]interface{}{
			"model": nanoBanana21,
			"input": []interface{}{map[string]interface{}{"role": "user", "content": content}},
		})
		require.NoError(t, err)
		return body
	}

	out, err := ResponsesRequestToVertex(request(14), nanoBanana21)
	require.NoError(t, err)
	assert.Equal(t, 14, strings.Count(string(out), `"inlineData"`))

	_, err = ResponsesRequestToVertex(request(15), nanoBanana21)
	testhelpers.RequireValidationError(t, err, "image", "too_many_images")
}

// TestResponsesRequestToVertex_ImageGenerationSize pins that the image_generation
// tool's size reaches Google as the imageConfig the images endpoints map it to.
func TestResponsesRequestToVertex_ImageGenerationSize(t *testing.T) {
	imageConfig := func(t *testing.T, model, tool string) map[string]interface{} {
		t.Helper()
		body := fmt.Sprintf(`{"model":%q,"input":"Draw a banana","tools":[%s]}`, model, tool)
		out, err := ResponsesRequestToVertex([]byte(body), model)
		require.NoError(t, err)
		var req struct {
			GenerationConfig struct {
				ResponseModalities []string               `json:"responseModalities"`
				ImageConfig        map[string]interface{} `json:"imageConfig"`
			} `json:"generationConfig"`
		}
		require.NoError(t, json.Unmarshal(out, &req))
		assert.Equal(t, []string{"IMAGE"}, req.GenerationConfig.ResponseModalities)
		return req.GenerationConfig.ImageConfig
	}

	assert.Equal(t, map[string]interface{}{"aspectRatio": "2:3", "imageSize": "1K"},
		imageConfig(t, nanoBanana21, `{"type":"image_generation","size":"1024x1536"}`))
	// The official grid size maps exactly; 512px is never picked for 2.1.
	assert.Equal(t, map[string]interface{}{"aspectRatio": "16:9", "imageSize": "2K"},
		imageConfig(t, nanoBanana21, `{"type":"image_generation","size":"2752x1536"}`))
	assert.Equal(t, map[string]interface{}{"aspectRatio": "1:1", "imageSize": "1K"},
		imageConfig(t, nanoBanana21, `{"type":"image_generation","size":"512x512"}`))
	// A model with a fixed resolution gets the ratio only.
	assert.Equal(t, map[string]interface{}{"aspectRatio": "3:2"},
		imageConfig(t, "gemini-2.5-flash-image", `{"type":"image_generation","size":"1536x1024"}`))
	for _, tool := range []string{`{"type":"image_generation"}`, `{"type":"image_generation","size":"auto"}`,
		`{"type":"image_generation","size":1024}`} {
		assert.Nil(t, imageConfig(t, nanoBanana21, tool), tool)
	}

	body := fmt.Sprintf(`{"model":%q,"input":"x","tools":[{"type":"image_generation","size":"big"}]}`, nanoBanana21)
	_, err := ResponsesRequestToVertex([]byte(body), nanoBanana21)
	testhelpers.RequireValidationError(t, err, "size", "invalid_image_size")
}

// TestResponsesRequestToVertex_ReasoningEffortMatchesChat pins that
// reasoning.effort resolves as the chat route's reasoning_effort does on models
// without a profile, "none" included.
func TestResponsesRequestToVertex_ReasoningEffortMatchesChat(t *testing.T) {
	tests := []struct {
		model, effort string
		want          string
	}{
		{model: "gemini-2.5-pro", effort: "none", want: `{"thinkingBudget":-1}`},
		{model: "gemini-2.5-flash", effort: "none", want: `{"thinkingBudget":0}`},
		{model: "gemini-3-pro-preview", effort: "none", want: `{"thinkingLevel":"LOW"}`},
		{model: "gemini-3-flash-preview", effort: "xhigh", want: `{"thinkingLevel":"HIGH"}`},
		// Image models without a profile take no ThinkingConfig on either route.
		{model: "gemini-2.5-flash-image", effort: "low", want: `null`},
		{model: "gemini-3.1-flash-image-preview", effort: "none", want: `null`},
		// Nor does none on a model without thinking, where a zero budget could be rejected.
		{model: "gemini-2.0-flash", effort: "none", want: `null`},
	}
	for _, tt := range tests {
		t.Run(tt.model+" "+tt.effort, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":%q,"input":"x","reasoning":{"effort":%q}}`, tt.model, tt.effort)
			out, err := ResponsesRequestToVertex([]byte(body), tt.model)
			require.NoError(t, err)
			var req struct {
				GenerationConfig struct {
					ThinkingConfig json.RawMessage `json:"thinkingConfig"`
				} `json:"generationConfig"`
			}
			require.NoError(t, json.Unmarshal(out, &req))
			assert.JSONEq(t, tt.want, cmp.Or(string(req.GenerationConfig.ThinkingConfig), "null"))
		})
	}
}

func TestVertexToResponsesResponse_GoogleSearchGroundingBilling(t *testing.T) {
	body := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP",` +
		`"groundingMetadata":{"webSearchQueries":["butterfly"],"imageSearchQueries":["butterfly","flower"]}}],` +
		`"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":900,"candidatesTokenCount":50,"totalTokenCount":1050}}`)
	resp, err := VertexToResponsesResponse(body, nanoBanana21, "resp_1", 1)
	require.NoError(t, err)

	require.NotNil(t, resp.Usage)
	assert.Equal(t, 100, resp.Usage.InputTokens, "search-retrieved context is not billed as input")
	assert.Equal(t, 150, resp.Usage.TotalTokens)
	require.NotNil(t, resp.Usage.ServerToolUse)
	assert.Equal(t, 3, resp.Usage.ServerToolUse.WebSearchRequests)

	// A model priced with its search context keeps it billed.
	resp, err = VertexToResponsesResponse(body, "gemini-2.5-flash", "resp_2", 1)
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 1000, resp.Usage.InputTokens)
	assert.Equal(t, 1050, resp.Usage.TotalTokens)

	var queries []string
	for _, item := range resp.Output {
		if item.Type == "web_search_call" {
			queries = append(queries, item.Queries...)
		}
	}
	assert.Equal(t, []string{"butterfly", "butterfly", "flower"}, queries)
}

func TestTransformVertexStreamToResponses_GoogleSearchGroundingBilling(t *testing.T) {
	stream := buildVertexSSEStream([]map[string]interface{}{{
		"candidates": []map[string]interface{}{{
			"content":      map[string]interface{}{"role": "model", "parts": []map[string]interface{}{{"text": "ok"}}},
			"finishReason": "STOP",
			"groundingMetadata": map[string]interface{}{
				"imageSearchQueries": []string{"butterfly"},
			},
		}},
		"usageMetadata": map[string]interface{}{
			"promptTokenCount":        100,
			"toolUsePromptTokenCount": 900,
			"candidatesTokenCount":    50,
			"totalTokenCount":         1050,
		},
	}})

	var out bytes.Buffer
	require.NoError(t, TransformVertexStreamToResponses(strings.NewReader(stream), &out, nanoBanana21, "", nil, nil))

	for _, event := range parseVertexSSEEvents(out.String()) {
		if event["type"] != "response.completed" {
			continue
		}
		usage := event["response"].(map[string]interface{})["usage"].(map[string]interface{})
		assert.Equal(t, float64(100), usage["input_tokens"])
		assert.Equal(t, float64(1), usage["server_tool_use"].(map[string]interface{})["web_search_requests"])
		return
	}
	t.Fatal("missing response.completed event")
}

func TestTransformVertexStreamToResponses_SkipsThoughtImages(t *testing.T) {
	stream := buildVertexSSEStream([]map[string]interface{}{{
		"candidates": []map[string]interface{}{{
			"content": map[string]interface{}{"role": "model", "parts": []map[string]interface{}{
				{"thought": true, "inlineData": map[string]interface{}{"mimeType": "image/png", "data": "ZHJhZnQ="}},
				{"inlineData": map[string]interface{}{"mimeType": "image/png", "data": "ZmluYWw="}},
			}},
			"finishReason": "STOP",
		}},
	}})

	var completed *responses.Response
	var out bytes.Buffer
	require.NoError(t, TransformVertexStreamToResponses(strings.NewReader(stream), &out, nanoBanana21, "", nil,
		func(resp *responses.Response) { completed = resp }))

	require.NotNil(t, completed)
	var images []string
	for _, item := range completed.Output {
		if item.Type == "image_generation_call" {
			images = append(images, item.Result)
		}
	}
	assert.Equal(t, []string{"ZmluYWw="}, images, "only the final image is a result")
}
