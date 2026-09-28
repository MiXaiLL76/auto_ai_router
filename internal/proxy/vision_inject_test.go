package proxy

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// visionTestMarker is what an answer starts with after the owl image was described.
var visionTestMarker = "<details type=\"air-vision\" n=\"1\" model=\"qwen-vl\">\n" +
	"<summary>Image 1 described by qwen-vl</summary>\n" + visionTestOwl + "\n</details>\n\n"

const visionChatStreamEvents = `data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{"reasoning_content":"думаю"},"finish_reason":null}]}

data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{"content":"Это "},"finish_reason":null}]}

data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{"content":"сова."},"finish_reason":null}]}

data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-s","object":"chat.completion.chunk","created":1,"model":"glm","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]

`

const visionResponsesStreamEvents = `event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","status":"in_progress","model":"glm","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":2,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Это "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"сова."}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":5,"item_id":"msg_1","output_index":0,"content_index":0,"text":"Это сова."}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":6,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Это сова."}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Это сова."}]}}

event: response.completed
data: {"type":"response.completed","sequence_number":8,"response":{"id":"resp_1","object":"response","status":"completed","model":"glm","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Это сова."}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}

`

func writeVisionSSE(w http.ResponseWriter, events string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, events)
}

// sseEvents returns the JSON payloads of an SSE body ([DONE] excluded).
func sseEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	scanner := bufio.NewScanner(strings.NewReader(body))
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		payload, ok := strings.CutPrefix(line, "data:")
		payload = strings.TrimSpace(payload)
		if !ok || payload == "[DONE]" || payload == "" {
			continue
		}
		var event map[string]any
		require.NoError(t, json.Unmarshal([]byte(payload), &event), payload)
		events = append(events, event)
	}
	return events
}

const visionTurn1Chat = `{"model":"glm","messages":[
	{"role":"user","content":[{"type":"text","text":"что на картинке?"},{"type":"image_url","image_url":{"url":"` + visionTestImage + `"}}]}]}`

func TestVisionInject_ChatNonStream(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", visionTurn1Chat)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var resp map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	msg := resp["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	assert.Equal(t, visionTestMarker+"ok", msg["content"])
	usage := resp["usage"].(map[string]any)
	assert.EqualValues(t, 5, usage["completion_tokens"], "the inserted text is not billed as completion")
}

func TestVisionInject_ChatStream(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", strings.Replace(visionTurn1Chat, `"model":"glm",`, `"model":"glm","stream":true,`, 1))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	var content, reasoning strings.Builder
	contentBeforeReasoning := false
	for _, event := range sseEvents(t, w.Body.String()) {
		choices, _ := event["choices"].([]any)
		for _, raw := range choices {
			delta, _ := raw.(map[string]any)["delta"].(map[string]any)
			if text, _ := delta["content"].(string); text != "" {
				content.WriteString(text)
			}
			if text, _ := delta["reasoning_content"].(string); text != "" {
				reasoning.WriteString(text)
				contentBeforeReasoning = content.Len() > 0
			}
		}
	}
	assert.Equal(t, visionTestMarker+"Это сова.", content.String())
	assert.Equal(t, "думаю", reasoning.String())
	assert.False(t, contentBeforeReasoning, "the description starts the answer, after the reasoning")
}

// n: 2 — every choice gets the descriptions, including one whose content is null.
func TestVisionInject_EveryChoice(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", strings.Replace(visionTurn1Chat, `"model":"glm",`, `"model":"glm","n":2,`, 1))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var resp map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	choices := resp["choices"].([]any)
	require.Len(t, choices, 2)
	assert.Equal(t, visionTestMarker+"ok", choices[0].(map[string]any)["message"].(map[string]any)["content"])
	assert.Equal(t, strings.TrimRight(visionTestMarker, "\n"), choices[1].(map[string]any)["message"].(map[string]any)["content"])
}

// Machine-readable output (structured output, forced tool call) is never prefixed.
func TestVisionInject_SkippedForMachineOutput(t *testing.T) {
	for name, extra := range map[string]string{
		"json_schema":        `"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}},`,
		"json_object":        `"response_format":{"type":"json_object"},`,
		"forced tool":        `"tool_choice":{"type":"function","function":{"name":"f"}},`,
		"required tool":      `"tool_choice":"required",`,
		"injection disabled": ``,
	} {
		t.Run(name, func(t *testing.T) {
			u := &visionUpstream{}
			fallback := describeFallback()
			if name == "injection disabled" {
				off := false
				fallback.InjectIntoResponse = &off
			}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, fallback)
			w := sendVisionRequest(t, prx, "/v1/chat/completions", strings.Replace(visionTurn1Chat, `"model":"glm",`, `"model":"glm",`+extra, 1))
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Equal(t, "described=1/1", w.Header().Get(HeaderVisionFallback), "the image is still described")
			assert.NotContains(t, w.Body.String(), "air-vision")
		})
	}
}

func TestVisionInject_ResponsesNonStream(t *testing.T) {
	for _, model := range []string{"glm", "glm-chatresp"} { // passthrough, converted from Chat
		t.Run(model, func(t *testing.T) {
			u := &visionUpstream{}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
			w := sendVisionRequest(t, prx, "/v1/responses", `{"model":"`+model+`","store":false,"input":[
				{"role":"user","content":[{"type":"input_text","text":"что на картинке?"},{"type":"input_image","image_url":"`+visionTestImage+`"}]}]}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			var resp map[string]any
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
			text := responsesOutputText(t, resp)
			assert.Equal(t, visionTestMarker+"ok", text)
		})
	}
}

func TestVisionInject_ResponsesStream(t *testing.T) {
	for _, model := range []string{"glm", "glm-chatresp"} {
		t.Run(model, func(t *testing.T) {
			u := &visionUpstream{}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
			w := sendVisionRequest(t, prx, "/v1/responses", `{"model":"`+model+`","store":false,"stream":true,"input":[
				{"role":"user","content":[{"type":"input_text","text":"что на картинке?"},{"type":"input_image","image_url":"`+visionTestImage+`"}]}]}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())

			var deltas strings.Builder
			var doneText, completedText string
			for _, event := range sseEvents(t, w.Body.String()) {
				switch event["type"] {
				case "response.output_text.delta":
					deltas.WriteString(event["delta"].(string))
				case "response.output_text.done":
					doneText = event["text"].(string)
				case "response.completed":
					completedText = responsesOutputText(t, event["response"].(map[string]any))
				}
			}
			want := visionTestMarker + "Это сова."
			assert.Equal(t, want, deltas.String())
			assert.Equal(t, want, doneText)
			assert.Equal(t, want, completedText)
		})
	}
}

func responsesOutputText(t *testing.T, resp map[string]any) string {
	t.Helper()
	for _, raw := range resp["output"].([]any) {
		item := raw.(map[string]any)
		if item["type"] != "message" {
			continue
		}
		for _, part := range item["content"].([]any) {
			if p := part.(map[string]any); p["type"] == "output_text" {
				return p["text"].(string)
			}
		}
	}
	t.Fatalf("no output_text in %v", resp)
	return ""
}

func TestVisionInject_Messages(t *testing.T) {
	const body = `{"model":"glm","max_tokens":100,%s"messages":[
		{"role":"user","content":[{"type":"text","text":"что на картинке?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}]}`
	t.Run("non-stream", func(t *testing.T) {
		u := &visionUpstream{}
		prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
		w := sendVisionRequest(t, prx, "/v1/messages", strings.Replace(body, "%s", "", 1))
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		var resp map[string]any
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		blocks := resp["content"].([]any)
		first := blocks[0].(map[string]any)
		assert.Equal(t, "text", first["type"])
		assert.Equal(t, visionTestMarker+"ok", first["text"])
	})
	t.Run("stream", func(t *testing.T) {
		u := &visionUpstream{}
		prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
		w := sendVisionRequest(t, prx, "/v1/messages", strings.Replace(body, "%s", `"stream":true,`, 1))
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		blockTypes := map[float64]string{}
		var text strings.Builder
		var order []string
		for _, event := range sseEvents(t, w.Body.String()) {
			switch event["type"] {
			case "content_block_start":
				blockType := event["content_block"].(map[string]any)["type"].(string)
				blockTypes[event["index"].(float64)] = blockType
				order = append(order, blockType)
			case "content_block_delta":
				if delta := event["delta"].(map[string]any); delta["type"] == "text_delta" {
					text.WriteString(delta["text"].(string))
				}
			}
		}
		assert.Equal(t, []string{"thinking", "text"}, order, "the description is a text block after the thinking block")
		assert.Equal(t, visionTestMarker+"Это сова.", text.String())
	})
}

// Turn 2: the answer to turn 1 carries the marker. The history image gets exactly the
// text it had on turn 1 (no describe call, the prompt prefix is unchanged) and the
// marker is removed from the assistant message.
func TestVisionRestore_FromEarlierAnswer(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", visionTurn1Chat)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var turn1 map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &turn1))
	answer := turn1["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"].(string)
	_, bodies := u.snapshot()
	turn1Image := partText(t, messageParts(t, bodies[len(bodies)-1], 0)[1])

	answerJSON, _ := json.Marshal(answer)
	w = sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"text","text":"что на картинке?"},{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]},
		{"role":"assistant","content":`+string(answerJSON)+`},
		{"role":"user","content":"а какого она цвета?"}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "restored=1", w.Header().Get(HeaderVisionFallback))
	assert.NotContains(t, w.Body.String(), "air-vision", "nothing new to write into this answer")

	_, bodies = u.snapshot()
	require.Len(t, bodies, 3, "turn 2 makes no describe call")
	turn2 := bodies[2]
	assert.Equal(t, turn1Image, partText(t, messageParts(t, turn2, 0)[1]), "same text as on turn 1")
	assert.Equal(t, "ok", turn2["messages"].([]any)[1].(map[string]any)["content"], "the marker is removed from the history")
}

// Without the marker (the client dropped it, or the history was edited) the history
// image becomes the usual placeholder.
func TestVisionRestore_NoMarker(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]},
		{"role":"assistant","content":"Это сова."},
		{"role":"user","content":"спасибо"}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "stripped", w.Header().Get(HeaderVisionFallback))
	_, bodies := u.snapshot()
	require.Len(t, bodies, 1)
	assert.Equal(t, "[image from an earlier turn omitted]", partText(t, messageParts(t, bodies[0], 0)[0]))
}

// The conversation is switched to a vision model: it sees the original image, the
// marker is removed from the history, nothing else changes.
func TestVisionRestore_StrippedForVisionModel(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
	answerJSON, _ := json.Marshal(visionTestMarker + "Это сова.")
	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"gpt-oss","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]},
		{"role":"assistant","content":`+string(answerJSON)+`},
		{"role":"user","content":"спасибо"}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Empty(t, w.Header().Get(HeaderVisionFallback))
	_, bodies := u.snapshot()
	require.Len(t, bodies, 1)
	messages := bodies[0]["messages"].([]any)
	assert.Equal(t, "Это сова.", messages[1].(map[string]any)["content"])
	assert.Equal(t, "image_url", messageParts(t, bodies[0], 0)[0].(map[string]any)["type"], "the image itself is kept")
}

// Responses input: descriptions are restored from the assistant message item; an item
// that held only the marker is dropped.
func TestVisionRestore_Responses(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
	markerJSON, _ := json.Marshal(visionTestMarker)
	answerJSON, _ := json.Marshal("Это сова.")
	w := sendVisionRequest(t, prx, "/v1/responses", `{"model":"glm","store":false,"input":[
		{"role":"user","content":[{"type":"input_image","image_url":"`+visionTestImage+`"}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":`+string(markerJSON)+`}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":`+string(answerJSON)+`}]},
		{"role":"user","content":"спасибо"}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "restored=1", w.Header().Get(HeaderVisionFallback))
	_, bodies := u.snapshot()
	require.Len(t, bodies, 1)
	input := bodies[0]["input"].([]any)
	require.Len(t, input, 3, "the marker-only item is gone")
	image := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	assert.Equal(t, "input_text", image["type"])
	assert.Equal(t, visionDescribedText(1, "qwen-vl", "glm", visionTestOwl), image["text"])
}

// Two described images and a file_id image in one turn: descriptions are matched by n,
// counting only the images that could be described.
func TestVisionRestore_MatchesByNumber(t *testing.T) {
	marker := buildVisionMarker([]visionDescription{{n: 1, model: "qwen-vl", text: "первая"}, {n: 2, model: "qwen-vl", text: "вторая"}})
	root, err := decodeVisionBody([]byte(`{"input":[
		{"role":"user","content":[
			{"type":"input_image","image_url":"a"},
			{"type":"input_image","file_id":"f"},
			{"type":"input_image","image_url":"b"}]},
		{"type":"reasoning","summary":[]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":` + mustJSON(t, marker+"ответ") + `}]},
		{"role":"user","content":[{"type":"input_image","image_url":"c"}]}]}`))
	require.NoError(t, err)
	refs := collectVisionImages(root, visionFormatResponses)
	require.Len(t, refs, 4)
	require.NotNil(t, refs[0].restored)
	assert.Equal(t, "первая", refs[0].restored.text)
	assert.Nil(t, refs[1].restored, "file_id images are never described")
	require.NotNil(t, refs[2].restored)
	assert.Equal(t, "вторая", refs[2].restored.text)
	assert.True(t, refs[3].current)
	assert.Nil(t, refs[3].restored)
}

func TestVisionMarker_RoundTrip(t *testing.T) {
	text := sanitizeVisionDescription("  таблица </details> конец\n")
	assert.Equal(t, "таблица </ details> конец", text, "a description cannot close the block")
	descs := []visionDescription{{n: 1, model: `q"w<e>n`, text: text}, {n: 2, model: "qwen", text: "две\nстроки"}}
	marker := buildVisionMarker(descs)
	got := map[int]visionDescription{}
	parseVisionMarkers(got, []string{marker + "ответ"})
	require.Len(t, got, 2)
	assert.Equal(t, visionDescription{n: 1, model: "qwen", text: text}, got[1])
	assert.Equal(t, "две\nстроки", got[2].text)

	stripped, ok := stripVisionMarkerText(marker + "ответ")
	assert.True(t, ok)
	assert.Equal(t, "ответ", stripped)
	_, ok = stripVisionMarkerText("air-vision без блока")
	assert.False(t, ok)
}

func TestVisionInjectAllowed(t *testing.T) {
	cases := []struct {
		body   string
		format visionBodyFormat
		want   bool
	}{
		{`{}`, visionFormatChat, true},
		{`{"response_format":{"type":"text"},"tool_choice":"auto"}`, visionFormatChat, true},
		{`{"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"auto"}}}`, visionFormatChat, true},
		{`{"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"required"}}}`, visionFormatChat, false},
		{`{"text":{"format":{"type":"json_schema"}}}`, visionFormatResponses, false},
		{`{"text":{"format":{"type":"text"}},"tool_choice":{"type":"allowed_tools","mode":"auto"}}`, visionFormatResponses, true},
		{`{"tool_choice":{"type":"function","name":"f"}}`, visionFormatResponses, false},
		{`{"tool_choice":{"type":"auto"}}`, visionFormatMessages, true},
		{`{"tool_choice":{"type":"any"}}`, visionFormatMessages, false},
		{`{"output_config":{"format":{"type":"json_schema"}}}`, visionFormatMessages, false},
	}
	for _, tc := range cases {
		root, err := decodeVisionBody([]byte(tc.body))
		require.NoError(t, err)
		assert.Equal(t, tc.want, visionInjectAllowed(root, tc.format), tc.body)
	}
}

// A streamed answer made only of tool calls still gets the descriptions (as content),
// so the next turn can restore them.
func TestVisionStreamInjector_ToolCallsOnly(t *testing.T) {
	const upstream = `data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}

data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	inj := &visionResponseInjection{prefix: visionTestMarker, choices: 1}
	out, err := io.ReadAll(newVisionStreamInjector(io.NopCloser(strings.NewReader(upstream)), inj, false))
	require.NoError(t, err)
	events := sseEvents(t, string(out))
	require.Len(t, events, 3)
	delta := events[0]["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	assert.Equal(t, visionTestMarker, delta["content"])
	assert.Equal(t, "assistant", delta["role"], "the first chunk of a choice carries the role")
	assert.Equal(t, "c", events[0]["id"])
	assert.True(t, strings.HasSuffix(string(out), "data: [DONE]\n\n"))
}

func TestInjectVisionIntoResponse_ResponsesToolCallsOnly(t *testing.T) {
	inj := &visionResponseInjection{prefix: visionTestMarker, choices: 1}
	out, ok := injectVisionIntoResponse([]byte(`{"output":[{"type":"reasoning"},{"type":"function_call","name":"f"}]}`), inj, true)
	require.True(t, ok)
	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output := resp["output"].([]any)
	require.Len(t, output, 3)
	assert.Equal(t, "reasoning", output[0].(map[string]any)["type"])
	assert.Equal(t, "message", output[1].(map[string]any)["type"], "a message item is added after the reasoning")
	assert.Equal(t, strings.TrimRight(visionTestMarker, "\n"), responsesOutputText(t, resp))

	_, ok = injectVisionIntoResponse([]byte(`not json`), inj, false)
	assert.False(t, ok)
}

func TestVisionFallbackConfig_InjectDefault(t *testing.T) {
	assert.True(t, (&config.VisionFallbackConfig{}).InjectEnabled())
}

// passthrough_messages: true on a vLLM model: the upstream answers in Anthropic format,
// which is not written into. The image is still described.
func TestVisionInject_MessagesPassthroughSkipped(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
	w := sendVisionRequest(t, prx, "/v1/messages", `{"model":"glm-msgpass","max_tokens":100,"messages":[
		{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "described=1/1", w.Header().Get(HeaderVisionFallback))
	assert.NotContains(t, w.Body.String(), "air-vision")
	paths, bodies := u.snapshot()
	require.Len(t, bodies, 2)
	assert.Equal(t, "/v1/messages", paths[1])
	assert.Contains(t, mustJSON(t, bodies[1]["messages"]), visionTestOwl)
}

// A streamed Responses answer made only of tool calls gets a message item with the
// descriptions before the final event, numbered in sequence.
func TestVisionStreamInjector_ResponsesToolCallsOnly(t *testing.T) {
	const upstream = `event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":""}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":"{}"}}

event: response.completed
data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}

`
	inj := &visionResponseInjection{prefix: visionTestMarker, choices: 1}
	out, err := io.ReadAll(newVisionStreamInjector(io.NopCloser(strings.NewReader(upstream)), inj, true))
	require.NoError(t, err)

	var types []string
	var seqs []float64
	for _, event := range sseEvents(t, string(out)) {
		types = append(types, event["type"].(string))
		seqs = append(seqs, event["sequence_number"].(float64))
	}
	assert.Equal(t, []string{
		"response.output_item.added", "response.output_item.done",
		"response.output_item.added", "response.content_part.added", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done", "response.output_item.done",
		"response.completed",
	}, types)
	assert.Equal(t, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, seqs)
	assert.Equal(t, 9, strings.Count(string(out), "event: "), "every data line keeps its event line")

	events := sseEvents(t, string(out))
	assert.Equal(t, strings.TrimRight(visionTestMarker, "\n"), events[4]["delta"])
	assert.EqualValues(t, 1, events[4]["output_index"], "appended after the function call")
	completed := events[8]["response"].(map[string]any)
	output := completed["output"].([]any)
	require.Len(t, output, 2)
	assert.Equal(t, "function_call", output[0].(map[string]any)["type"])
	assert.Equal(t, strings.TrimRight(visionTestMarker, "\n"), responsesOutputText(t, completed))
	assert.NotNil(t, completed["usage"], "usage is untouched")

	// The next turn restores the description from that item.
	answers := visionTurnAnswers([]any{
		map[string]any{"role": "user", "content": []any{}},
		output[0], output[1],
	}, visionFormatResponses)
	assert.Equal(t, visionTestOwl, answers[0][1].text)
}

func TestVisionChatChunkMayStartAnswer(t *testing.T) {
	cases := map[string]bool{
		`{"choices":[{"delta":{"role":"assistant","content":""}}]}`:                  false,
		`{"choices":[{"delta":{"reasoning_content":"думаю"},"finish_reason":null}]}`: false,
		`{"choices":[{"delta":{"content":null,"reasoning_content":"x"}}]}`:           false,
		`{"choices":[],"usage":{"prompt_tokens":1}}`:                                 false,
		`{"choices":[{"delta":{"content":"Это"}}]}`:                                  true,
		`{"choices":[{"delta":{"content": "с пробелом"}}]}`:                          true,
		`{"choices":[{"delta":{},"finish_reason":"stop"}]}`:                          true,
		`{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`:                       true,
	}
	for payload, want := range cases {
		assert.Equal(t, want, visionChatChunkMayStartAnswer([]byte(payload)), payload)
	}
}
