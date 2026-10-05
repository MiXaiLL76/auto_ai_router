package responses

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

func TestChatRequestToResponses_BasicTextMessage(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":100,"messages":[{"role":"user","content":"What is 2+2?"}]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))

	assert.Equal(t, "gpt-5-pro", raw["model"])
	assert.Equal(t, float64(100), raw["max_output_tokens"])
	assert.NotContains(t, raw, "max_tokens")
	assert.NotContains(t, raw, "messages")

	input, ok := raw["input"].([]interface{})
	require.True(t, ok)
	require.Len(t, input, 1)
	item := input[0].(map[string]interface{})
	assert.Equal(t, "user", item["role"])
	assert.Equal(t, "What is 2+2?", item["content"])
}

func TestChatRequestToResponses_SystemMessage(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"system","content":"Be terse."},
		{"role":"user","content":"hi"}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	require.Len(t, input, 2)
	assert.Equal(t, "system", input[0].(map[string]interface{})["role"])
	assert.Equal(t, "Be terse.", input[0].(map[string]interface{})["content"])
	assert.Equal(t, "user", input[1].(map[string]interface{})["role"])
}

func TestChatRequestToResponses_ImageContent(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"user","content":[
			{"type":"text","text":"describe this"},
			{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"high"}}
		]}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	require.Len(t, input, 1)
	parts := input[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, parts, 2)
	assert.Equal(t, "input_text", parts[0].(map[string]interface{})["type"])
	imgPart := parts[1].(map[string]interface{})
	assert.Equal(t, "input_image", imgPart["type"])
	assert.Equal(t, "https://example.com/a.png", imgPart["image_url"])
	assert.Equal(t, "high", imgPart["detail"])
}

// TestChatRequestToResponses_FileContent covers a client-supplied "file" content part
// (e.g. a PDF attachment) surviving the conversion to Responses API's "input_file", both
// for the inline file_data form and the file_id form. Before this fix, "file" fell into
// chatContentPartsToInput's default case and was silently dropped -- the attachment
// vanished with no client-visible error, and the model answered as though it had never
// been sent.
func TestChatRequestToResponses_FileContent(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"user","content":[
			{"type":"text","text":"summarize this"},
			{"type":"file","file":{"file_data":"data:application/pdf;base64,AAAA","filename":"doc.pdf"}}
		]}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	require.Len(t, input, 1)
	parts := input[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, parts, 2, "the file part must survive, not be silently dropped")
	filePart := parts[1].(map[string]interface{})
	assert.Equal(t, "input_file", filePart["type"])
	assert.Equal(t, "data:application/pdf;base64,AAAA", filePart["file_data"])
	assert.Equal(t, "doc.pdf", filePart["filename"])
	assert.NotContains(t, filePart, "file_id")
}

func TestChatRequestToResponses_FileContentByID(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"user","content":[
			{"type":"file","file":{"file_id":"file-abc123"}}
		]}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	parts := input[0].(map[string]interface{})["content"].([]interface{})
	require.Len(t, parts, 1)
	filePart := parts[0].(map[string]interface{})
	assert.Equal(t, "input_file", filePart["type"])
	assert.Equal(t, "file-abc123", filePart["file_id"])
	assert.NotContains(t, filePart, "file_data")
}

// TestChatRequestToResponses_ToolUseRoundTrip covers the peculiarity that
// matters most for multi-turn agentic conversations: Responses API has no
// "tool_calls" field on a message and no "tool" role -- an assistant tool
// call and its result must become standalone function_call /
// function_call_output items, correlated by call_id, which is exactly the ID
// the client already knows as tool_call_id.
func TestChatRequestToResponses_ToolUseRoundTrip(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"user","content":"what's the weather in paris?"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"paris\"}"}}
		]},
		{"role":"tool","tool_call_id":"call_abc123","content":"18C, cloudy"}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	require.Len(t, input, 3)

	assert.Equal(t, "user", input[0].(map[string]interface{})["role"])

	fc := input[1].(map[string]interface{})
	assert.Equal(t, "function_call", fc["type"])
	assert.Equal(t, "call_abc123", fc["call_id"])
	assert.Equal(t, "get_weather", fc["name"])
	assert.Equal(t, `{"city":"paris"}`, fc["arguments"])

	fco := input[2].(map[string]interface{})
	assert.Equal(t, "function_call_output", fco["type"])
	assert.Equal(t, "call_abc123", fco["call_id"])
	assert.Equal(t, "18C, cloudy", fco["output"])
}

// TestChatRequestToResponses_ToolResultContentParts covers a tool-role message whose
// content is the content-parts array shape ([{"type":"text","text":"..."}]) rather than
// a plain string -- both are valid Chat Completions shapes. Before this fix, the array
// fell into chatToolContentToString's default case and got JSON-marshaled whole, so the
// model received the literal JSON source (e.g. [{"type":"text","text":"18C, cloudy"}])
// as the tool result instead of the text itself.
func TestChatRequestToResponses_ToolResultContentParts(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"user","content":"what's the weather in paris?"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"paris\"}"}}
		]},
		{"role":"tool","tool_call_id":"call_abc123","content":[{"type":"text","text":"18C, cloudy"}]}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	require.Len(t, input, 3)
	fco := input[2].(map[string]interface{})
	assert.Equal(t, "function_call_output", fco["type"])
	assert.Equal(t, "18C, cloudy", fco["output"], "the text must be extracted, not the JSON source of the content-parts array")
}

// TestChatRequestToResponses_ToolResultMultiPartContent covers multiple text parts in a
// tool result -- they must be concatenated in order, same interpretation as content
// parts on a user/assistant message.
func TestChatRequestToResponses_ToolResultMultiPartContent(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[
		{"role":"user","content":"?"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}
		]},
		{"role":"tool","tool_call_id":"call_1","content":[
			{"type":"text","text":"part one. "},
			{"type":"text","text":"part two."}
		]}
	]}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	input := raw["input"].([]interface{})
	fco := input[2].(map[string]interface{})
	assert.Equal(t, "part one. part two.", fco["output"])
}

func TestChatRequestToResponses_ToolsFlattening(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_tokens":50,"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object"}}}],
		"tool_choice":{"type":"function","function":{"name":"get_weather"}}
	}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))

	tools := raw["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "function", tool["type"])
	assert.Equal(t, "get_weather", tool["name"])
	assert.Equal(t, "d", tool["description"])
	assert.NotContains(t, tool, "function")

	tc := raw["tool_choice"].(map[string]interface{})
	assert.Equal(t, "function", tc["type"])
	assert.Equal(t, "get_weather", tc["name"])
	assert.NotContains(t, tc, "function")
}

func TestChatRequestToResponses_ReasoningEffortAndResponseFormat(t *testing.T) {
	body := `{"model":"gpt-5-pro","max_completion_tokens":50,"messages":[{"role":"user","content":"hi"}],
		"reasoning_effort":"high",
		"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"},"strict":true}}
	}`

	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))

	assert.Equal(t, float64(50), raw["max_output_tokens"])
	assert.NotContains(t, raw, "max_completion_tokens")
	assert.NotContains(t, raw, "reasoning_effort")
	assert.NotContains(t, raw, "response_format")

	reasoning := raw["reasoning"].(map[string]interface{})
	assert.Equal(t, "high", reasoning["effort"])

	text := raw["text"].(map[string]interface{})
	format := text["format"].(map[string]interface{})
	assert.Equal(t, "json_schema", format["type"])
	assert.Equal(t, "x", format["name"])
	assert.Equal(t, true, format["strict"])
}

func TestChatRequestToResponses_MissingMessages(t *testing.T) {
	_, err := ChatRequestToResponses([]byte(`{"model":"gpt-5-pro"}`))
	assert.Error(t, err)
}

func TestChatRequestToResponses_UnsupportedRole(t *testing.T) {
	_, err := ChatRequestToResponses([]byte(`{"model":"gpt-5-pro","messages":[{"role":"bogus","content":"x"}]}`))
	assert.Error(t, err)
}

// TestChatRequestToResponses_InputAudioRejected covers a critical review
// finding: Chat Completions' {"type":"input_audio",...} content part has no
// faithful Responses API equivalent (the Responses input content union has
// no audio variant at all), so silently forwarding a guessed wire shape the
// provider would 400 on (or worse, silently ignore) is wrong -- the request
// must be rejected with a clear client-facing error instead.
func TestChatRequestToResponses_InputAudioRejected(t *testing.T) {
	body := `{"model":"gpt-5-pro","messages":[{"role":"user","content":[
		{"type":"text","text":"transcribe this"},
		{"type":"input_audio","input_audio":{"data":"BASE64","format":"wav"}}
	]}]}`
	_, err := ChatRequestToResponses([]byte(body))
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	assert.True(t, errors.As(err, &validationErr), "must be a RequestValidationError so the client gets a 4xx, not a 500")
}

func TestChatRequestToResponses_DeletesChatOnlyFields(t *testing.T) {
	body := `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
		"stop":["\n"],"frequency_penalty":0.5,"presence_penalty":0.5,"logit_bias":{"123":-100},
		"n":2,"stream_options":{"include_usage":true}
	}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	for _, key := range []string{"stop", "frequency_penalty", "presence_penalty", "logit_bias", "n", "stream_options"} {
		assert.NotContains(t, raw, key)
	}
}

func TestChatRequestToResponses_ForcesStatelessReasoning(t *testing.T) {
	body := `{"model":"o3-pro","messages":[{"role":"user","content":"hi"}]}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.Equal(t, false, raw["store"])
	assert.Equal(t, []interface{}{"reasoning.encrypted_content"}, raw["include"])
}

func TestChatRequestToResponses_ForcesStatelessReasoning_KeepsExistingInclude(t *testing.T) {
	body := `{"model":"o3-pro","messages":[{"role":"user","content":"hi"}],"include":["file_search_call.results"]}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.ElementsMatch(t, []interface{}{"file_search_call.results", "reasoning.encrypted_content"}, raw["include"])
}

// TestChatRequestToResponses_NonReasoningModelSkipsEncryptedContentInclude covers review
// finding #4: "include":["reasoning.encrypted_content"] was added unconditionally
// regardless of the model, and at least one other OpenAI-API client project
// independently reports some non-reasoning models reject that include value outright --
// which would 400 every single request to a non-reasoning responses_only model. store
// must still be pinned false (needed for every model this feature serves, reasoning or
// not), but "include" must stay absent for a model outside every known reasoning family
// with no explicit reasoning request either.
func TestChatRequestToResponses_NonReasoningModelSkipsEncryptedContentInclude(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.Equal(t, false, raw["store"], "store:false is needed for every model this feature serves")
	assert.NotContains(t, raw, "include", "a non-reasoning model must not get reasoning.encrypted_content added")
}

// TestChatRequestToResponses_ExplicitReasoningEffortForcesInclude covers a model
// outside every family openai.IsReasoningModel recognizes, that the client
// nonetheless explicitly asked for reasoning on (e.g. a reasoning-capable model this
// family list hasn't been updated for yet) -- the explicit ask alone must be enough to
// still request encrypted_content, since ResponseToChat/chatThinkingBlocksToReasoningItems
// need it for continuity regardless of whether the model name is recognized.
func TestChatRequestToResponses_ExplicitReasoningEffortForcesInclude(t *testing.T) {
	body := `{"model":"some-future-reasoning-model","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.Equal(t, []interface{}{"reasoning.encrypted_content"}, raw["include"])
}

func TestChatRequestToResponses_DropsUnsupportedChatOnlyFields(t *testing.T) {
	body := `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
		"seed":42,"logprobs":true,"top_logprobs":3,"modalities":["text","audio"],
		"audio":{"voice":"alloy","format":"wav"},"prediction":{"type":"content","content":"foo"},
		"web_search_options":{"search_context_size":"high"}
	}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	for _, key := range []string{"seed", "logprobs", "modalities", "audio", "prediction", "web_search_options"} {
		assert.NotContains(t, raw, key)
	}
	// top_logprobs is a valid Responses API field under the same name, unlike the others.
	assert.Equal(t, float64(3), raw["top_logprobs"])
}

func TestChatRequestToResponses_VerbosityMovesUnderText(t *testing.T) {
	body := `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],"verbosity":"low"}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.NotContains(t, raw, "verbosity")
	text, ok := raw["text"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "low", text["verbosity"])
}

func TestChatRequestToResponses_VerbosityMergesWithResponseFormat(t *testing.T) {
	body := `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
		"verbosity":"high","response_format":{"type":"json_object"}
	}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	text, ok := raw["text"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "high", text["verbosity"])
	format, ok := text["format"].(map[string]interface{})
	require.True(t, ok, "verbosity must not clobber the format text.format already carries")
	assert.Equal(t, "json_object", format["type"])
}

func TestChatRequestToResponses_CustomToolRejected(t *testing.T) {
	cases := map[string]string{
		"nested custom tool": `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"custom","custom":{"name":"code_exec"}}]}`,
		"flat custom tool": `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"custom","name":"code_exec"}]}`,
		"custom tool_choice": `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"function","function":{"name":"f"}}],
			"tool_choice":{"type":"custom","custom":{"name":"code_exec"}}}`,
		"custom tool inside allowed_tools": `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"function","function":{"name":"f"}}],
			"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[{"type":"custom","custom":{"name":"x"}}]}}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ChatRequestToResponses([]byte(body))
			var verr *converterutil.RequestValidationError
			require.ErrorAs(t, err, &verr)
		})
	}
}

func TestChatRequestToResponses_AllowedToolsChoiceFlattened(t *testing.T) {
	body := `{"model":"gpt-5-pro","messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}},{"type":"function","function":{"name":"get_time","parameters":{}}}],
		"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[
			{"type":"function","function":{"name":"get_weather"}},
			{"type":"image_generation"}
		]}}}`
	out, err := ChatRequestToResponses([]byte(body))
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.Equal(t, map[string]interface{}{
		"type": "allowed_tools",
		"mode": "required",
		"tools": []interface{}{
			map[string]interface{}{"type": "function", "name": "get_weather"},
			map[string]interface{}{"type": "image_generation"},
		},
	}, raw["tool_choice"])
}
