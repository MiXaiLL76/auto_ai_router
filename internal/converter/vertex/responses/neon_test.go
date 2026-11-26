package vertexresponses

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"github.com/stretchr/testify/require"
)

const neon = "gemini-4-neon"

func TestNeonParallelCallsRoundTrip(t *testing.T) {
	require.True(t, vertex.SupportsLongDecode(neon))

	const raw = `{"modelVersion":"gemini-4-neon","candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[
		{"text":"Checking","thoughtSignature":"dGV4dA=="},
		{"functionCall":{"id":"lookup-a","name":"lookup","args":{"query":"a"}},"thoughtSignature":"c2ln"},
		{"functionCall":{"id":"lookup-b","name":"lookup","args":{"query":"b"}},"thoughtSignature":"c3RhbGU="}
	]}}]}`
	stream := "data: " + strings.ReplaceAll(strings.ReplaceAll(raw, "\t", ""), "\n", "") + "\n\n"
	user := map[string]string{"role": "user", "content": "Look up both entries"}

	checkWire := func(t *testing.T, wire []byte) {
		t.Helper()
		var upstream vertex.VertexRequest
		require.NoError(t, json.Unmarshal(wire, &upstream))
		require.Len(t, upstream.Contents, 3)
		calls, results := upstream.Contents[1], upstream.Contents[2]
		require.Equal(t, "model", calls.Role)
		require.Equal(t, "user", results.Role)
		require.Len(t, calls.Parts, 3)
		require.Len(t, results.Parts, 2)
		require.Equal(t, "Checking", calls.Parts[0].Text)
		require.Equal(t, []byte("text"), calls.Parts[0].ThoughtSignature)
		require.Equal(t, []byte("sig"), calls.Parts[1].ThoughtSignature)
		require.Empty(t, calls.Parts[2].ThoughtSignature)
		for i, id := range []string{"lookup-a", "lookup-b"} {
			require.Equal(t, id, calls.Parts[i+1].FunctionCall.ID)
			require.Equal(t, id, results.Parts[i].FunctionResponse.ID)
			require.Equal(t, "lookup", results.Parts[i].FunctionResponse.Name)
		}
	}
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("chat/stream=%t", streaming), func(t *testing.T) {
			message := neonChatMessage(t, raw, stream, streaming)
			require.Len(t, message.ToolCalls, 2)
			history := []any{user, message}
			for i := len(message.ToolCalls) - 1; i >= 0; i-- {
				history = append(history, map[string]string{
					"role": "tool", "tool_call_id": message.ToolCalls[i].ID, "content": "ok",
				})
			}
			wire, err := geminiWire(t, "chat", neon, map[string]any{"messages": history})
			require.NoError(t, err)
			checkWire(t, wire)
		})
		t.Run(fmt.Sprintf("responses/stream=%t", streaming), func(t *testing.T) {
			var response *responses.Response
			if streaming {
				var out bytes.Buffer
				require.NoError(t, TransformVertexStreamToResponses(strings.NewReader(stream), &out, neon, "", nil, func(r *responses.Response) { response = r }))
			} else {
				var err error
				response, err = VertexToResponsesResponse([]byte(raw), neon, "", 0)
				require.NoError(t, err)
			}
			require.NotNil(t, response)
			history := []any{user}
			var callIDs []string
			for _, item := range response.Output {
				history = append(history, item)
				if item.Type == "function_call" {
					callIDs = append(callIDs, item.CallID)
				}
			}
			require.Len(t, callIDs, 2)
			for i := len(callIDs) - 1; i >= 0; i-- {
				history = append(history, map[string]string{
					"type": "function_call_output", "call_id": callIDs[i], "output": "ok",
				})
			}
			wire, err := geminiWire(t, "responses", neon, map[string]any{"input": history})
			require.NoError(t, err)
			checkWire(t, wire)
		})
	}
}

func neonChatMessage(t *testing.T, raw, stream string, streaming bool) openai.OpenAIResponseMessage {
	t.Helper()
	if !streaming {
		wire, err := vertex.VertexToOpenAI([]byte(raw), neon)
		require.NoError(t, err)
		var response openai.OpenAIResponse
		require.NoError(t, json.Unmarshal(wire, &response))
		require.Len(t, response.Choices, 1)
		return response.Choices[0].Message
	}

	var out bytes.Buffer
	require.NoError(t, vertex.TransformVertexStreamToOpenAI(strings.NewReader(stream), neon, &out))
	message := openai.OpenAIResponseMessage{Role: "assistant"}
	for _, line := range strings.Split(out.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		require.NoError(t, json.Unmarshal([]byte(data), &chunk))
		for _, choice := range chunk.Choices {
			message.Content += choice.Delta.Content
			if fields := choice.Delta.ProviderSpecificFields; fields != nil {
				if message.ProviderSpecificFields == nil {
					message.ProviderSpecificFields = &openai.MessageProviderFields{}
				}
				message.ProviderSpecificFields.ThoughtSignatures = append(message.ProviderSpecificFields.ThoughtSignatures, fields.ThoughtSignatures...)
			}
			for _, call := range choice.Delta.ToolCalls {
				require.NotNil(t, call.Function)
				message.ToolCalls = append(message.ToolCalls, openai.OpenAIToolCall{
					ID: call.ID, Type: call.Type, ProviderSpecificFields: call.ProviderSpecificFields,
					Function: openai.OpenAIToolFunction{Name: call.Function.Name, Arguments: call.Function.Arguments},
				})
			}
		}
	}
	return message
}
