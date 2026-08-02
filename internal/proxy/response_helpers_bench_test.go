package proxy

import (
	stdjson "encoding/json"
	"strings"
	"testing"
)

// extractMetadataFromBodyOld reproduces the pre-fix extractMetadataFromBody
// exactly (full map[string]interface{} decode + re-marshal on the streaming
// path), kept only so the benchmarks below can isolate this fix's own
// before/after numbers instead of comparing against a moving target.
func extractMetadataFromBodyOld(body []byte, contentType string) (string, bool, string, []byte) {
	if len(body) == 0 {
		return "", false, "", body
	}

	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		model, sessionID := extractMetadataFromMultipartBody(body, contentType)
		return model, false, sessionID, body
	}

	var reqBody map[string]interface{}
	if err := stdjson.Unmarshal(body, &reqBody); err != nil {
		return "", false, "", body
	}

	model, ok := reqBody["model"].(string)
	if !ok {
		return "", false, "", body
	}

	sessionID := ""
	if extraBody, ok := reqBody["extra_body"].(map[string]interface{}); ok {
		if sid, ok := extraBody["litellm_session_id"].(string); ok && sid != "" {
			sessionID = sid
		} else if cid, ok := extraBody["chat_id"].(string); ok && cid != "" {
			sessionID = cid
		} else if sid, ok := extraBody["session_id"].(string); ok && sid != "" {
			sessionID = sid
		}
	}
	if sessionID == "" {
		if sid, ok := reqBody["session_id"].(string); ok && sid != "" {
			sessionID = sid
		} else if uid, ok := reqBody["user"].(string); ok && uid != "" {
			sessionID = uid
		} else if sid, ok := reqBody["safety_identifier"].(string); ok && sid != "" {
			sessionID = sid
		} else if pck, ok := reqBody["prompt_cache_key"].(string); ok && pck != "" {
			sessionID = pck
		}
	}

	stream, ok := reqBody["stream"].(bool)
	if !ok || !stream {
		return model, false, sessionID, body
	}

	_, hasInput := reqBody["input"]
	_, hasMessages := reqBody["messages"]
	isResponsesAPI := hasInput && !hasMessages

	if !isResponsesAPI {
		streamOptions, exists := reqBody["stream_options"]
		if !exists {
			reqBody["stream_options"] = map[string]interface{}{
				"include_usage": true,
			}
		} else if streamOptionsMap, ok := streamOptions.(map[string]interface{}); ok {
			streamOptionsMap["include_usage"] = true
		} else {
			reqBody["stream_options"] = map[string]interface{}{
				"include_usage": true,
			}
		}
	}

	modifiedBody, err := stdjson.Marshal(reqBody)
	if err != nil {
		return model, stream, sessionID, body
	}

	return model, stream, sessionID, modifiedBody
}

// buildRealisticChatBenchBody builds a chat-completions request body shaped
// like mock-go's production chat profile (~5632 prompt tokens, ~20-25KB) so
// the benchmark reflects the actual dominant cost: decoding/copying the large
// "messages" field, not the handful of scalar fields extractMetadataFromBody
// actually reads.
func buildRealisticChatBenchBody(stream bool) []byte {
	const filler = "the quick brown fox jumps over the lazy dog while pondering distributed systems architecture and load balancing strategies across multiple regions "

	var prompt strings.Builder
	for prompt.Len() < 22000 {
		prompt.WriteString(filler)
	}

	var b strings.Builder
	b.WriteString(`{"model":"gpt-4o","stream":`)
	if stream {
		b.WriteString("true")
	} else {
		b.WriteString("false")
	}
	b.WriteString(`,"user":"user-abc123","extra_body":{"litellm_session_id":"sess-abc-123-def-456"},"messages":[`)
	b.WriteString(`{"role":"system","content":"You are a helpful assistant."},`)
	b.WriteString(`{"role":"user","content":"`)
	b.WriteString(prompt.String())
	b.WriteString(`"}]}`)
	return []byte(b.String())
}

var (
	benchChatBodyStreaming    = buildRealisticChatBenchBody(true)
	benchChatBodyNonStreaming = buildRealisticChatBenchBody(false)
)

// BenchmarkExtractMetadataFromBody_Old: pre-fix baseline, full
// map[string]interface{} decode (+ re-marshal on the streaming branch).
func BenchmarkExtractMetadataFromBody_Old(b *testing.B) {
	b.Run("streaming", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extractMetadataFromBodyOld(benchChatBodyStreaming, "application/json")
		}
	})
	b.Run("non-streaming", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extractMetadataFromBodyOld(benchChatBodyNonStreaming, "application/json")
		}
	})
}

// BenchmarkExtractMetadataFromBody exercises the actual production function
// (response_helpers.go), now on map[string]goccyjson.RawMessage.
func BenchmarkExtractMetadataFromBody(b *testing.B) {
	b.Run("streaming", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, _, _, _, err := extractMetadataFromBody(benchChatBodyStreaming, "application/json"); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("non-streaming", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, _, _, _, err := extractMetadataFromBody(benchChatBodyNonStreaming, "application/json"); err != nil {
				b.Fatal(err)
			}
		}
	})
}
