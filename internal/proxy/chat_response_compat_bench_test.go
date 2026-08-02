package proxy

import (
	"bytes"
	stdjson "encoding/json"
	"errors"
	"io"
	"testing"
)

// decodeJSONObjectStdlib mirrors decodeJSONObject (chat_response_compat.go,
// now on goccy) exactly, on stdlib json, kept only to isolate the engine
// swap's own effect in the benchmarks below.
func decodeJSONObjectStdlib(data []byte) (map[string]interface{}, error) {
	decoder := stdjson.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var object map[string]interface{}
	if err := decoder.Decode(&object); err != nil {
		return nil, err
	}
	var trailing interface{}
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("multiple JSON values")
		}
		return nil, err
	}
	return object, nil
}

// normalizeSSEDataLineModelStdlib mirrors normalizeSSEDataLineModel exactly,
// using decodeJSONObjectStdlib + stdjson.Marshal instead of goccy.
func normalizeSSEDataLineModelStdlib(line []byte, publicModel string) []byte {
	newline := []byte(nil)
	body := line
	switch {
	case bytes.HasSuffix(body, []byte("\r\n")):
		newline = []byte("\r\n")
		body = body[:len(body)-2]
	case bytes.HasSuffix(body, []byte("\n")):
		newline = []byte("\n")
		body = body[:len(body)-1]
	}
	if !bytes.HasPrefix(body, []byte("data:")) {
		return line
	}
	rest := body[len("data:"):]
	payload := bytes.TrimSpace(rest)
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return line
	}
	event, err := decodeJSONObjectStdlib(payload)
	if err != nil || event == nil {
		return line
	}
	rawError, hasError := event["error"]
	eventType, _ := event["type"].(string)
	if isStreamErrorEvent(eventType, hasError && rawError != nil) {
		return line
	}
	changed := false
	if current, exists := event["model"]; exists {
		if current != publicModel {
			event["model"] = publicModel
			changed = true
		}
	} else if object, _ := event["object"].(string); object == "chat.completion.chunk" || object == "text_completion" {
		event["model"] = publicModel
		changed = true
	}
	if response, ok := event["response"].(map[string]interface{}); ok {
		if response["model"] != publicModel {
			response["model"] = publicModel
			changed = true
		}
	}
	if !changed {
		return line
	}
	normalized, err := stdjson.Marshal(event)
	if err != nil {
		return line
	}
	prefixLen := len(rest) - len(bytes.TrimLeft(rest, " \t"))
	result := make([]byte, 0, len(body)+len(newline))
	result = append(result, []byte("data:")...)
	result = append(result, rest[:prefixLen]...)
	result = append(result, normalized...)
	result = append(result, newline...)
	return result
}

var (
	chatChunkSSELine = []byte(`data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730000000,"model":"backend-internal","choices":[{"index":0,"delta":{"content":"hello world"},"finish_reason":null}]}` + "\n")

	responsesEventSSELine = []byte(`data: {"type":"response.output_text.delta","item_id":"item_abc123","output_index":0,"content_index":0,"delta":"hello world","response":{"model":"backend-internal"}}` + "\n")
)

func BenchmarkNormalizeSSEDataLineModel_Stdlib(b *testing.B) {
	b.Run("chat-chunk", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			normalizeSSEDataLineModelStdlib(chatChunkSSELine, "public-model")
		}
	})
	b.Run("responses-event", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			normalizeSSEDataLineModelStdlib(responsesEventSSELine, "public-model")
		}
	})
}

// BenchmarkNormalizeSSEDataLineModel_Goccy exercises the actual production
// function (chat_response_compat.go), now on goccy.
func BenchmarkNormalizeSSEDataLineModel_Goccy(b *testing.B) {
	b.Run("chat-chunk", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			normalizeSSEDataLineModel(chatChunkSSELine, "public-model")
		}
	})
	b.Run("responses-event", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			normalizeSSEDataLineModel(responsesEventSSELine, "public-model")
		}
	})
}
