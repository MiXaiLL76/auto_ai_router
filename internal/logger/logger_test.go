package logger

import (
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew_InfoLevel(t *testing.T) {
	logger := New("info")
	assert.NotNil(t, logger)
}

func TestNew_DebugLevel(t *testing.T) {
	logger := New("debug")
	assert.NotNil(t, logger)
}

func TestNew_ErrorLevel(t *testing.T) {
	logger := New("error")
	assert.NotNil(t, logger)
}

func TestNew_DefaultLevel(t *testing.T) {
	logger := New("unknown")
	assert.NotNil(t, logger)
}

func TestNewJSON(t *testing.T) {
	logger := NewJSON("info")
	assert.NotNil(t, logger)
}

func TestTruncateLongFields_InvalidJSON(t *testing.T) {
	body := "not valid json"
	result := TruncateLongFields(body, 100)
	assert.Equal(t, body, result)
}

func TestTruncateLongFields_EmbeddingField(t *testing.T) {
	longEmbedding := strings.Repeat("x", 200)
	input := `{"embedding":"` + longEmbedding + `"}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	embedding := data["embedding"].(string)
	assert.True(t, strings.Contains(embedding, "truncated"))
	assert.True(t, len(embedding) < len(longEmbedding))
}

func TestTruncateLongFields_B64JSONField(t *testing.T) {
	longB64 := strings.Repeat("a", 150)
	input := `{"b64_json":"` + longB64 + `"}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	b64 := data["b64_json"].(string)
	assert.True(t, strings.Contains(b64, "truncated"))
}

func TestTruncateLongFields_ContentField(t *testing.T) {
	longContent := strings.Repeat("x", 200)
	input := `{"content":"` + longContent + `"}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	content := data["content"].(string)
	assert.True(t, strings.Contains(content, "truncated"))
}

func TestTruncateLongFields_ShortContent(t *testing.T) {
	input := `{"content":"short content"}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	content := data["content"].(string)
	assert.Equal(t, "short content", content)
}

func TestTruncateLongFields_RegularStringField(t *testing.T) {
	longString := strings.Repeat("y", 150)
	input := `{"message":"` + longString + `"}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	message := data["message"].(string)
	assert.True(t, strings.Contains(message, "truncated"))
}

func TestTruncateLongFields_MessagesArray(t *testing.T) {
	input := `{
		"messages": [
			{"role":"user","content":"` + strings.Repeat("x", 100) + `"},
			{"role":"assistant","content":"` + strings.Repeat("y", 100) + `"}
		]
	}`

	result := TruncateLongFields(input, 50)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	messages := data["messages"].([]interface{})
	assert.Len(t, messages, 2)

	msg1 := messages[0].(map[string]interface{})
	content1 := msg1["content"].(string)
	assert.True(t, strings.Contains(content1, "truncated"))
}

func TestTruncateLongFields_NestedFields(t *testing.T) {
	input := `{
		"level1": {
			"level2": {
				"field":"` + strings.Repeat("x", 150) + `"
			}
		}
	}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	level1 := data["level1"].(map[string]interface{})
	level2 := level1["level2"].(map[string]interface{})
	field := level2["field"].(string)
	assert.True(t, strings.Contains(field, "truncated"))
}

func TestTruncateLongFields_MultipleFields(t *testing.T) {
	input := `{
		"id":"short",
		"embedding":"` + strings.Repeat("e", 100) + `",
		"b64_json":"` + strings.Repeat("b", 100) + `",
		"content":"` + strings.Repeat("c", 100) + `"
	}`

	result := TruncateLongFields(input, 50)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	assert.Equal(t, "short", data["id"].(string))
	assert.True(t, strings.Contains(data["embedding"].(string), "truncated"))
	assert.True(t, strings.Contains(data["b64_json"].(string), "truncated"))
	assert.True(t, strings.Contains(data["content"].(string), "truncated"))
}

func TestTruncateLongFields_EmptyJSON(t *testing.T) {
	input := `{}`
	result := TruncateLongFields(input, 100)
	assert.Equal(t, `{}`, result)
}

func TestTruncateLongFields_JSONArray(t *testing.T) {
	input := `[
		{"content":"` + strings.Repeat("x", 100) + `"},
		{"content":"` + strings.Repeat("y", 100) + `"}
	]`

	result := TruncateLongFields(input, 50)

	// JSON arrays are not directly supported as top-level (Unmarshal into map[string]interface{} won't work)
	// So it should return the original
	assert.Equal(t, input, result)
}

func TestTruncateLongFields_MarshalError(t *testing.T) {
	input := `{"valid":"json"}`
	result := TruncateLongFields(input, 100)
	var data map[string]interface{}
	assert.NoError(t, json.Unmarshal([]byte(result), &data))
}

func TestTruncateLongFields_SpecificTruncationLength(t *testing.T) {
	input := `{"field":"` + strings.Repeat("x", 200) + `"}`

	result1 := TruncateLongFields(input, 50)
	result2 := TruncateLongFields(input, 100)

	var data1, data2 map[string]interface{}
	_ = json.Unmarshal([]byte(result1), &data1)
	_ = json.Unmarshal([]byte(result2), &data2)

	field1 := data1["field"].(string)
	field2 := data2["field"].(string)

	assert.True(t, strings.Contains(field1, "truncated"))
	assert.True(t, strings.Contains(field2, "truncated"))
	assert.Less(t, len(field1), len(field2))
}

func TestParseLevel_CaseInsensitive(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected slog.Level
	}{
		{"lowercase debug", "debug", slog.LevelDebug},
		{"uppercase DEBUG", "DEBUG", slog.LevelDebug},
		{"mixed cAsE", "DeBuG", slog.LevelDebug},
		{"lowercase info", "info", slog.LevelInfo},
		{"uppercase INFO", "INFO", slog.LevelInfo},
		{"lowercase error", "error", slog.LevelError},
		{"uppercase ERROR", "ERROR", slog.LevelError},
		{"unknown", "unknown", slog.LevelInfo},
		{"empty", "", slog.LevelInfo},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			level := parseLevel(tt.input)
			assert.Equal(t, tt.expected, level)
		})
	}
}

func TestTruncateLongFields_EmbeddingLessThan50(t *testing.T) {
	input := `{"embedding":"` + strings.Repeat("x", 60) + `"}`

	result := TruncateLongFields(input, 100)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	embedding := data["embedding"].(string)
	assert.True(t, strings.Contains(embedding, "truncated"))
}

func TestTruncateLongFields_ComplexStructure(t *testing.T) {
	input := `{
		"request": {
			"model":"gpt-4",
			"messages":[
				{
					"role":"user",
					"content":"` + strings.Repeat("x", 100) + `"
				}
			]
		},
		"response":{
			"embedding":"` + strings.Repeat("e", 100) + `"
		}
	}`

	result := TruncateLongFields(input, 50)

	var data map[string]interface{}
	_ = json.Unmarshal([]byte(result), &data)

	assert.NotNil(t, data["request"])
	assert.NotNil(t, data["response"])
	assert.True(t, strings.Contains(result, "truncated"))
}
