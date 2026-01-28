package vertex_transform

import (
	"encoding/json"
	"testing"
)

func TestOpenAIToVertex(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "basic chat completion",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [
					{"role": "system", "content": "You are helpful"},
					{"role": "user", "content": "Hello"}
				],
				"temperature": 0.7,
				"max_tokens": 100
			}`,
			expected: map[string]interface{}{
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "Hello"},
						},
					},
				},
				"systemInstruction": map[string]interface{}{
					"parts": []interface{}{
						map[string]interface{}{"text": "You are helpful"},
					},
				},
				"generationConfig": map[string]interface{}{
					"temperature":     0.7,
					"maxOutputTokens": float64(100),
				},
			},
			wantErr: false,
		},
		{
			name: "with stop sequences",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Test"}],
				"stop": ["END", "STOP"]
			}`,
			expected: map[string]interface{}{
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "Test"},
						},
					},
				},
				"generationConfig": map[string]interface{}{
					"stopSequences": []interface{}{"END", "STOP"},
				},
			},
			wantErr: false,
		},
		{
			name:    "invalid json",
			input:   `{"invalid": json}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := OpenAIToVertex([]byte(tt.input))

			if tt.wantErr {
				if err == nil {
					t.Errorf("OpenAIToVertex() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("OpenAIToVertex() error = %v", err)
				return
			}

			var resultMap map[string]interface{}
			if err := json.Unmarshal(result, &resultMap); err != nil {
				t.Errorf("Failed to unmarshal result: %v", err)
				return
			}

			// Check key fields exist
			if tt.expected["contents"] != nil {
				if _, ok := resultMap["contents"]; !ok {
					t.Errorf("Expected 'contents' field in result")
				}
			}

			if tt.expected["systemInstruction"] != nil {
				if _, ok := resultMap["systemInstruction"]; !ok {
					t.Errorf("Expected 'systemInstruction' field in result")
				}
			}
		})
	}
}

func TestVertexToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		model    string
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "basic vertex response",
			input: `{
				"candidates": [{
					"content": {
						"parts": [{"text": "Hello there!"}],
						"role": "model"
					},
					"finishReason": "STOP"
				}],
				"usageMetadata": {
					"promptTokenCount": 5,
					"candidatesTokenCount": 3,
					"totalTokenCount": 8
				}
			}`,
			model: "gemini-2.5-pro",
			expected: map[string]interface{}{
				"object": "chat.completion",
				"model":  "gemini-2.5-pro",
				"choices": []interface{}{
					map[string]interface{}{
						"index": float64(0),
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": "Hello there!",
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     float64(5),
					"completion_tokens": float64(3),
					"total_tokens":      float64(8),
				},
			},
			wantErr: false,
		},
		{
			name:    "invalid json",
			input:   `{"invalid": json}`,
			model:   "test",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := VertexToOpenAI([]byte(tt.input), tt.model)

			if tt.wantErr {
				if err == nil {
					t.Errorf("VertexToOpenAI() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("VertexToOpenAI() error = %v", err)
				return
			}

			var resultMap map[string]interface{}
			if err := json.Unmarshal(result, &resultMap); err != nil {
				t.Errorf("Failed to unmarshal result: %v", err)
				return
			}

			// Check required fields
			if resultMap["object"] != tt.expected["object"] {
				t.Errorf("Expected object %v, got %v", tt.expected["object"], resultMap["object"])
			}

			if resultMap["model"] != tt.expected["model"] {
				t.Errorf("Expected model %v, got %v", tt.expected["model"], resultMap["model"])
			}

			if _, ok := resultMap["choices"]; !ok {
				t.Errorf("Expected 'choices' field in result")
			}
		})
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
		{"RECITATION", "content_filter"},
		{"UNKNOWN", "stop"},
		{"", "stop"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := mapFinishReason(tt.input)
			if result != tt.expected {
				t.Errorf("mapFinishReason(%s) = %s, expected %s", tt.input, result, tt.expected)
			}
		})
	}
}

func TestGenerateID(t *testing.T) {
	id1 := generateID()
	id2 := generateID()

	// Check format
	if len(id1) != 29 { // "chatcmpl-" + 20 hex chars
		t.Errorf("Expected ID length 29, got %d", len(id1))
	}

	if id1[:9] != "chatcmpl-" {
		t.Errorf("Expected ID to start with 'chatcmpl-', got %s", id1[:9])
	}

	// Check uniqueness
	if id1 == id2 {
		t.Errorf("Expected unique IDs, got same: %s", id1)
	}
}
