package vertex

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
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
			name: "with all parameters",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Test"}],
				"temperature": 0.8,
				"max_tokens": 200,
				"top_p": 0.9,
				"n": 2,
				"seed": 42,
				"frequency_penalty": 0.1,
				"presence_penalty": 0.2,
				"stop": ["END", "STOP"]
			}`,
			expected: map[string]interface{}{
				"generationConfig": map[string]interface{}{
					"temperature":      0.8,
					"maxOutputTokens":  float64(200),
					"topP":             0.9,
					"candidateCount":   float64(2),
					"seed":             float64(42),
					"frequencyPenalty": 0.1,
					"presencePenalty":  0.2,
					"stopSequences":    []interface{}{"END", "STOP"},
				},
			},
			wantErr: false,
		},
		{
			name: "with extra_body",
			input: `{
				"model": "gemini-2.5-flash-image",
				"messages": [{"role": "user", "content": "Generate image"}],
				"extra_body": {
					"modalities": ["image"],
					"generation_config": {
						"top_k": 40,
						"seed": 123,
						"temperature": 0.4
					}
				}
			}`,
			expected: map[string]interface{}{
				"generationConfig": map[string]interface{}{
					"responseModalities": []interface{}{"IMAGE"},
					"topK":               float64(40),
					"seed":               float64(123),
					"temperature":        0.4,
				},
			},
			wantErr: false,
		},
		{
			name: "multimodal content",
			input: `{
				"model": "gemini-2.5-flash-image",
				"messages": [{
					"role": "user",
					"content": [
						{"type": "text", "text": "Describe this image"},
						{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}}
					]
				}]
			}`,
			expected: map[string]interface{}{
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "Describe this image"},
							map[string]interface{}{
								"inlineData": map[string]interface{}{
									"mimeType": "image/png",
									"data":     "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "assistant role mapping",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [
					{"role": "user", "content": "Hello"},
					{"role": "assistant", "content": "Hi there!"}
				]
			}`,
			expected: map[string]interface{}{
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "Hello"},
						},
					},
					map[string]interface{}{
						"role": "model",
						"parts": []interface{}{
							map[string]interface{}{"text": "Hi there!"},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "developer role as system instruction",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [
					{"role": "developer", "content": "You are a helpful assistant"},
					{"role": "user", "content": "Hello"}
				]
			}`,
			expected: map[string]interface{}{
				"systemInstruction": map[string]interface{}{
					"parts": []interface{}{
						map[string]interface{}{"text": "You are a helpful assistant"},
					},
				},
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "Hello"},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "single stop string",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Test"}],
				"stop": "END"
			}`,
			expected: map[string]interface{}{
				"generationConfig": map[string]interface{}{
					"stopSequences": []interface{}{"END"},
				},
			},
			wantErr: false,
		},
		{
			name:    "invalid json",
			input:   `{"invalid": json}`,
			wantErr: true,
		},
		{
			name: "assistant with tool_calls",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [
					{"role": "user", "content": "Call a function"},
					{
						"role": "assistant",
						"content": null,
						"tool_calls": [
							{
								"id": "call_abc123",
								"type": "function",
								"function": {
									"name": "test_function",
									"arguments": "{\"param1\": 123, \"param2\": \"value\"}"
								}
							}
						]
					}
				]
			}`,
			expected: map[string]interface{}{
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "Call a function"},
						},
					},
					map[string]interface{}{
						"role": "model",
						"parts": []interface{}{
							map[string]interface{}{
								"functionCall": map[string]interface{}{
									"name": "test_function",
									"args": map[string]interface{}{
										"param1": float64(123),
										"param2": "value",
									},
								},
							},
						},
					},
				},
			},
			wantErr: false,
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

			if tt.expected["generationConfig"] != nil {
				if _, ok := resultMap["generationConfig"]; !ok {
					t.Errorf("Expected 'generationConfig' field in result")
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
			},
			wantErr: false,
		},
		{
			name: "response with images",
			input: `{
				"candidates": [{
					"content": {
						"parts": [
							{"text": "Here's your image:"},
							{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}
						],
						"role": "model"
					},
					"finishReason": "STOP"
				}]
			}`,
			model: "gemini-2.5-flash-image",
			expected: map[string]interface{}{
				"object": "chat.completion",
				"model":  "gemini-2.5-flash-image",
			},
			wantErr: false,
		},
		{
			name: "empty parts with finish reason",
			input: `{
				"candidates": [{
					"content": {
						"parts": [],
						"role": "model"
					},
					"finishReason": "MAX_TOKENS"
				}]
			}`,
			model: "gemini-2.5-pro",
			expected: map[string]interface{}{
				"object": "chat.completion",
				"model":  "gemini-2.5-pro",
			},
			wantErr: false,
		},
		{
			name: "multiple candidates",
			input: `{
				"candidates": [
					{
						"content": {
							"parts": [{"text": "Response 1"}],
							"role": "model"
						},
						"finishReason": "STOP"
					},
					{
						"content": {
							"parts": [{"text": "Response 2"}],
							"role": "model"
						},
						"finishReason": "STOP"
					}
				]
			}`,
			model: "gemini-2.5-pro",
			expected: map[string]interface{}{
				"object": "chat.completion",
				"model":  "gemini-2.5-pro",
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

			// Check choices structure
			choices, ok := resultMap["choices"].([]interface{})
			if !ok {
				t.Errorf("Expected 'choices' to be array")
				return
			}

			if len(choices) == 0 {
				t.Errorf("Expected at least one choice")
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

func TestExtractTextContent(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "string content",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name: "array with text block",
			input: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Hello from array",
				},
			},
			expected: "Hello from array",
		},
		{
			name: "array with mixed blocks",
			input: []interface{}{
				map[string]interface{}{
					"type":      "image_url",
					"image_url": map[string]interface{}{"url": "data:image/png;base64,abc"},
				},
				map[string]interface{}{
					"type": "text",
					"text": "Found text",
				},
			},
			expected: "Found text",
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: "",
		},
		{
			name:     "nil input",
			input:    nil,
			expected: "",
		},
		{
			name:     "number input",
			input:    123,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractTextContent(tt.input)
			if result != tt.expected {
				t.Errorf("extractTextContent() = %q, expected %q", result, tt.expected)
			}
		})
	}
}

func TestConvertContentToParts(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected int // number of parts
	}{
		{
			name:     "string content",
			input:    "Hello world",
			expected: 1,
		},
		{
			name: "text block",
			input: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Hello",
				},
			},
			expected: 1,
		},
		{
			name: "image block",
			input: []interface{}{
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
					},
				},
			},
			expected: 1,
		},
		{
			name: "mixed content",
			input: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Describe this:",
				},
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
					},
				},
			},
			expected: 2,
		},
		{
			name: "invalid image url",
			input: []interface{}{
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "https://example.com/image.png", // not data URL
					},
				},
			},
			expected: 0, // should be ignored
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: 0,
		},
		{
			name:     "number fallback",
			input:    123,
			expected: 1, // fallback to string conversion
		},
		{
			name: "data url without base64 encoding",
			input: []interface{}{
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "data:image/png;,iVBORw0KGgoAAAA", // no base64 prefix
					},
				},
			},
			expected: 1, // should handle gracefully
		},
		{
			name: "data url without semicolon",
			input: []interface{}{
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "data:image/jpeg,/9j/4AAQSkZJRg==", // no semicolon before comma
					},
				},
			},
			expected: 1,
		},
		{
			name: "data url with different mime types",
			input: []interface{}{
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "data:image/webp;base64,UklGRiYA",
					},
				},
			},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertContentToParts(tt.input)
			if len(result) != tt.expected {
				t.Errorf("convertContentToParts() returned %d parts, expected %d", len(result), tt.expected)
			}

			// Additional checks for specific cases
			if tt.name == "image block" && len(result) > 0 {
				if result[0].InlineData == nil {
					t.Errorf("Expected InlineData to be set for image block")
				} else {
					if result[0].InlineData.MimeType != "image/png" {
						t.Errorf("Expected mime type 'image/png', got %s", result[0].InlineData.MimeType)
					}
				}
			}

			if tt.name == "string content" && len(result) > 0 {
				if result[0].Text != "Hello world" {
					t.Errorf("Expected text 'Hello world', got %s", result[0].Text)
				}
			}

			if tt.name == "data url with different mime types" && len(result) > 0 {
				if result[0].InlineData.MimeType != "image/webp" {
					t.Errorf("Expected mime type 'image/webp', got %s", result[0].InlineData.MimeType)
				}
			}

			if tt.name == "data url without semicolon" && len(result) > 0 {
				if result[0].InlineData.MimeType != "image/jpeg" {
					t.Errorf("Expected mime type 'image/jpeg', got %s", result[0].InlineData.MimeType)
				}
			}
		})
	}
}

func TestVertexToOpenAIWithMixedContent(t *testing.T) {
	input := `{
		"candidates": [{
			"content": {
				"parts": [
					{"text": "Here is text:"},
					{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
					{"text": "And more text"}
				],
				"role": "model"
			},
			"finishReason": "STOP"
		}]
	}`

	result, err := VertexToOpenAI([]byte(input), "test-model")
	if err != nil {
		t.Errorf("VertexToOpenAI() error = %v", err)
		return
	}

	var respMap map[string]interface{}
	if err := json.Unmarshal(result, &respMap); err != nil {
		t.Errorf("Failed to unmarshal result: %v", err)
		return
	}

	// Check that text content is concatenated
	if choices, ok := respMap["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				content := msg["content"].(string)
				if !strings.Contains(content, "Here is text:") || !strings.Contains(content, "And more text") {
					t.Errorf("Expected concatenated text, got: %s", content)
				}
			}
		}
	}

	// Check that images are captured
	if choices, ok := respMap["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				if images, ok := msg["images"].([]interface{}); ok {
					if len(images) != 1 {
						t.Errorf("Expected 1 image, got %d", len(images))
					}
				}
			}
		}
	}
}

func TestDetermineVertexPublisher(t *testing.T) {
	tests := []struct {
		name     string
		modelID  string
		expected string
	}{
		{
			name:     "claude model lowercase",
			modelID:  "claude-3-opus",
			expected: "anthropic",
		},
		{
			name:     "claude model uppercase",
			modelID:  "CLAUDE-3-SONNET",
			expected: "anthropic",
		},
		{
			name:     "claude model mixed case",
			modelID:  "Claude-3-Haiku",
			expected: "anthropic",
		},
		{
			name:     "gemini model",
			modelID:  "gemini-pro",
			expected: "google",
		},
		{
			name:     "other model",
			modelID:  "gpt-4",
			expected: "google",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			publisher := determineVertexPublisher(tt.modelID)
			assert.Equal(t, tt.expected, publisher)
		})
	}
}

func TestBuildVertexURL(t *testing.T) {
	tests := []struct {
		name      string
		cred      *config.CredentialConfig
		modelID   string
		streaming bool
		expected  string
	}{
		{
			name: "global location non-streaming claude",
			cred: &config.CredentialConfig{
				ProjectID: "test-project",
				Location:  "global",
			},
			modelID:   "claude-3-opus",
			streaming: false,
			expected:  "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/anthropic/models/claude-3-opus:generateContent",
		},
		{
			name: "global location streaming claude",
			cred: &config.CredentialConfig{
				ProjectID: "test-project",
				Location:  "global",
			},
			modelID:   "claude-3-sonnet",
			streaming: true,
			expected:  "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/anthropic/models/claude-3-sonnet:streamGenerateContent?alt=sse",
		},
		{
			name: "regional location non-streaming gemini",
			cred: &config.CredentialConfig{
				ProjectID: "test-project",
				Location:  "us-central1",
			},
			modelID:   "gemini-pro",
			streaming: false,
			expected:  "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:generateContent",
		},
		{
			name: "regional location streaming gemini",
			cred: &config.CredentialConfig{
				ProjectID: "test-project",
				Location:  "europe-west1",
			},
			modelID:   "gemini-pro",
			streaming: true,
			expected:  "https://europe-west1-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west1/publishers/google/models/gemini-pro:streamGenerateContent?alt=sse",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := BuildVertexURL(tt.cred, tt.modelID, tt.streaming)
			assert.Equal(t, tt.expected, url)
		})
	}
}

func TestBuildVertexImageURL(t *testing.T) {
	tests := []struct {
		name     string
		cred     *config.CredentialConfig
		modelID  string
		expected string
	}{
		{
			name: "global location",
			cred: &config.CredentialConfig{
				ProjectID: "test-project",
				Location:  "global",
			},
			modelID:  "imagegeneration",
			expected: "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/imagegeneration:predict",
		},
		{
			name: "regional location",
			cred: &config.CredentialConfig{
				ProjectID: "test-project",
				Location:  "us-central1",
			},
			modelID:  "imagen-3.0-generate-001",
			expected: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/imagen-3.0-generate-001:predict",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := BuildVertexImageURL(tt.cred, tt.modelID)
			assert.Equal(t, tt.expected, url)
		})
	}
}
