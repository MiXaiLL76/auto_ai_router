package vertex_transform

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestTransformVertexStreamToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		model    string
		expected []string
		wantErr  bool
	}{
		{
			name: "basic streaming response",
			input: `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}]}

data: [DONE]

`,
			model: "gemini-2.5-pro",
			expected: []string{
				"Hello",
				" world",
				"!",
			},
			wantErr: false,
		},
		{
			name: "empty content",
			input: `data: {"candidates":[{"content":{"parts":[],"role":"model"}}]}

data: [DONE]

`,
			model: "gemini-2.5-pro",
			expected: []string{
				"",
			},
			wantErr: false,
		},
		{
			name: "malformed json",
			input: `data: {"invalid": json}

data: {"candidates":[{"content":{"parts":[{"text":"Valid"}],"role":"model"}}]}

data: [DONE]

`,
			model: "gemini-2.5-pro",
			expected: []string{
				"Valid",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := strings.NewReader(tt.input)
			var output bytes.Buffer

			err := TransformVertexStreamToOpenAI(input, tt.model, &output)

			if tt.wantErr {
				if err == nil {
					t.Errorf("TransformVertexStreamToOpenAI() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("TransformVertexStreamToOpenAI() error = %v", err)
				return
			}

			result := output.String()

			// Check that output contains expected content
			for _, expectedContent := range tt.expected {
				if !strings.Contains(result, expectedContent) {
					t.Errorf("Expected output to contain %q, got: %s", expectedContent, result)
				}
			}

			// Check that output contains OpenAI format markers
			if !strings.Contains(result, "chat.completion.chunk") {
				t.Errorf("Expected output to contain 'chat.completion.chunk'")
			}

			if !strings.Contains(result, tt.model) {
				t.Errorf("Expected output to contain model name %q", tt.model)
			}

			// Check that output ends with [DONE]
			if !strings.Contains(result, "data: [DONE]") {
				t.Errorf("Expected output to end with '[DONE]'")
			}
		})
	}
}

func TestTransformVertexStreamToOpenAI_EmptyInput(t *testing.T) {
	input := strings.NewReader("")
	var output bytes.Buffer

	err := TransformVertexStreamToOpenAI(input, "test-model", &output)

	if err != nil {
		t.Errorf("TransformVertexStreamToOpenAI() with empty input error = %v", err)
	}

	result := output.String()
	if result != "" {
		t.Errorf("Expected empty output for empty input, got: %s", result)
	}
}

func TestTransformVertexStreamToOpenAI_NonDataLines(t *testing.T) {
	input := strings.NewReader(`event: start

: comment line

data: {"candidates":[{"content":{"parts":[{"text":"Test"}],"role":"model"}}]}

event: end

data: [DONE]

`)
	var output bytes.Buffer

	err := TransformVertexStreamToOpenAI(input, "test-model", &output)

	if err != nil {
		t.Errorf("TransformVertexStreamToOpenAI() error = %v", err)
	}

	result := output.String()

	// Should contain the valid data line
	if !strings.Contains(result, "Test") {
		t.Errorf("Expected output to contain 'Test', got: %s", result)
	}

	// Should not contain event lines or comments
	if strings.Contains(result, "event:") {
		t.Errorf("Output should not contain event lines")
	}

	if strings.Contains(result, ": comment") {
		t.Errorf("Output should not contain comment lines")
	}
}

func TestTransformVertexStreamToOpenAI_RoleInFirstChunk(t *testing.T) {
	input := strings.NewReader(`data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"}}]}

data: [DONE]

`)
	var output bytes.Buffer

	err := TransformVertexStreamToOpenAI(input, "test-model", &output)

	if err != nil {
		t.Errorf("TransformVertexStreamToOpenAI() error = %v", err)
	}

	result := output.String()

	// Parse each chunk to verify role handling
	lines := strings.Split(result, "\n")
	chunkCount := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "data: {") {
			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
				continue
			}

			if chunkCount == 0 {
				// First chunk should have role
				if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if delta, ok := choice["delta"].(map[string]interface{}); ok {
							if role, ok := delta["role"].(string); !ok || role != "assistant" {
								t.Errorf("First chunk should have role='assistant', got: %v", delta["role"])
							}
						}
					}
				}
			} else if chunkCount < 2 {
				// Second chunk should not have role
				if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if delta, ok := choice["delta"].(map[string]interface{}); ok {
							if role, ok := delta["role"]; ok && role != "" {
								t.Errorf("Subsequent chunks should not have role, got: %v", role)
							}
						}
					}
				}
			}
			chunkCount++
		}
	}

	if chunkCount < 2 {
		t.Errorf("Expected at least 2 data chunks, got %d", chunkCount)
	}
}

func TestTransformVertexStreamToOpenAI_EmptyCandidates(t *testing.T) {
	input := strings.NewReader(`data: {"candidates":[]}

data: {"candidates":[{"content":{"parts":[{"text":"Valid"}],"role":"model"}}]}

data: [DONE]

`)
	var output bytes.Buffer

	err := TransformVertexStreamToOpenAI(input, "test-model", &output)

	if err != nil {
		t.Errorf("TransformVertexStreamToOpenAI() error = %v", err)
	}

	result := output.String()

	// Should skip empty candidates and only contain valid data
	if !strings.Contains(result, "Valid") {
		t.Errorf("Expected output to contain 'Valid', got: %s", result)
	}

	// Count data chunks - should skip the empty candidates chunk
	count := strings.Count(result, "data: {")
	if count < 1 {
		t.Errorf("Expected at least 1 valid data chunk, got %d", count)
	}
}
