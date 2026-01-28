package vertex_transform

import (
	"bytes"
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
