package vertex

import (
	"encoding/json"
	"testing"
)

func TestOpenAIImageToVertex(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "basic image generation",
			input: `{
				"model": "imagen-3.0-fast-generate-001",
				"prompt": "A beautiful sunset",
				"size": "1024x1024",
				"n": 1
			}`,
			expected: map[string]interface{}{
				"instances": []interface{}{
					map[string]interface{}{
						"prompt": "A beautiful sunset",
					},
				},
				"parameters": map[string]interface{}{
					"sampleCount":       float64(1),
					"aspectRatio":       "1:1",
					"safetyFilterLevel": "block_some",
					"personGeneration":  "allow_adult",
				},
			},
			wantErr: false,
		},
		{
			name: "landscape aspect ratio",
			input: `{
				"model": "imagen-3.0-fast-generate-001",
				"prompt": "Wide landscape",
				"size": "1792x1024",
				"n": 2
			}`,
			expected: map[string]interface{}{
				"instances": []interface{}{
					map[string]interface{}{
						"prompt": "Wide landscape",
					},
				},
				"parameters": map[string]interface{}{
					"sampleCount":       float64(2),
					"aspectRatio":       "16:9",
					"safetyFilterLevel": "block_some",
					"personGeneration":  "allow_adult",
				},
			},
			wantErr: false,
		},
		{
			name: "portrait aspect ratio",
			input: `{
				"model": "imagen-3.0-fast-generate-001",
				"prompt": "Portrait photo",
				"size": "1024x1792"
			}`,
			expected: map[string]interface{}{
				"parameters": map[string]interface{}{
					"aspectRatio": "9:16",
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
			result, err := OpenAIImageToVertex([]byte(tt.input))

			if tt.wantErr {
				if err == nil {
					t.Errorf("OpenAIImageToVertex() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("OpenAIImageToVertex() error = %v", err)
				return
			}

			var resultMap map[string]interface{}
			if err := json.Unmarshal(result, &resultMap); err != nil {
				t.Errorf("Failed to unmarshal result: %v", err)
				return
			}

			// Check instances exist
			if _, ok := resultMap["instances"]; !ok {
				t.Errorf("Expected 'instances' field in result")
			}

			// Check parameters exist
			if _, ok := resultMap["parameters"]; !ok {
				t.Errorf("Expected 'parameters' field in result")
			}

			// Check specific aspect ratio if provided
			if expectedParams, ok := tt.expected["parameters"].(map[string]interface{}); ok {
				if expectedAspect, ok := expectedParams["aspectRatio"].(string); ok {
					params := resultMap["parameters"].(map[string]interface{})
					if params["aspectRatio"] != expectedAspect {
						t.Errorf("Expected aspectRatio %s, got %s", expectedAspect, params["aspectRatio"])
					}
				}
			}
		})
	}
}

func TestVertexImageToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "basic vertex image response",
			input: `{
				"predictions": [
					{
						"bytesBase64Encoded": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
						"mimeType": "image/png"
					}
				]
			}`,
			expected: map[string]interface{}{
				"data": []interface{}{
					map[string]interface{}{
						"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple images",
			input: `{
				"predictions": [
					{
						"bytesBase64Encoded": "image1data",
						"mimeType": "image/png"
					},
					{
						"bytesBase64Encoded": "image2data",
						"mimeType": "image/png"
					}
				]
			}`,
			expected: map[string]interface{}{
				"data": []interface{}{
					map[string]interface{}{
						"b64_json": "image1data",
					},
					map[string]interface{}{
						"b64_json": "image2data",
					},
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
			result, err := VertexImageToOpenAI([]byte(tt.input))

			if tt.wantErr {
				if err == nil {
					t.Errorf("VertexImageToOpenAI() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("VertexImageToOpenAI() error = %v", err)
				return
			}

			var resultMap map[string]interface{}
			if err := json.Unmarshal(result, &resultMap); err != nil {
				t.Errorf("Failed to unmarshal result: %v", err)
				return
			}

			// Check data field exists
			data, ok := resultMap["data"]
			if !ok {
				t.Errorf("Expected 'data' field in result")
				return
			}

			// Check data is array
			dataArray, ok := data.([]interface{})
			if !ok {
				t.Errorf("Expected 'data' to be array")
				return
			}

			// Check expected number of images
			if expectedData, ok := tt.expected["data"].([]interface{}); ok {
				if len(dataArray) != len(expectedData) {
					t.Errorf("Expected %d images, got %d", len(expectedData), len(dataArray))
				}
			}

			// Check created timestamp exists
			if _, ok := resultMap["created"]; !ok {
				t.Errorf("Expected 'created' field in result")
			}
		})
	}
}

func TestOpenAIImageToVertex_SampleCountLimit(t *testing.T) {
	tests := []struct {
		name          string
		n             int
		expectedCount int
	}{
		{"n below limit", 5, 5},
		{"n at limit", 10, 10},
		{"n exceeds limit", 15, 10},
		{"n way over limit", 100, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := map[string]interface{}{
				"model":  "imagen-3.0-fast-generate-001",
				"prompt": "Test",
				"n":      tt.n,
			}
			inputJSON, _ := json.Marshal(input)
			result, err := OpenAIImageToVertex(inputJSON)

			if err != nil {
				t.Fatalf("OpenAIImageToVertex() error = %v", err)
			}

			var resultMap map[string]interface{}
			if err := json.Unmarshal(result, &resultMap); err != nil {
				t.Fatalf("Failed to unmarshal result: %v", err)
			}
			params := resultMap["parameters"].(map[string]interface{})
			count := int(params["sampleCount"].(float64))

			if count != tt.expectedCount {
				t.Errorf("Expected sampleCount %d, got %d", tt.expectedCount, count)
			}
		})
	}
}

func TestOpenAIImageToVertex_Quality(t *testing.T) {
	tests := []struct {
		name           string
		quality        string
		expectedFilter string
	}{
		{"standard quality", "standard", "block_some"},
		{"hd quality", "hd", "block_few"},
		{"no quality specified", "", "block_some"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := map[string]interface{}{
				"model":  "imagen-3.0-fast-generate-001",
				"prompt": "Test",
			}
			if tt.quality != "" {
				input["quality"] = tt.quality
			}
			inputJSON, _ := json.Marshal(input)
			result, err := OpenAIImageToVertex(inputJSON)

			if err != nil {
				t.Fatalf("OpenAIImageToVertex() error = %v", err)
			}

			var resultMap map[string]interface{}
			if err := json.Unmarshal(result, &resultMap); err != nil {
				t.Fatalf("Failed to unmarshal result: %v", err)
			}
			params := resultMap["parameters"].(map[string]interface{})
			filter := params["safetyFilterLevel"].(string)

			if filter != tt.expectedFilter {
				t.Errorf("Expected safetyFilterLevel %s, got %s", tt.expectedFilter, filter)
			}
		})
	}
}
