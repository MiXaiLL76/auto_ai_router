package openai

import (
	"encoding/json"
	"testing"
)

// helper: unmarshal body into map for assertions
func bodyToMap(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("failed to unmarshal body: %v", err)
	}
	return m
}

func makeBody(t *testing.T, fields map[string]any) []byte {
	t.Helper()
	b, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("failed to marshal body: %v", err)
	}
	return b
}

// --- matchModelFamily tests ---

func TestMatchModelFamily(t *testing.T) {
	tests := []struct {
		modelID string
		family  string
		want    bool
	}{
		// Exact match
		{"o1", "o1", true},
		{"o3", "o3", true},
		{"gpt-5", "gpt-5", true},

		// Dash-separated variants
		{"o1-mini", "o1", true},
		{"o1-preview", "o1", true},
		{"o1-preview-2024-12-17", "o1", true},
		{"o1-pro", "o1", true},
		{"o3-mini", "o3", true},
		{"o3-pro", "o3", true},
		{"gpt-5-mini", "gpt-5", true},
		{"gpt-5-nano", "gpt-5", true},

		// Dot-separated variants
		{"gpt-5.1", "gpt-5", true},
		{"gpt-5.2", "gpt-5", true},

		// Provider prefix with slash
		{"openai/gpt-5", "gpt-5", true},
		{"openai/o1-mini", "o1", true},
		{"openai/o3-pro", "o3", true},
		{"vertex/gpt-5-mini", "gpt-5", true},

		// Provider prefix with colon
		{"openai:gpt-5", "gpt-5", true},
		{"openai:o3-mini", "o3", true},

		// Suffix _chat / -chat
		{"gpt-5_chat", "gpt-5", true},
		{"gpt-5-chat", "gpt-5", true},
		{"o1_chat", "o1", true},
		{"o3-mini_chat", "o3", true},

		// Combined prefix + suffix
		{"openai/gpt-5_chat", "gpt-5", true},
		{"openai:o1-mini_chat", "o1", true},

		// Case insensitive
		{"GPT-5", "gpt-5", true},
		{"O1-Mini", "o1", true},
		{"OpenAI/GPT-5", "gpt-5", true},

		// Should NOT match
		{"o1", "o3", false},
		{"o3", "o1", false},
		{"gpt-4o", "gpt-5", false},
		{"o10-something", "o1", false},      // "o10" != "o1-..."
		{"not-o1-model", "o1", false},       // doesn't start with "o1"
		{"my-gpt-5-custom", "gpt-5", false}, // doesn't start with "gpt-5"
		{"gpt-50", "gpt-5", false},          // "gpt-50" != "gpt-5-..."
		{"openai/gpt-4o", "gpt-5", false},
	}

	for _, tt := range tests {
		t.Run(tt.modelID+"_in_"+tt.family, func(t *testing.T) {
			got := matchModelFamily(tt.modelID, tt.family)
			if got != tt.want {
				t.Errorf("matchModelFamily(%q, %q) = %v, want %v", tt.modelID, tt.family, got, tt.want)
			}
		})
	}
}

// --- UpdateJSONField tests ---

func TestUpdateJSONField_RenameKey(t *testing.T) {
	body := makeBody(t, map[string]any{
		"model":      "test",
		"max_tokens": 100,
	})

	mapping := ModelParamsMapping{
		KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"},
	}

	result := bodyToMap(t, UpdateJSONField(body, mapping))

	if _, ok := result["max_tokens"]; ok {
		t.Error("max_tokens should have been removed")
	}
	if v, ok := result["max_completion_tokens"]; !ok || v != float64(100) {
		t.Errorf("max_completion_tokens = %v, want 100", v)
	}
}

func TestUpdateJSONField_NoOverwriteExistingKey(t *testing.T) {
	// If both max_tokens and max_completion_tokens are present,
	// the explicit max_completion_tokens should be preserved.
	body := makeBody(t, map[string]any{
		"model":                 "test",
		"max_tokens":            100,
		"max_completion_tokens": 200,
	})

	mapping := ModelParamsMapping{
		KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"},
	}

	result := bodyToMap(t, UpdateJSONField(body, mapping))

	if _, ok := result["max_tokens"]; ok {
		t.Error("max_tokens should have been removed")
	}
	if v := result["max_completion_tokens"]; v != float64(200) {
		t.Errorf("max_completion_tokens = %v, want 200 (should keep explicit value)", v)
	}
}

func TestUpdateJSONField_RemoveKeys(t *testing.T) {
	body := makeBody(t, map[string]any{
		"model":       "test",
		"temperature": 0.7,
		"top_p":       0.9,
		"messages":    []any{},
	})

	mapping := ModelParamsMapping{
		KeysToRemove: []string{"temperature", "top_p"},
	}

	result := bodyToMap(t, UpdateJSONField(body, mapping))

	if _, ok := result["temperature"]; ok {
		t.Error("temperature should have been removed")
	}
	if _, ok := result["top_p"]; ok {
		t.Error("top_p should have been removed")
	}
	if _, ok := result["messages"]; !ok {
		t.Error("messages should be preserved")
	}
}

func TestUpdateJSONField_InvalidJSON(t *testing.T) {
	body := []byte(`not valid json`)
	mapping := ModelParamsMapping{
		KeysToRemove: []string{"temperature"},
	}

	result := UpdateJSONField(body, mapping)
	if string(result) != string(body) {
		t.Error("invalid JSON should return original body unchanged")
	}
}

func TestUpdateJSONField_NoMatchingKeys(t *testing.T) {
	body := makeBody(t, map[string]any{
		"model":    "test",
		"messages": []any{},
	})

	mapping := ModelParamsMapping{
		KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"},
		KeysToRemove:  []string{"temperature", "top_p"},
	}

	result := bodyToMap(t, UpdateJSONField(body, mapping))

	if result["model"] != "test" {
		t.Error("model should be unchanged")
	}
}

// --- ReplaceBodyParam integration tests ---

func TestReplaceBodyParam_O1(t *testing.T) {
	models := []string{"o1", "o1-mini", "o1-preview", "o1-preview-2024-12-17", "o1-pro"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := makeBody(t, map[string]any{
				"model":             model,
				"max_tokens":        1000,
				"temperature":       0.7,
				"top_p":             0.9,
				"frequency_penalty": 0.5,
				"presence_penalty":  0.5,
				"logprobs":          true,
				"top_logprobs":      5,
				"messages":          []any{},
			})

			result := bodyToMap(t, ReplaceBodyParam(model, body))

			// Should be renamed
			if _, ok := result["max_tokens"]; ok {
				t.Error("max_tokens should be renamed to max_completion_tokens")
			}
			if v := result["max_completion_tokens"]; v != float64(1000) {
				t.Errorf("max_completion_tokens = %v, want 1000", v)
			}

			// Should be removed
			for _, key := range []string{"temperature", "top_p", "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs"} {
				if _, ok := result[key]; ok {
					t.Errorf("%s should be removed for o1 models", key)
				}
			}

			// Should be preserved
			if _, ok := result["messages"]; !ok {
				t.Error("messages should be preserved")
			}
		})
	}
}

func TestReplaceBodyParam_O3(t *testing.T) {
	models := []string{"o3", "o3-mini", "o3-pro"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := makeBody(t, map[string]any{
				"model":             model,
				"max_tokens":        1000,
				"temperature":       0.7,
				"top_p":             0.9,
				"frequency_penalty": 0.5,
				"reasoning_effort":  "high",
				"messages":          []any{},
			})

			result := bodyToMap(t, ReplaceBodyParam(model, body))

			// Should be renamed
			if _, ok := result["max_tokens"]; ok {
				t.Error("max_tokens should be renamed")
			}
			if v := result["max_completion_tokens"]; v != float64(1000) {
				t.Errorf("max_completion_tokens = %v, want 1000", v)
			}

			// Should be removed
			if _, ok := result["temperature"]; ok {
				t.Error("temperature should be removed for o3")
			}
			if _, ok := result["top_p"]; ok {
				t.Error("top_p should be removed for o3")
			}

			// Should be preserved (o3 supports these unlike o1)
			if _, ok := result["frequency_penalty"]; !ok {
				t.Error("frequency_penalty should be preserved for o3")
			}
			if _, ok := result["reasoning_effort"]; !ok {
				t.Error("reasoning_effort should be preserved for o3")
			}
		})
	}
}

func TestReplaceBodyParam_GPT5(t *testing.T) {
	models := []string{"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := makeBody(t, map[string]any{
				"model":       model,
				"max_tokens":  500,
				"temperature": 0.5,
				"top_p":       0.8,
				"messages":    []any{},
			})

			result := bodyToMap(t, ReplaceBodyParam(model, body))

			if _, ok := result["max_tokens"]; ok {
				t.Error("max_tokens should be renamed")
			}
			if v := result["max_completion_tokens"]; v != float64(500) {
				t.Errorf("max_completion_tokens = %v, want 500", v)
			}
			if _, ok := result["temperature"]; ok {
				t.Error("temperature should be removed for gpt-5")
			}
			if _, ok := result["top_p"]; ok {
				t.Error("top_p should be removed for gpt-5")
			}
		})
	}
}

func TestReplaceBodyParam_UnknownModel_NoChanges(t *testing.T) {
	models := []string{"gpt-4o", "gpt-4o-mini", "claude-opus-4-1", "gemini-2.5-flash", "deepseek-v3"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := makeBody(t, map[string]any{
				"model":       model,
				"max_tokens":  1000,
				"temperature": 0.7,
				"top_p":       0.9,
				"messages":    []any{},
			})

			result := bodyToMap(t, ReplaceBodyParam(model, body))

			// Nothing should be changed
			if _, ok := result["max_tokens"]; !ok {
				t.Error("max_tokens should be preserved for regular models")
			}
			if _, ok := result["temperature"]; !ok {
				t.Error("temperature should be preserved for regular models")
			}
			if _, ok := result["top_p"]; !ok {
				t.Error("top_p should be preserved for regular models")
			}
		})
	}
}

func TestReplaceBodyParam_FalsePositives(t *testing.T) {
	// These model names should NOT match any family
	models := []string{"o10-large", "gpt-50-turbo", "not-o1-model", "my-gpt-5-custom", "o3000"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := makeBody(t, map[string]any{
				"model":       model,
				"max_tokens":  1000,
				"temperature": 0.7,
			})

			result := bodyToMap(t, ReplaceBodyParam(model, body))

			if _, ok := result["max_tokens"]; !ok {
				t.Errorf("max_tokens should be preserved for %q (should not match any family)", model)
			}
			if _, ok := result["temperature"]; !ok {
				t.Errorf("temperature should be preserved for %q", model)
			}
		})
	}
}

func TestReplaceBodyParam_MaxCompletionTokensAlreadyPresent(t *testing.T) {
	// Client explicitly sets max_completion_tokens AND max_tokens.
	// max_completion_tokens should NOT be overwritten.
	body := makeBody(t, map[string]any{
		"model":                 "o3-mini",
		"max_tokens":            100,
		"max_completion_tokens": 500,
		"messages":              []any{},
	})

	result := bodyToMap(t, ReplaceBodyParam("o3-mini", body))

	if _, ok := result["max_tokens"]; ok {
		t.Error("max_tokens should be removed")
	}
	if v := result["max_completion_tokens"]; v != float64(500) {
		t.Errorf("max_completion_tokens = %v, want 500 (explicit value preserved)", v)
	}
}

// --- extractBaseModelName tests ---

func TestExtractBaseModelName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		// No transformation needed
		{"gpt-5", "gpt-5"},
		{"o1-mini", "o1-mini"},
		{"o3", "o3"},

		// Slash prefix
		{"openai/gpt-5", "gpt-5"},
		{"openai/o1-mini", "o1-mini"},
		{"vertex/o3-pro", "o3-pro"},
		{"provider/sub/gpt-5", "gpt-5"}, // nested slashes

		// Colon prefix
		{"openai:gpt-5", "gpt-5"},
		{"openai:o3-mini", "o3-mini"},

		// Suffix _chat / -chat
		{"gpt-5_chat", "gpt-5"},
		{"gpt-5-chat", "gpt-5"},
		{"o1_chat", "o1"},
		{"o3-mini_chat", "o3-mini"},

		// Combined
		{"openai/gpt-5_chat", "gpt-5"},
		{"openai:o1-mini_chat", "o1-mini"},

		// Case insensitive
		{"GPT-5", "gpt-5"},
		{"O1-Mini", "o1-mini"},
		{"OpenAI/GPT-5", "gpt-5"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := extractBaseModelName(tt.input)
			if got != tt.want {
				t.Errorf("extractBaseModelName(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// --- ReplaceBodyParam with provider prefixes ---

func TestReplaceBodyParam_WithProviderPrefixes(t *testing.T) {
	// Models with provider prefixes should still get parameter transformations
	tests := []struct {
		modelID      string
		shouldRemove []string
		shouldRename bool
	}{
		{"openai/o1-mini", []string{"temperature", "top_p", "frequency_penalty", "presence_penalty", "logprobs"}, true},
		{"openai:o3-mini", []string{"temperature", "top_p"}, true},
		{"openai/gpt-5_chat", []string{"temperature", "top_p"}, true},
		{"vertex/gpt-5.1", []string{"temperature", "top_p"}, true},
		{"OpenAI/O1-Preview", []string{"temperature", "top_p", "frequency_penalty", "presence_penalty", "logprobs"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			body := makeBody(t, map[string]any{
				"model":             tt.modelID,
				"max_tokens":        1000,
				"temperature":       0.7,
				"top_p":             0.9,
				"frequency_penalty": 0.5,
				"presence_penalty":  0.5,
				"logprobs":          true,
				"messages":          []any{},
			})

			result := bodyToMap(t, ReplaceBodyParam(tt.modelID, body))

			if tt.shouldRename {
				if _, ok := result["max_tokens"]; ok {
					t.Error("max_tokens should be renamed to max_completion_tokens")
				}
				if v := result["max_completion_tokens"]; v != float64(1000) {
					t.Errorf("max_completion_tokens = %v, want 1000", v)
				}
			}

			for _, key := range tt.shouldRemove {
				if _, ok := result[key]; ok {
					t.Errorf("%s should be removed for %s", key, tt.modelID)
				}
			}
		})
	}
}

// --- ReplaceModelInBody tests ---

func TestReplaceModelInBody(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		oldModel  string
		newModel  string
		wantModel string
	}{
		{
			name:      "simple replacement",
			body:      `{"model":"alias","messages":[]}`,
			oldModel:  "alias",
			newModel:  "gpt-4o",
			wantModel: "gpt-4o",
		},
		{
			name:      "with space after colon",
			body:      `{"model": "alias","messages":[]}`,
			oldModel:  "alias",
			newModel:  "gpt-4o",
			wantModel: "gpt-4o",
		},
		{
			name:      "model with special chars",
			body:      `{"model":"my/custom-model:v1","messages":[]}`,
			oldModel:  "my/custom-model:v1",
			newModel:  "gpt-5",
			wantModel: "gpt-5",
		},
		{
			name:      "no match returns unchanged",
			body:      `{"model":"gpt-4o","messages":[]}`,
			oldModel:  "nonexistent",
			newModel:  "gpt-5",
			wantModel: "gpt-4o",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ReplaceModelInBody([]byte(tt.body), tt.oldModel, tt.newModel)
			m := bodyToMap(t, result)
			if m["model"] != tt.wantModel {
				t.Errorf("model = %v, want %v", m["model"], tt.wantModel)
			}
		})
	}
}
