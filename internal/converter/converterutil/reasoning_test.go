package converterutil

import "testing"

func TestReasoningText(t *testing.T) {
	tests := []struct {
		name      string
		rc, r     interface{}
		wantText  string
		wantField interface{}
	}{
		{"only reasoning_content", "a", nil, "a", "a"},
		{"only reasoning", nil, "b", "b", "b"},
		{"both prefers reasoning_content", "a", "b", "a", "a"},
		{"empty reasoning_content falls back", "", "b", "b", "b"},
		{"empty reasoning_content alone", "", nil, "", ""},
		{"non-string reasoning_content", 42, "b", "", 42},
		{"neither", nil, nil, "", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReasoningText(tt.rc, tt.r); got != tt.wantText {
				t.Errorf("ReasoningText = %q, want %q", got, tt.wantText)
			}
			if got := PickReasoningField(tt.rc, tt.r); got != tt.wantField {
				t.Errorf("PickReasoningField = %v, want %v", got, tt.wantField)
			}
		})
	}
}
