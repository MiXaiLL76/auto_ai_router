package converter

import (
	"strings"
	"testing"
	"time"
)

func TestGenerateID_FormatAndUniqueness(t *testing.T) {
	id1 := GenerateID()
	id2 := GenerateID()

	if !strings.HasPrefix(id1, "chatcmpl-") {
		t.Fatalf("expected prefix chatcmpl-, got %q", id1)
	}
	if len(id1) != len("chatcmpl-")+20 {
		t.Fatalf("unexpected id length: %d", len(id1))
	}
	if id1 == id2 {
		t.Fatalf("expected ids to differ, got %q", id1)
	}
}

func TestGetCurrentTimestamp_CloseToNow(t *testing.T) {
	start := time.Now().UTC().Unix()
	got := GetCurrentTimestamp()
	end := time.Now().UTC().Unix()

	if got < start-2 || got > end+2 {
		t.Fatalf("timestamp out of range: got %d, window [%d, %d]", got, start-2, end+2)
	}
}

func TestGetString(t *testing.T) {
	m := map[string]interface{}{
		"ok":   "value",
		"num":  123,
		"bool": true,
	}

	if got := GetString(m, "ok"); got != "value" {
		t.Fatalf("expected value, got %q", got)
	}
	if got := GetString(m, "num"); got != "" {
		t.Fatalf("expected empty string for non-string value, got %q", got)
	}
	if got := GetString(m, "missing"); got != "" {
		t.Fatalf("expected empty string for missing key, got %q", got)
	}
}

func TestExtractTextBlocks(t *testing.T) {
	if got := ExtractTextBlocks(""); got != nil {
		t.Fatalf("expected nil for empty string, got %v", got)
	}

	if got := ExtractTextBlocks("hello"); len(got) != 1 || got[0] != "hello" {
		t.Fatalf("unexpected result for string: %v", got)
	}

	content := []interface{}{
		map[string]interface{}{"type": "text", "text": "a"},
		map[string]interface{}{"type": "image", "text": "ignored"},
		map[string]interface{}{"type": "text", "text": ""},
		"not-a-map",
		map[string]interface{}{"type": "text", "text": "b"},
	}

	got := ExtractTextBlocks(content)
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("unexpected text blocks: %v", got)
	}

	if got := ExtractTextBlocks(123); got != nil {
		t.Fatalf("expected nil for unsupported type, got %v", got)
	}
}
