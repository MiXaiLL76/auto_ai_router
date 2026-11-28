package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

// CountCacheMarkers no longer decodes the body; countCacheMarkersFullDecode
// (the previous implementation, kept in cache_markers.go) is what it must
// keep agreeing with.

var cacheMarkerEquivalenceBodies = []string{
	``, `null`, `[]`, `{`, `{"messages":[}`, `{} trailing`,
	`{}`,
	`{"messages":[{"role":"user","content":"hi"}]}`,
	`{"messages":[{"role":"user","content":"the text cache_control is not a key"}]}`,
	`{"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]}]}`,
	`{"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`,
	`{"messages":[{"content":[{"cache_control":{"type":"ephemeral","ttl":"5m"}},{"cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`,
	`{"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],"messages":[],"tools":[{"name":"t","cache_control":{"type":"ephemeral","ttl":"1h"}}]}`,
	`{"system":"plain","instructions":[{"cache_control":{"type":"ephemeral"}}]}`,
	`{"system":null,"instructions":[{"cache_control":{"type":"ephemeral"}}]}`,
	`{"instructions":[{"cache_control":{"type":"ephemeral"}}],"input":[{"content":[{"cache_control":{"type":"ephemeral"}}]}]}`,
	`{"messages":null,"input":[{"cache_control":{"type":"ephemeral"}}]}`,
	`{"messages":[],"input":[{"cache_control":{"type":"ephemeral"}}]}`,
	// Not markers: wrong type, not an object, a parameter that happens to share the name.
	`{"messages":[{"cache_control":{"type":"persistent"}},{"cache_control":"ephemeral"},{"cache_control":null},{"cache_control":[{"type":"ephemeral"}]}]}`,
	`{"tools":[{"function":{"parameters":{"properties":{"cache_control":{"type":"string"}}}}}]}`,
	`{"tools":[{"function":{"parameters":{"properties":{"cache_control":{"type":"ephemeral"}}}}}]}`,
	// Nothing inside a marker is itself a marker.
	`{"messages":[{"cache_control":{"type":"ephemeral","cache_control":{"type":"ephemeral"},"x":[{"cache_control":{"type":"ephemeral"}}]}}]}`,
	// Markers outside the three sections are not counted.
	`{"metadata":{"cache_control":{"type":"ephemeral"}},"cache_control":{"type":"ephemeral"},"messages":[]}`,
	// Escaped spellings of the key and of its values.
	`{"messages":[{"cache_control":{"type":"ephemeral"}}]}`,
	`{"messages":[{"cache_control":{"type":"ephemeral","ttl":"1h"}}]}`,
	`{"messages":[{"cache_control":{"type":"ephemeral"}}]}`,
	// Duplicate keys: a decode keeps the last one.
	`{"messages":[{"cache_control":{"type":"ephemeral"},"cache_control":{"type":"other"}}]}`,
	`{"messages":[{"cache_control":{"type":"other"},"cache_control":{"type":"ephemeral","ttl":"1h"}}]}`,
	`{"messages":[{"a":{"cache_control":{"type":"ephemeral"}},"a":{"x":1}}]}`,
	`{"messages":[{"cache_control":{"type":"ephemeral"}}],"messages":[]}`,
	`{"messages":[],"messages":[{"cache_control":{"type":"ephemeral"}}]}`,
	// Sections of the wrong shape.
	`{"messages":"cache_control","system":7,"tools":true}`,
	`{"messages":{"cache_control":{"type":"ephemeral"},"nested":{"cache_control":{"type":"ephemeral"}}}}`,
	// Numbers a decode rejects: in a section, in a marker, elsewhere.
	`{"messages":[{"n":1e999,"cache_control":{"type":"ephemeral"}}]}`,
	`{"messages":[{"cache_control":{"type":"ephemeral","n":1e999}}]}`,
	`{"messages":[{"n":1e308,"m":-0.0e-400,"cache_control":{"type":"ephemeral"}}]}`,
	"{\"messages\":[{\"text\":\"\xff\xfe\",\"cache_control\":{\"type\":\"ephemeral\"}}]}",
	`{"messages":[{"cache_control":{"type":"ephemeral"}}` + strings.Repeat(`,{"content":"x"}`, 50) + `]}`,
}

func TestCountCacheMarkersMatchesFullDecode(t *testing.T) {
	for _, body := range cacheMarkerEquivalenceBodies {
		got, want := CountCacheMarkers([]byte(body)), countCacheMarkersFullDecode([]byte(body))
		if got != want {
			t.Errorf("CountCacheMarkers(%q) = %+v, full decode says %+v", body, got, want)
		}
	}
}

func TestCountCacheMarkersCountsSomething(t *testing.T) {
	// Guards the equivalence test against both sides being trivially zero.
	body := []byte(`{"system":[{"cache_control":{"type":"ephemeral"}}],"messages":[{"content":[{"cache_control":{"type":"ephemeral","ttl":"1h"}}]}],"tools":[{"cache_control":{"type":"ephemeral"}}]}`)
	want := CacheMarkerCounts{System: 1, Messages: 1, Tools: 1, TTL1h: 1}
	if got := CountCacheMarkers(body); got != want {
		t.Fatalf("CountCacheMarkers = %+v, want %+v", got, want)
	}
}

// The one deliberate difference: a number outside the float64 range in a part
// of the body the counter does not look at. The full decode fails on it and
// reports no markers at all; the walk never reads that number.
func TestCountCacheMarkersOutOfRangeNumberElsewhere(t *testing.T) {
	body := []byte(`{"temperature":1e999,"messages":[{"cache_control":{"type":"ephemeral"}}]}`)
	if got := countCacheMarkersFullDecode(body); got != (CacheMarkerCounts{}) {
		t.Fatalf("reference changed: full decode used to fail on 1e999, got %+v", got)
	}
	if got := CountCacheMarkers(body); got.Messages != 1 {
		t.Fatalf("CountCacheMarkers = %+v, want the marker in messages counted", got)
	}
}

func FuzzCountCacheMarkers(f *testing.F) {
	for _, body := range cacheMarkerEquivalenceBodies {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		want := countCacheMarkersFullDecode(body)
		got := CountCacheMarkers(body)
		if got == want {
			return
		}
		// Differences are allowed only where the full decode refuses a body
		// that is nevertheless a JSON object (an out-of-range number outside
		// the sections walked).
		var asRaw map[string]json.RawMessage
		var full map[string]interface{}
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &full) != nil && want == (CacheMarkerCounts{}) {
			return
		}
		t.Fatalf("CountCacheMarkers(%q) = %+v, full decode says %+v", body, got, want)
	})
}
