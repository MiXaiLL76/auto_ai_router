package anthropic

import (
	"bytes"
	"encoding/json"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
)

// CacheMarkerCounts is how many cache_control markers a request body carries,
// broken down by where they sit -- system/instructions, messages/input, or
// tool definitions -- plus how many specify the 1h extended TTL rather than
// the 5m default. Counts only: no request content is captured.
//
// This exists because CacheControlOutcome (cache_control.go) only ever looks
// at messages/system: a marker on a tool definition (Anthropic supports
// cache_control there too, and the router's own OpenAI->Anthropic tool
// converter already carries one through verbatim, see tools.go's
// convertOpenAIToolsToAnthropic/applyWebSearchOptions) reaches the provider
// and can produce real cache reads/writes while CacheControlOutcome still
// reports "none" -- "none" therefore means "nothing in messages/system", not
// "the client sent nothing at all". CountCacheMarkers answers that more
// precisely by scanning every location cache_control can legally appear in,
// independent of InjectCacheControl's own decision logic.
type CacheMarkerCounts struct {
	System   int
	Messages int
	Tools    int
	// TTL1h is how many of the markers above (across all three locations)
	// specify "ttl":"1h"; the rest are the 5m default (either "ttl":"5m"
	// explicitly, or no ttl field at all).
	TTL1h int
}

// Total is the marker count across all three locations.
func (c CacheMarkerCounts) Total() int {
	return c.System + c.Messages + c.Tools
}

// CountCacheMarkers scans a request body -- Chat Completions, native
// Anthropic Messages, or Responses API shape, whichever it turns out to be
// -- for cache_control markers. It does not need to know which shape it got:
// system prompt and messages travel under different keys per shape
// ("system"/"instructions", "messages"/"input"), so both are checked, and
// within each of the three sections it walks every nested object/array
// looking for a "cache_control" key, which finds a marker regardless of how
// deeply nested the content block, tool definition, or image/file part
// holding it is for that shape. Malformed JSON reports all-zero counts.
//
// The body is not decoded to find them. A body that cannot contain the key
// at all is answered from a substring check; otherwise only the three
// sections are walked, and the only values ever decoded are
// the cache_control objects themselves. The full decode (countCacheMarkersFullDecode)
// remains as the reference, and as the fallback for the inputs where a token
// walk and a decode could disagree.
func CountCacheMarkers(body []byte) CacheMarkerCounts {
	return CountCacheMarkersScanned(body, func() *jsonscan.Top {
		top, _ := jsonscan.ScanTop(body)
		return top
	})
}

// CountCacheMarkersScanned is CountCacheMarkers for a caller that may already
// hold a scan of body; scan returns it (nil for a body that is not a JSON
// object).
func CountCacheMarkersScanned(body []byte, scan func() *jsonscan.Top) CacheMarkerCounts {
	// A marker's key is either written as the text cache_control or spelled
	// with an escape. The first is a substring search; the second the scan
	// already knows (Top.EscapedASCII). With neither there is no marker.
	literal := bytes.Contains(body, cacheControlKeyBytes)
	top := scan()
	if top == nil || (!literal && !top.EscapedASCII()) {
		return CacheMarkerCounts{}
	}
	sysCount, sysTTL1h, okSys := countSectionMarkers(top, firstNonNull(top, "system", "instructions"))
	msgCount, msgTTL1h, okMsg := countSectionMarkers(top, firstNonNull(top, "messages", "input"))
	toolsRaw, _ := top.Raw("tools")
	toolsCount, toolsTTL1h, okTools := countSectionMarkers(top, toolsRaw)
	if !okSys || !okMsg || !okTools {
		return countCacheMarkersFullDecode(body)
	}
	return CacheMarkerCounts{
		System:   sysCount,
		Messages: msgCount,
		Tools:    toolsCount,
		TTL1h:    sysTTL1h + msgTTL1h + toolsTTL1h,
	}
}

var cacheControlKeyBytes = []byte("cache_control")

// firstNonNull returns the value of top-level key primary, or of fallback
// when primary is absent or null -- the same choice the full decode makes
// with `v := req[primary]; if v == nil { v = req[fallback] }`.
func firstNonNull(top *jsonscan.Top, primary, fallback string) []byte {
	if raw, ok := top.Raw(primary); ok && !jsonscan.IsNull(raw) {
		return raw
	}
	raw, _ := top.Raw(fallback)
	return raw
}

// countSectionMarkers counts the markers in one section (the raw JSON of
// "messages", "system" or "tools"; empty for an absent section) the way
// countCacheControlMarkers counts them in the decoded value, without
// building that value: jsonscan.FindKey walks the section and hands back the
// cache_control values, and only those few dozen bytes each are decoded and
// judged, by the same rule as everywhere else.
//
// ok=false means "decode the body instead": FindKey could not promise the
// answer a decode would give (see its doc: a repeated or escaped object name,
// a number a decode rejects, a section that does not parse).
func countSectionMarkers(top *jsonscan.Top, section []byte) (count, ttl1h int, ok bool) {
	if len(section) == 0 || (!top.EscapedASCII() && !bytes.Contains(section, cacheControlKeyBytes)) {
		return 0, 0, true
	}
	markers, ok := jsonscan.FindKey(section, "cache_control")
	if !ok {
		return 0, 0, false
	}
	for _, raw := range markers {
		var marker interface{}
		if err := json.Unmarshal(raw, &marker); err != nil {
			return 0, 0, false
		}
		if cc, isMarker := asEphemeralCacheControl(marker); isMarker {
			count++
			if ttl, _ := cc["ttl"].(string); ttl == "1h" {
				ttl1h++
			}
		}
	}
	return count, ttl1h, true
}

// countCacheMarkersFullDecode is the reference implementation: decode the
// whole body and walk the decoded value.
func countCacheMarkersFullDecode(body []byte) CacheMarkerCounts {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return CacheMarkerCounts{}
	}

	systemSrc := req["system"]
	if systemSrc == nil {
		systemSrc = req["instructions"] // Responses API's equivalent field
	}
	messagesSrc := req["messages"]
	if messagesSrc == nil {
		messagesSrc = req["input"] // Responses API's equivalent field
	}

	sysCount, sysTTL1h := countCacheControlMarkers(systemSrc)
	msgCount, msgTTL1h := countCacheControlMarkers(messagesSrc)
	toolsCount, toolsTTL1h := countCacheControlMarkers(req["tools"])

	return CacheMarkerCounts{
		System:   sysCount,
		Messages: msgCount,
		Tools:    toolsCount,
		TTL1h:    sysTTL1h + msgTTL1h + toolsTTL1h,
	}
}

// countCacheControlMarkers walks v (expected to be a map or slice, or
// nil/anything else for an absent section) depth-first, counting every
// "cache_control" key it finds and, among those, how many carry "ttl":"1h".
// It does not recurse into a cache_control value's own fields -- that object
// (e.g. {"type":"ephemeral","ttl":"1h"}) never contains a nested marker, so
// skipping it is a pure (tiny) efficiency, not a correctness requirement.
func countCacheControlMarkers(v interface{}) (count, ttl1h int) {
	switch t := v.(type) {
	case map[string]interface{}:
		// See asEphemeralCacheControl (cache_control.go) for why a field
		// literally named "cache_control" isn't on its own proof of a
		// marker, and why this same check is shared with InjectCacheControl's
		// own client-supplied detection.
		if cc, ok := asEphemeralCacheControl(t["cache_control"]); ok {
			count++
			if ttl, _ := cc["ttl"].(string); ttl == "1h" {
				ttl1h++
			}
		}
		for key, val := range t {
			if key == "cache_control" {
				continue
			}
			c, h := countCacheControlMarkers(val)
			count += c
			ttl1h += h
		}
	case []interface{}:
		for _, item := range t {
			c, h := countCacheControlMarkers(item)
			count += c
			ttl1h += h
		}
	}
	return count, ttl1h
}
