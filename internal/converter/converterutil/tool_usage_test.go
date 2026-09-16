package converterutil

import (
	"encoding/json"
	"testing"
)

func TestToolUsageExtensions_WebSearchRequests(t *testing.T) {
	tests := []struct {
		name  string
		usage string
		want  int
	}{
		{
			name:  "no extensions",
			usage: `{"input_tokens":10}`,
			want:  0,
		},
		{
			name:  "x_tools",
			usage: `{"x_tools":{"web_search":{"count":2}}}`,
			want:  2,
		},
		{
			name:  "x_tools wins over x_details and is not summed with it",
			usage: `{"x_tools":{"web_search":{"count":1}},"x_details":[{"x_billing_type":"response_api","plugins":{"web_search":{"count":1}}}]}`,
			want:  1,
		},
		{
			name:  "x_details fallback when x_tools is absent",
			usage: `{"x_details":[{"x_billing_type":"response_api","plugins":{"web_search":{"count":3}}}]}`,
			want:  3,
		},
		{
			name:  "x_details fallback when x_tools has no web search",
			usage: `{"x_tools":{"code_interpreter":{"count":1}},"x_details":[{"plugins":{"web_search":{"count":1}}}]}`,
			want:  1,
		},
		{
			name:  "x_details lines are not summed",
			usage: `{"x_details":[{"plugins":{"web_search":{"count":2}}},{"plugins":{"web_search":{"count":2}}}]}`,
			want:  2,
		},
		{
			name:  "plugins.search",
			usage: `{"plugins":{"search":{"count":1,"strategy":"agent"}}}`,
			want:  1,
		},
		{
			name:  "x_tools wins over plugins.search",
			usage: `{"x_tools":{"web_search":{"count":2}},"plugins":{"search":{"count":1}}}`,
			want:  2,
		},
		{
			name:  "unexpected shapes are ignored",
			usage: `{"x_tools":"web_search","x_details":{"plugins":{}},"plugins":["search"]}`,
			want:  0,
		},
		{
			name:  "malformed x_tools falls through to plugins.search",
			usage: `{"x_tools":{"web_search":{"count":"1"}},"plugins":{"search":{"count":1}}}`,
			want:  1,
		},
		{
			name:  "negative counts are ignored",
			usage: `{"x_tools":{"web_search":{"count":-1}},"plugins":{"search":{"count":-2}}}`,
			want:  0,
		},
		{
			name:  "nulls",
			usage: `{"x_tools":null,"x_details":null,"plugins":null}`,
			want:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var usage struct {
				InputTokens int `json:"input_tokens"`
				ToolUsageExtensions
			}
			if err := json.Unmarshal([]byte(tt.usage), &usage); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got := usage.WebSearchRequests(); got != tt.want {
				t.Fatalf("WebSearchRequests() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestToolUsageExtensions_RoundTrip(t *testing.T) {
	const usage = `{"input_tokens":10,"x_tools":{"web_search":{"count":1}},"x_details":[{"plugins":{"web_search":{"count":1}},"x_billing_type":"response_api"}]}`

	var decoded struct {
		InputTokens int `json:"input_tokens"`
		ToolUsageExtensions
	}
	if err := json.Unmarshal([]byte(usage), &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	encoded, err := json.Marshal(decoded)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(encoded) != usage {
		t.Fatalf("round trip changed usage:\n got %s\nwant %s", encoded, usage)
	}

	empty, err := json.Marshal(struct {
		InputTokens int `json:"input_tokens"`
		ToolUsageExtensions
	}{InputTokens: 1})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(empty) != `{"input_tokens":1}` {
		t.Fatalf("absent extensions must be omitted, got %s", empty)
	}
}
