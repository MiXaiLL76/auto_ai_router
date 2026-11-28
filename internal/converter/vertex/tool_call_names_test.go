package vertex

import (
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// indexToolCallNames must answer every lookup the way
// findFunctionNameByToolCallID does.
func TestIndexToolCallNamesMatchesLookup(t *testing.T) {
	call := func(id string, fields map[string]interface{}) map[string]interface{} {
		out := map[string]interface{}{"id": id, "type": "function"}
		for k, v := range fields {
			out[k] = v
		}
		return out
	}
	fn := func(name string) map[string]interface{} {
		return map[string]interface{}{"function": map[string]interface{}{"name": name, "arguments": "{}"}}
	}
	messages := []openai.OpenAIMessage{
		{Role: "user"},
		{Role: "assistant", ToolCalls: []interface{}{
			call("call_1", fn("first")),
			call("call_2", map[string]interface{}{"name": "flat"}),
			call("call_3", map[string]interface{}{"function": map[string]interface{}{"name": ""}, "name": "flat-after-empty-nested"}),
			call("call_4", nil), // no name anywhere
			"not a map",
			call("", fn("empty id")),
		}},
		{Role: "tool", ToolCalls: []interface{}{call("call_ignored_role", fn("ignored"))}},
		{Role: "assistant"},
		{Role: "assistant", ToolCalls: []interface{}{
			call("call_1", fn("second definition of call_1")),
			call("call_4", fn("named later")),
			call("call_5__thought__c2ln", fn("with signature suffix")),
		}},
	}
	for i := 0; i < 40; i++ {
		messages = append(messages, openai.OpenAIMessage{Role: "assistant", ToolCalls: []interface{}{call(fmt.Sprintf("call_n%d", i), fn(fmt.Sprintf("tool_%d", i%3)))}})
	}
	names := indexToolCallNames(messages)
	ids := []string{"call_1", "call_2", "call_3", "call_4", "call_5", "call_5__thought__c2ln", "call_5__thought__other", "", "missing", "call_ignored_role", "call_n0", "call_n39", "call_1__thought__x"}
	for _, id := range ids {
		want := findFunctionNameByToolCallID(messages, id)
		base, _ := converterutil.SplitToolCallIDSignature(id)
		if got := names[base]; got != want {
			t.Errorf("tool call %q: index says %q, lookup says %q", id, got, want)
		}
	}
	if findFunctionNameByToolCallID(messages, "call_1") != "first" || findFunctionNameByToolCallID(messages, "call_4") != "named later" {
		t.Fatal("test bug: the reference no longer behaves as assumed")
	}
}
