package converter

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	openaiconv "github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// vertexThinking converts a chat body for a Vertex model and returns its
// generationConfig.thinkingConfig.
func vertexThinking(t *testing.T, model string, body []byte) map[string]any {
	t.Helper()
	got, err := New(config.ProviderTypeVertexAI, RequestMode{ModelID: model}).RequestFrom(body)
	require.NoError(t, err)
	m := mustUnmarshal[map[string]any](t, got)
	gen, _ := m["generationConfig"].(map[string]any)
	tc, _ := gen["thinkingConfig"].(map[string]any)
	return tc
}

// The router rewrites the reasoning effort (reasoning_effort_map) on the chat body
// before the Vertex converter runs, so a map composes with the converter's own
// effort -> thinking mapping instead of replacing it.
func TestReasoningEffortMap_ComposesWithVertexThinking(t *testing.T) {
	effortMap, err := config.NewReasoningEffortMap(map[string]string{"xhigh": "high", "max": "high"})
	require.NoError(t, err)
	body := []byte(`{"model":"gemini-3-flash","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"xhigh"}`)

	mapped := openaiconv.MapReasoningEffort(body, effortMap.Resolve)
	assert.Equal(t, "HIGH", vertexThinking(t, "gemini-3-flash", mapped)["thinkingLevel"])

	// Values the map accepts pass through, and Vertex keeps its own per-model rules
	// (pro has no MEDIUM, minimal is floored per model).
	medium := []byte(`{"model":"gemini-3-pro","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"medium"}`)
	withMap := vertexThinking(t, "gemini-3-pro", openaiconv.MapReasoningEffort(medium, effortMap.Resolve))
	assert.Equal(t, vertexThinking(t, "gemini-3-pro", medium), withMap)
	assert.Equal(t, "HIGH", withMap["thinkingLevel"])
}

// Dropping an empty tools array does not change what Vertex receives: the converter
// already ignores empty tools and a tool_choice without functions.
func TestDropEmptyTools_NoChangeForVertex(t *testing.T) {
	body := []byte(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"tools":[],"tool_choice":"auto"}`)
	c := New(config.ProviderTypeVertexAI, RequestMode{ModelID: "gemini-2.5-flash"})
	before, err := c.RequestFrom(body)
	require.NoError(t, err)
	after, err := c.RequestFrom(openaiconv.DropEmptyTools(body))
	require.NoError(t, err)
	assert.JSONEq(t, string(before), string(after))
}
