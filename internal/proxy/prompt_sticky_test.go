package proxy

import (
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	stickyChatCached   = `{"model":"gpt-4","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":[{"type":"text","text":"rules","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":"hello"}]}`
	stickyChatMovedBP  = `{"model":"gpt-4","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":[{"type":"text","text":"rules"}]},{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":[{"type":"text","text":"next","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`
	stickyChatNoCache  = `{"model":"gpt-4","messages":[{"role":"system","content":"rules"},{"role":"user","content":"hello"}]}`
	stickyChatOtherMsg = `{"model":"gpt-4","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":[{"type":"text","text":"rules","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":"bye"}]}`
)

func promptStickyKeyFor(body, token, model string) string {
	return promptStickyKey([]byte(body), token, model, anthropicconv.CountCacheMarkers([]byte(body)).Total() > 0)
}

func TestPromptStickyKey(t *testing.T) {
	base := promptStickyKeyFor(stickyChatCached, "sk-a", "gpt-4")
	require.True(t, validPromptStickyKey(base), base)

	assert.Equal(t, base, promptStickyKeyFor(stickyChatMovedBP, "sk-a", "gpt-4"), "later turns and a moved breakpoint keep the key")
	assert.NotEqual(t, base, promptStickyKeyFor(stickyChatCached, "sk-b", "gpt-4"), "client key")
	assert.NotEqual(t, base, promptStickyKeyFor(stickyChatCached, "sk-a", "gpt-5"), "model")
	assert.NotEqual(t, base, promptStickyKeyFor(stickyChatOtherMsg, "sk-a", "gpt-4"), "first user message")
	assert.Empty(t, promptStickyKeyFor(stickyChatNoCache, "sk-a", "gpt-4"), "no cache_control")
	assert.Empty(t, promptStickyKey([]byte(`not json`), "sk-a", "gpt-4", true))
}

func TestPromptStickyKey_Shapes(t *testing.T) {
	messages := `{"model":"claude","system":[{"type":"text","text":"rules","cache_control":{"type":"ephemeral"}}],"tools":[{"name":"f","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hello"}]}`
	messagesNext := `{"model":"claude","system":[{"type":"text","text":"rules"}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":[{"type":"text","text":"more","cache_control":{"type":"ephemeral"}}]}]}`
	a := promptStickyKeyFor(messages, "sk", "claude")
	require.NotEmpty(t, a)
	assert.Equal(t, a, promptStickyKeyFor(messagesNext, "sk", "claude"))

	topLevel := `{"model":"claude","cache_control":{"type":"ephemeral"},"system":"rules","messages":[{"role":"user","content":"hello"}]}`
	topLevelNext := `{"model":"claude","cache_control":{"type":"ephemeral"},"system":"rules","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"more"}]}`
	b := promptStickyKeyFor(topLevel, "sk", "claude")
	require.NotEmpty(t, b, "top-level cache_control")
	assert.Equal(t, b, promptStickyKeyFor(topLevelNext, "sk", "claude"))
	assert.Empty(t, promptStickyKeyFor(`{"model":"claude","cache_control":{"type":"other"},"messages":[{"role":"user","content":"hello"}]}`, "sk", "claude"))

	responses := `{"model":"claude","instructions":"rules","input":[{"role":"user","content":[{"type":"input_text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}`
	responsesNext := `{"model":"claude","instructions":"rules","input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]},{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":"x","cache_control":{"type":"ephemeral"}}]}`
	c := promptStickyKeyFor(responses, "sk", "claude")
	require.NotEmpty(t, c)
	assert.Equal(t, c, promptStickyKeyFor(responsesNext, "sk", "claude"))

	stringInput := promptStickyKeyFor(`{"model":"claude","cache_control":{"type":"ephemeral"},"instructions":"rules","input":"hello"}`, "sk", "claude")
	assert.NotEmpty(t, stringInput)
	assert.NotEqual(t, stringInput, promptStickyKeyFor(`{"model":"claude","cache_control":{"type":"ephemeral"},"instructions":"rules","input":"bye"}`, "sk", "claude"))
}

func TestPromptStickyKey_ContentForms(t *testing.T) {
	for _, tc := range []struct {
		name        string
		first, next string
	}{
		{
			"chat: marker on the first message, then on the tail",
			`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}`,
			`{"model":"m","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":[{"type":"text","text":"more","cache_control":{"type":"ephemeral"}}]}]}`,
		},
		{
			"messages: system string and blocks",
			`{"model":"m","cache_control":{"type":"ephemeral"},"system":"rules","messages":[{"role":"user","content":"hello"}]}`,
			`{"model":"m","system":[{"type":"text","text":"rules","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"more"}]}`,
		},
		{
			"responses: string input, then history",
			`{"model":"m","cache_control":{"type":"ephemeral"},"instructions":"rules","input":"hello"}`,
			`{"model":"m","cache_control":{"type":"ephemeral"},"instructions":"rules","input":[{"role":"user","content":"hello"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]},{"role":"user","content":"more"}]}`,
		},
		{
			"responses: message item with input_text",
			`{"model":"m","cache_control":{"type":"ephemeral"},"input":"hello"}`,
			`{"model":"m","cache_control":{"type":"ephemeral"},"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},{"role":"user","content":"more"}]}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			first := promptStickyKeyFor(tc.first, "sk", "m")
			require.NotEmpty(t, first)
			assert.Equal(t, first, promptStickyKeyFor(tc.next, "sk", "m"))
		})
	}
}

func TestPromptStickyKey_KeepsSchemaPropertyNamedCacheControl(t *testing.T) {
	tool := func(propType string) string {
		return `{"model":"m","tools":[{"name":"f","input_schema":{"type":"object","properties":{"cache_control":{"type":"` + propType + `"}}}}],"system":[{"type":"text","text":"rules","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hello"}]}`
	}
	assert.NotEqual(t, promptStickyKeyFor(tool("string"), "sk", "m"), promptStickyKeyFor(tool("integer"), "sk", "m"))
}

func TestValidPromptStickyKey(t *testing.T) {
	assert.True(t, validPromptStickyKey(promptStickyKeyPrefix+strings.Repeat("a", promptStickyKeyHexSize)))
	for _, key := range []string{
		"",
		"sess-1",
		promptStickyKeyPrefix + strings.Repeat("a", promptStickyKeyHexSize-1),
		promptStickyKeyPrefix + strings.Repeat("z", promptStickyKeyHexSize),
		"x:" + strings.Repeat("a", promptStickyKeyHexSize),
	} {
		assert.False(t, validPromptStickyKey(key), key)
	}
}

func newCountingServers(t *testing.T, n int) ([]*httptest.Server, []*atomic.Int32) {
	t.Helper()
	servers := make([]*httptest.Server, n)
	hits := make([]*atomic.Int32, n)
	for i := range servers {
		counter := &atomic.Int32{}
		hits[i] = counter
		servers[i] = newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(chatCompletionOKBody("id", "ok")))
		}))
		t.Cleanup(servers[i].Close)
	}
	return servers, hits
}

func usedServers(hits []*atomic.Int32) int {
	used := 0
	for _, h := range hits {
		if h.Load() > 0 {
			used++
		}
	}
	return used
}

func sendChat(t *testing.T, prx *Proxy, body string) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
}

func TestProxyRequest_PromptHashSticky(t *testing.T) {
	for _, tc := range []struct {
		name     string
		enabled  bool
		spendLog bool
		body     string
		wantUsed int
	}{
		{"cache_control binds", true, false, stickyChatCached, 1},
		{"cache_control binds on direct credentials with spend log", true, true, stickyChatCached, 1},
		{"no cache_control balances", true, false, stickyChatNoCache, 3},
		{"flag off balances", false, false, stickyChatCached, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			servers, hits := newCountingServers(t, 3)
			creds := []config.CredentialConfig{
				testProxyCredential("cred1", servers[0].URL, false),
				testProxyCredential("cred2", servers[1].URL, false),
				testProxyCredential("cred3", servers[2].URL, false),
			}
			if tc.spendLog {
				for i := range creds {
					creds[i].Type = config.ProviderTypeOpenAI
				}
			}
			builder := NewTestProxyBuilder().
				WithCredentials(creds...).
				WithMasterKey("master-key").
				WithSessionSticky(5 * time.Minute)
			if tc.enabled {
				builder = builder.WithPromptHashSticky()
			}
			if tc.spendLog {
				builder = builder.WithRouterResponseLog(&collectingRouterResponseManager{}, true)
			}
			prx := builder.Build()

			for range 5 {
				sendChat(t, prx, tc.body)
			}
			assert.Equal(t, tc.wantUsed, usedServers(hits))
		})
	}
}

func TestProxyRequest_PromptHashStickyForwardedToAIRHop(t *testing.T) {
	var gotSession string
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotSession = r.Header.Get(HeaderSessionID)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("chatcmpl-1", "gpt-4", "ok"))
	}))
	defer upstream.Close()

	cred := proxyCred("downstream", upstream.URL, 1)
	cred.Type = config.ProviderTypeAIR
	prx := NewTestProxyBuilder().WithCredentials(cred).WithMasterKey("master-key").
		WithSessionSticky(5 * time.Minute).WithPromptHashSticky().Build()
	registerTestModel(prx, cred.Name, "gpt-4")

	sendChat(t, prx, stickyChatCached)

	assert.True(t, validPromptStickyKey(gotSession), gotSession)
}

func validPromptStickyKey(key string) bool {
	hexPart, ok := strings.CutPrefix(key, promptStickyKeyPrefix)
	if !ok || len(hexPart) != promptStickyKeyHexSize {
		return false
	}
	_, err := hex.DecodeString(hexPart)
	return err == nil
}

func TestProxyRequest_PromptHashStickyKeepsKeyAcrossPreviousResponseID(t *testing.T) {
	for _, model := range []string{"gpt-4", "gpt-alias"} {
		t.Run(model, func(t *testing.T) { testPromptHashStickyAcrossPreviousResponseID(t, model) })
	}
}

func testPromptHashStickyAcrossPreviousResponseID(t *testing.T, model string) {
	var sessions []string
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sessions = append(sessions, r.Header.Get(HeaderSessionID))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp-1","object":"response","model":"gpt-4","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	cred := proxyCred("downstream", upstream.URL, 1)
	cred.Type = config.ProviderTypeAIR
	prx := NewTestProxyBuilder().WithCredentials(cred).WithMasterKey("master-key").
		WithSessionSticky(5 * time.Minute).WithPromptHashSticky().Build()
	mm := models.New(testhelpers.NewTestLogger(), 50, []config.ModelRPMConfig{{Name: "gpt-4", Credential: cred.Name}})
	mm.LoadModelsFromConfig([]config.CredentialConfig{cred})
	mm.SetModelAliases(map[string]string{"gpt-alias": "gpt-4"})
	prx.modelManager = mm
	prx.responseStore = &fakeResponseStore{entry: &responsestore.StoredEntry{
		ResponseID:       "resp-prev",
		CredentialName:   cred.Name,
		AccumulatedInput: json.RawMessage(`[{"role":"user","content":"hello"}]`),
		ResponseJSON:     &responses.Response{ID: "resp-prev", Model: "gpt-4"},
	}}

	for _, body := range []string{
		`{"model":"` + model + `","cache_control":{"type":"ephemeral"},"instructions":"rules","input":"hello"}`,
		`{"model":"` + model + `","cache_control":{"type":"ephemeral"},"instructions":"rules","input":"more","previous_response_id":"resp-prev"}`,
	} {
		req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		prx.ProxyRequest(w, req)
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	}

	require.Len(t, sessions, 2)
	assert.NotEmpty(t, sessions[0])
	assert.Equal(t, sessions[0], sessions[1])
}
