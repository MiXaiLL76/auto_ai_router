package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type nativeWSTestDB struct {
	*clientAuthTestDB
	entries chan *dbmodels.SpendLogEntry
}

func (d *nativeWSTestDB) SpendLoggingEnabled() bool { return true }
func (d *nativeWSTestDB) LogSpend(entry *dbmodels.SpendLogEntry) error {
	d.entries <- entry
	return nil
}

type nativeWSFixture struct {
	proxy    *Proxy
	db       *nativeWSTestDB
	client   *websocket.Conn
	accepted chan *websocket.Conn
}

func newNativeWSFixture(t *testing.T, provider config.ProviderType, basePath string, options ...func(*Proxy)) *nativeWSFixture {
	t.Helper()
	accepted := make(chan *websocket.Conn, 1)
	done := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, basePath+"/v1/responses", r.URL.Path)
		assert.Equal(t, "configured=yes", r.URL.RawQuery)
		assert.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("Cookie"))
		assert.Empty(t, r.Header.Get("X-Api-Key"))
		assert.Equal(t, "responses_websockets=2026-02-06", r.Header.Get("OpenAI-Beta"))
		if provider.IsProxyLike() {
			assert.Equal(t, "1", r.Header.Get(HeaderAIRProxyClient))
		}
		conn, err := wsUpgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = conn.Close() }()
		accepted <- conn
		<-done
	}))
	t.Cleanup(upstream.Close)
	t.Cleanup(func() { close(done) })
	cred := config.CredentialConfig{Name: "upstream", Type: provider, BaseURL: upstream.URL + basePath + "?configured=yes", APIKey: "upstream-key", RPM: 1000, TPM: 1000000}
	prx := NewTestProxyBuilder().WithCredentials(cred).WithMasterKey("master-key").Build()
	prx.modelManager = models.New(prx.logger, 1000, []config.ModelRPMConfig{{Name: "astra", Model: "gpt-6-astra", Credential: "upstream", WebSocketResponses: true}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{cred})
	db := &nativeWSTestDB{clientAuthTestDB: &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{"client-key": {Token: "client-hash", Models: []string{"astra"}}}, errors: make(map[string]error)}, entries: make(chan *dbmodels.SpendLogEntry, 32)}
	prx.LiteLLMDB = db
	prx.strictAllTeamModelsACL = true
	for _, option := range options {
		option(prx)
	}
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"astra": {InputCostPerToken: 1, OutputCostPerToken: 2, CacheReadInputTokenCost: 0.1, CacheCreationInputTokenCost: 1.25}})
	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	t.Cleanup(server.Close)
	client, response, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/v1/responses?api-key=do-not-forward", http.Header{
		"Authorization": {"Bearer client-key"}, "Cookie": {"session=private"}, "Openai-Beta": {"responses_websockets=2026-02-06"},
	})
	require.NoError(t, err)
	require.Equal(t, http.StatusSwitchingProtocols, response.StatusCode)
	t.Cleanup(func() { _ = client.Close() })
	return &nativeWSFixture{prx, db, client, accepted}
}

func wsWrite(t *testing.T, conn *websocket.Conn, body string) {
	t.Helper()
	require.NoError(t, writeNativeWS(conn, []byte(body)))
}
func wsRead(t *testing.T, conn *websocket.Conn) map[string]json.RawMessage {
	t.Helper()
	require.NoError(t, conn.SetReadDeadline(time.Now().Add(5*time.Second)))
	_, body, err := conn.ReadMessage()
	require.NoError(t, err)
	var event map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &event))
	return event
}
func wsEvent(t *testing.T, conn *websocket.Conn, typ string) map[string]json.RawMessage {
	t.Helper()
	event := wsRead(t, conn)
	assert.JSONEq(t, fmt.Sprintf("%q", typ), string(event["type"]))
	return event
}
func (f *nativeWSFixture) upstream(t *testing.T) *websocket.Conn {
	t.Helper()
	select {
	case conn := <-f.accepted:
		return conn
	case <-time.After(5 * time.Second):
		t.Fatal("no upstream handshake")
		return nil
	}
}
func (f *nativeWSFixture) entry(t *testing.T) *dbmodels.SpendLogEntry {
	t.Helper()
	select {
	case entry := <-f.db.entries:
		return entry
	case <-time.After(5 * time.Second):
		t.Fatal("no spend entry")
		return nil
	}
}
func wsCreated(id string) string {
	return fmt.Sprintf(`{"type":"response.created","response":{"id":%q,"model":"gpt-6-astra","status":"in_progress"}}`, id)
}
func wsCompleted(id string) string {
	return fmt.Sprintf(`{"type":"response.completed","response":{"id":%q,"model":"gpt-6-astra","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":20,"input_tokens_details":{"cached_tokens":40,"cache_write_tokens":10},"output_tokens_details":{"reasoning_tokens":5}}}}`, id)
}

func TestNativeWebSocketSteeringAndBilling(t *testing.T) {
	for _, provider := range []config.ProviderType{config.ProviderTypeOpenAI, config.ProviderTypeProxy} {
		t.Run(string(provider), func(t *testing.T) {
			f := newNativeWSFixture(t, provider, "/openai")
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"plan","reasoning":{"effort":"max"},"temperature":0.4,"max_output_tokens":256,"prompt_cache_options":{"ttl":"30m","mode":"explicit"},"additional_tools":[{"type":"function","name":"lookup","async":true}],"tool_choice":"required"}`)
			upstream := f.upstream(t)
			request := wsEvent(t, upstream, "response.create")
			expectedModel := `"gpt-6-astra"`
			if provider.IsProxyLike() {
				expectedModel = `"astra"`
			}
			assert.JSONEq(t, expectedModel, string(request["model"]))
			assert.NotContains(t, request, "stream")
			if !provider.IsProxyLike() {
				assert.NotContains(t, request, "temperature")
			}
			assert.Equal(t, "false", string(request["store"]))
			assert.JSONEq(t, `{"effort":"max"}`, string(request["reasoning"]))
			assert.JSONEq(t, `{"ttl":"30m","mode":"explicit"}`, string(request["prompt_cache_options"]))
			assert.JSONEq(t, `[{"type":"function","name":"lookup","async":true}]`, string(request["additional_tools"]))
			wsWrite(t, upstream, wsCreated("resp_1"))
			created := wsEvent(t, f.client, "response.created")
			assert.Contains(t, string(created["response"]), `"model":"astra"`)
			wsWrite(t, upstream, `{"type":"response.output_text.delta","response_id":"resp_1","delta":"Draft"}`)
			wsEvent(t, f.client, "response.output_text.delta")
			steer := `{"type":"response.steer","previous_response_id":"resp_1","input":"Keep it short"}`
			wsWrite(t, f.client, steer)
			got := wsEvent(t, upstream, "response.steer")
			assert.JSONEq(t, `"resp_1"`, string(got["previous_response_id"]))
			wsWrite(t, upstream, `{"type":"response.steer.accepted","steer":{"id":"steer_1","previous_response_id":"resp_1"}}`)
			wsEvent(t, f.client, "response.steer.accepted")
			wsWrite(t, upstream, `{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"steered"},"usage":{"input_tokens":10,"output_tokens":3}}}`)
			wsEvent(t, f.client, "response.incomplete")
			first := f.entry(t)
			assert.Equal(t, 10, first.PromptTokens)
			assert.InDelta(t, 16, first.Spend, 1e-9)
			wsWrite(t, upstream, wsCreated("resp_2"))
			wsEvent(t, f.client, "response.created")
			wsWrite(t, upstream, wsCompleted("resp_2"))
			wsEvent(t, f.client, "response.completed")
			second := f.entry(t)
			assert.Equal(t, 100, second.PromptTokens)
			assert.Equal(t, 20, second.CompletionTokens)
			assert.InDelta(t, 106.5, second.Spend, 1e-9)
			assert.NotEqual(t, first.RequestID, second.RequestID)
			wsWrite(t, upstream, wsCompleted("resp_2"))
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","previous_response_id":"resp_2","input":[{"type":"function_call_output","call_id":"call_1","output":"done"},{"type":"configuration_update","reasoning":{"effort":"high"}}]}`)
			next := wsEvent(t, upstream, "response.create")
			assert.JSONEq(t, `"resp_2"`, string(next["previous_response_id"]))
			assert.JSONEq(t, `[{"type":"function_call_output","call_id":"call_1","output":"done"},{"type":"configuration_update","reasoning":{"effort":"high"}}]`, string(next["input"]))
			wsWrite(t, upstream, wsCreated("resp_3"))
			wsEvent(t, f.client, "response.created")
			wsWrite(t, upstream, wsCompleted("resp_3"))
			wsEvent(t, f.client, "response.completed")
			third := f.entry(t)
			assert.NotEqual(t, second.RequestID, third.RequestID)
			select {
			case extra := <-f.db.entries:
				t.Fatalf("duplicate charge: %+v", extra)
			default:
			}
		})
	}
}

func TestNativeWebSocketPendingToolSteer(t *testing.T) {
	f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"lookup"}`)
	upstream := f.upstream(t)
	wsEvent(t, upstream, "response.create")
	wsWrite(t, upstream, wsCreated("resp_1"))
	wsEvent(t, f.client, "response.created")
	wsWrite(t, f.client, `{"type":"response.steer","previous_response_id":"resp_1","input":"summarize"}`)
	wsEvent(t, upstream, "response.steer")
	wsWrite(t, upstream, `{"type":"response.completed","response":{"id":"resp_1","output":[{"type":"function_call","name":"lookup","call_id":"call_1","arguments":"{}"}],"usage":{"input_tokens":4,"output_tokens":2}}}`)
	wsEvent(t, f.client, "response.completed")
	f.entry(t)
	wsWrite(t, upstream, `{"type":"response.steer.pending","steer":{"previous_response_id":"resp_1"},"required_input":[{"type":"function_call_output","call_id":"call_1"}]}`)
	wsEvent(t, f.client, "response.steer.pending")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"42"}]}`)
	resumed := wsEvent(t, upstream, "response.create")
	assert.JSONEq(t, `[{"type":"function_call_output","call_id":"call_1","output":"42"}]`, string(resumed["input"]))
	wsWrite(t, upstream, wsCreated("resp_2"))
	wsEvent(t, f.client, "response.created")
	wsWrite(t, upstream, wsCompleted("resp_2"))
	wsEvent(t, f.client, "response.completed")
	f.entry(t)
}

func TestNativeWebSocketRechecksAccess(t *testing.T) {
	for _, steering := range []bool{false, true} {
		t.Run(fmt.Sprint(steering), func(t *testing.T) {
			f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"hi"}`)
			upstream := f.upstream(t)
			wsEvent(t, upstream, "response.create")
			wsWrite(t, upstream, wsCreated("resp_1"))
			wsEvent(t, f.client, "response.created")
			if !steering {
				wsWrite(t, upstream, wsCompleted("resp_1"))
				wsEvent(t, f.client, "response.completed")
				f.entry(t)
			}
			f.db.mu.Lock()
			f.db.errors["client-key"] = litellmdb.ErrTokenNotFound
			f.db.mu.Unlock()
			request := `{"type":"response.create","model":"astra","previous_response_id":"resp_1","input":"hi"}`
			if steering {
				request = `{"type":"response.steer","previous_response_id":"resp_1","input":"hi"}`
			}
			wsWrite(t, f.client, request)
			event := wsEvent(t, f.client, "error")
			assert.Contains(t, string(event["error"]), "Invalid")
		})
	}
}

func TestNativeWebSocketRejectsInvalidContinuation(t *testing.T) {
	f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"hi"}`)
	upstream := f.upstream(t)
	wsEvent(t, upstream, "response.create")
	wsWrite(t, upstream, wsCreated("resp_1"))
	wsEvent(t, f.client, "response.created")
	for _, request := range []string{
		`{"type":"response.create","model":"astra","input":"overlap"}`,
		`{"type":"response.steer","previous_response_id":"foreign","input":"hi"}`,
		`{"type":"response.steer","previous_response_id":"resp_1","model":"other","input":"hi"}`,
		`{"type":"unsupported"}`,
	} {
		wsWrite(t, f.client, request)
		wsEvent(t, f.client, "error")
	}
	wsWrite(t, upstream, wsCompleted("resp_1"))
	wsEvent(t, f.client, "response.completed")
	f.entry(t)
	for _, request := range []string{
		`{"type":"response.create","model":"other","input":"hi"}`,
		`{"type":"response.create","model":"astra","previous_response_id":"foreign","input":"hi"}`,
	} {
		wsWrite(t, f.client, request)
		wsEvent(t, f.client, "error")
	}
	f.proxy.balancer.BanUntil("upstream", "astra", 429, time.Now().Add(time.Hour), "test")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","previous_response_id":"resp_1","input":"hi"}`)
	rejected := wsEvent(t, f.client, "error")
	assert.Contains(t, string(rejected["error"]), "credential unavailable")
}

func TestNativeWebSocketURL(t *testing.T) {
	for _, tt := range []struct{ base, want string }{
		{"https://api.openai.com/v1", "wss://api.openai.com/v1/responses"},
		{"https://resource.openai.azure.com/openai/v1", "wss://resource.openai.azure.com/openai/v1/responses"},
		{"https://resource.openai.azure.com/openai", "wss://resource.openai.azure.com/openai/v1/responses"},
		{"http://localhost:8080?api-version=preview", "ws://localhost:8080/v1/responses?api-version=preview"},
	} {
		got, err := nativeWebSocketURL(tt.base)
		require.NoError(t, err)
		assert.Equal(t, tt.want, got)
	}
	for _, base := range []string{"wss://example.com", "/relative", "https://user:password@example.com"} {
		_, err := nativeWebSocketURL(base)
		require.Error(t, err)
	}
}

func TestNativeWebSocketHistoryBudgetEstimate(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	setTestModelPrice(prx, "gpt-6-astra", &models.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
	body := []byte(`{"input":"new input","max_output_tokens":10}`)
	plain := &RequestLogContext{}
	base, ok := prx.estimateRequestCost(plain, "gpt-6-astra", "gpt-6-astra", "gpt-6-astra", body)
	require.True(t, ok)
	ctx := context.WithValue(context.Background(), nativeWSRoutingKey{}, &nativeWSRouting{historyTokens: 1000})
	logCtx := &RequestLogContext{Request: httptest.NewRequest(http.MethodPost, "/v1/responses", nil).WithContext(ctx)}
	got, ok := prx.estimateRequestCost(logCtx, "gpt-6-astra", "gpt-6-astra", "gpt-6-astra", body)
	require.True(t, ok)
	assert.Equal(t, base+1000, got)
	prx.setPromptTokensEstimate(logCtx, body, "gpt-6-astra")
	assert.Equal(t, 1000+estimatePromptTokensForModel(body, "gpt-6-astra"), logCtx.promptTokensEstimate())
}

func TestNativeWebSocketSteerFailureAllowsRetry(t *testing.T) {
	f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"hi"}`)
	upstream := f.upstream(t)
	wsEvent(t, upstream, "response.create")
	wsWrite(t, upstream, wsCreated("resp_1"))
	wsEvent(t, f.client, "response.created")
	steer := `{"type":"response.steer","previous_response_id":"resp_1","input":"shorter"}`
	wsWrite(t, f.client, steer)
	wsEvent(t, upstream, "response.steer")
	wsWrite(t, upstream, `{"type":"response.steer.failed","steer":{"previous_response_id":"resp_1"},"error":{"code":"invalid_input","message":"upstream private details"}}`)
	event := wsEvent(t, f.client, "response.steer.failed")
	assert.NotContains(t, string(event["error"]), "upstream private details")
	wsWrite(t, f.client, steer)
	wsEvent(t, upstream, "response.steer")
	wsWrite(t, upstream, wsCompleted("resp_1"))
	wsEvent(t, f.client, "response.completed")
	f.entry(t)
	wsWrite(t, upstream, wsCreated("resp_2"))
	wsEvent(t, f.client, "response.created")
	wsWrite(t, upstream, wsCompleted("resp_2"))
	wsEvent(t, f.client, "response.completed")
	f.entry(t)
	select {
	case extra := <-f.db.entries:
		t.Fatalf("unused steer charged: %+v", extra)
	default:
	}
}

func TestNativeWebSocketHandshakeFailureDoesNotCharge(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "private provider error", http.StatusBadGateway)
	}))
	defer upstream.Close()
	cred := config.CredentialConfig{Name: "upstream", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "upstream-key", RPM: 100, TPM: 10000}
	prx := NewTestProxyBuilder().WithCredentials(cred).Build()
	prx.modelManager = models.New(prx.logger, 100, []config.ModelRPMConfig{{Name: "gpt-6-astra", Credential: "upstream", WebSocketResponses: true}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{cred})
	db := &nativeWSTestDB{clientAuthTestDB: &clientAuthTestDB{}, entries: make(chan *dbmodels.SpendLogEntry, 1)}
	prx.LiteLLMDB = db
	setTestModelPrice(prx, "gpt-6-astra", &models.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	defer server.Close()
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/v1/responses", http.Header{"Authorization": {"Bearer master-key"}})
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	wsWrite(t, conn, `{"type":"response.create","model":"gpt-6-astra","input":"should not be billed"}`)
	event := wsEvent(t, conn, "error")
	assert.NotContains(t, string(event["error"]), "private provider error")
	select {
	case entry := <-db.entries:
		assert.Zero(t, entry.Spend)
		assert.Zero(t, entry.PromptTokens)
	case <-time.After(5 * time.Second):
		t.Fatal("missing failure log")
	}
}

func TestNativeWebSocketOptOutUsesHTTP(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.False(t, websocket.IsWebSocketUpgrade(r))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", wsCompleted("resp_http"))
	}))
	defer upstream.Close()
	prx := NewTestProxyBuilder().WithSingleCredential("upstream", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").Build()
	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	defer server.Close()
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/v1/responses", http.Header{"Authorization": {"Bearer master-key"}})
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	wsWrite(t, conn, `{"type":"response.create","model":"gpt-6-astra","input":"hi"}`)
	wsEvent(t, conn, "response.completed")
}

func TestNativeWebSocketDrainsUsageAfterDisconnect(t *testing.T) {
	f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "", func(p *Proxy) { p.drainUpstreamOnAbort = true })
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"hi"}`)
	upstream := f.upstream(t)
	wsEvent(t, upstream, "response.create")
	wsWrite(t, upstream, wsCreated("resp_1"))
	wsEvent(t, f.client, "response.created")
	require.NoError(t, f.client.Close())
	wsWrite(t, upstream, wsCompleted("resp_1"))
	entry := f.entry(t)
	assert.Equal(t, 100, entry.PromptTokens)
	assert.Equal(t, 20, entry.CompletionTokens)
	assert.InDelta(t, 106.5, entry.Spend, 1e-9)
	require.NoError(t, upstream.SetReadDeadline(time.Now().Add(5*time.Second)))
	_, _, err := upstream.ReadMessage()
	require.Error(t, err)
}

func TestNativeWebSocketRechecksModelAndBudget(t *testing.T) {
	for _, blockedBy := range []string{"model", "budget"} {
		t.Run(blockedBy, func(t *testing.T) {
			f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"hi"}`)
			upstream := f.upstream(t)
			wsEvent(t, upstream, "response.create")
			wsWrite(t, upstream, wsCreated("resp_1"))
			wsEvent(t, f.client, "response.created")
			wsWrite(t, upstream, wsCompleted("resp_1"))
			wsEvent(t, f.client, "response.completed")
			f.entry(t)
			f.db.mu.Lock()
			info := *f.db.tokens["client-key"]
			if blockedBy == "model" {
				info.Models = []string{"different-model"}
			} else {
				info.Spend = 10
				info.MaxBudget = pointerTo(1.0)
			}
			f.db.tokens["client-key"] = &info
			f.db.mu.Unlock()
			wsWrite(t, f.client, `{"type":"response.create","model":"astra","previous_response_id":"resp_1","input":"hi"}`)
			wsEvent(t, f.client, "error")
		})
	}
}

func TestNativeWebSocketCannotResumeAnotherConnection(t *testing.T) {
	f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","previous_response_id":"resp_other_connection","input":"hi"}`)
	event := wsEvent(t, f.client, "error")
	assert.Contains(t, string(event["error"]), "previous_response_not_found")
	select {
	case <-f.accepted:
		t.Fatal("foreign response reached upstream")
	default:
	}
}

func TestNativeWebSocketDisconnectUsesUsageEstimate(t *testing.T) {
	f := newNativeWSFixture(t, config.ProviderTypeOpenAI, "")
	wsWrite(t, f.client, `{"type":"response.create","model":"astra","input":"estimate these input tokens"}`)
	upstream := f.upstream(t)
	wsEvent(t, upstream, "response.create")
	wsWrite(t, upstream, wsCreated("resp_1"))
	wsEvent(t, f.client, "response.created")
	wsWrite(t, upstream, `{"type":"response.output_text.delta","response_id":"resp_1","delta":"partial response"}`)
	wsEvent(t, f.client, "response.output_text.delta")
	require.NoError(t, upstream.Close())
	wsEvent(t, f.client, "error")
	entry := f.entry(t)
	assert.Positive(t, entry.PromptTokens)
	assert.Positive(t, entry.CompletionTokens)
	select {
	case <-f.accepted:
		t.Fatal("unexpected automatic reconnect")
	default:
	}
}
