package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A model that always runs adaptive thinking (see anthropic.ValidateThinkingControls):
// its thinking controls reach an Anthropic-wire upstream as adaptive thinking plus the
// client's effort, and the settings it rejects are answered by the router with a 400
// that never reaches the upstream.
func TestNativeAdaptiveModelThinkingControls(t *testing.T) {
	const model = "MiniMax-M3.1-Flash"
	tests := []struct {
		name          string
		path          string
		body          string
		wantStatus    int
		wantParam     string                 // for a 400
		wantThinking  map[string]interface{} // forwarded thinking, nil = absent
		wantEffort    string                 // forwarded output_config.effort, "" = absent
		wantMaxTokens float64
		wantTemp      interface{} // forwarded temperature, nil = absent
	}{
		{
			name: "messages adaptive xhigh", path: "/v1/messages", wantStatus: http.StatusOK,
			body:         `{"model":"` + model + `","max_tokens":1000,"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"},"messages":[{"role":"user","content":"hi"}]}`,
			wantThinking: map[string]interface{}{"type": "adaptive"}, wantEffort: "xhigh", wantMaxTokens: 1000,
		},
		{
			name: "messages effort without thinking", path: "/v1/messages", wantStatus: http.StatusOK,
			body:       `{"model":"` + model + `","max_tokens":1000,"output_config":{"effort":"low"},"messages":[{"role":"user","content":"hi"}]}`,
			wantEffort: "low", wantMaxTokens: 1000,
		},
		{
			name: "chat max", path: "/v1/chat/completions", wantStatus: http.StatusOK,
			body:         `{"model":"` + model + `","max_tokens":1000,"reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`,
			wantThinking: map[string]interface{}{"type": "adaptive"}, wantEffort: "max", wantMaxTokens: 1000,
		},
		{
			name: "chat without effort keeps the model default", path: "/v1/chat/completions", wantStatus: http.StatusOK,
			body:          `{"model":"` + model + `","max_tokens":1000,"messages":[{"role":"user","content":"hi"}]}`,
			wantMaxTokens: 1000,
		},
		{
			name: "responses medium", path: "/v1/responses", wantStatus: http.StatusOK,
			body:         `{"model":"` + model + `","max_output_tokens":1000,"reasoning":{"effort":"medium"},"input":"hi"}`,
			wantThinking: map[string]interface{}{"type": "adaptive"}, wantEffort: "medium", wantMaxTokens: 1000,
		},
		{
			name: "client temperature stands", path: "/v1/messages", wantStatus: http.StatusOK,
			body:         `{"model":"` + model + `","max_tokens":1000,"temperature":0.3,"thinking":{"type":"adaptive","effort":"high"},"messages":[{"role":"user","content":"hi"}]}`,
			wantThinking: map[string]interface{}{"type": "adaptive"}, wantEffort: "high", wantMaxTokens: 1000, wantTemp: 0.3,
		},
		{
			name: "chat temperature stands", path: "/v1/chat/completions", wantStatus: http.StatusOK,
			body:         `{"model":"` + model + `","max_tokens":1000,"temperature":0.3,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`,
			wantThinking: map[string]interface{}{"type": "adaptive"}, wantEffort: "high", wantMaxTokens: 1000, wantTemp: 0.3,
		},
		{
			name: "standard service tier is accepted", path: "/v1/chat/completions", wantStatus: http.StatusOK,
			body:          `{"model":"` + model + `","max_tokens":1000,"service_tier":"auto","messages":[{"role":"user","content":"hi"}]}`,
			wantMaxTokens: 1000,
		},
		{
			name: "messages disabled", path: "/v1/messages", wantStatus: http.StatusBadRequest, wantParam: "thinking.type",
			body: `{"model":"` + model + `","max_tokens":1000,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "messages effort none", path: "/v1/messages", wantStatus: http.StatusBadRequest, wantParam: "output_config.effort",
			body: `{"model":"` + model + `","max_tokens":1000,"output_config":{"effort":"none"},"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "chat effort none", path: "/v1/chat/completions", wantStatus: http.StatusBadRequest, wantParam: "reasoning_effort",
			body: `{"model":"` + model + `","reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "responses effort none", path: "/v1/responses", wantStatus: http.StatusBadRequest, wantParam: "reasoning.effort",
			body: `{"model":"` + model + `","reasoning":{"effort":"none"},"input":"hi"}`,
		},
		{
			name: "priority service tier", path: "/v1/chat/completions", wantStatus: http.StatusBadRequest, wantParam: "service_tier",
			body: `{"model":"` + model + `","service_tier":"priority","messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "priority service tier in extra_body", path: "/v1/messages", wantStatus: http.StatusBadRequest, wantParam: "service_tier",
			body: `{"model":"` + model + `","max_tokens":1000,"extra_body":{"service_tier":"priority"},"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			// The key with its first letter spelled as a JSON unicode escape.
			name: "escaped service tier key in extra_body", path: "/v1/chat/completions", wantStatus: http.StatusBadRequest, wantParam: "service_tier",
			body: `{"model":"` + model + `","extra_body":{"` + "\x5cu0073" + `ervice_tier":"priority"},"messages":[{"role":"user","content":"hi"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			var outbound map[string]interface{}
			var outboundBeta string
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				require.Equal(t, "/v1/messages", r.URL.Path)
				outboundBeta = r.Header.Get("anthropic-beta")
				require.NoError(t, json.NewDecoder(r.Body).Decode(&outbound))
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"`+model+`",`+
					`"content":[{"type":"thinking","thinking":"t","signature":"s"},{"type":"text","text":"ok"}],"stop_reason":"end_turn",`+
					`"usage":{"input_tokens":10,"output_tokens":5,"output_tokens_details":{"thinking_tokens":3}}}`)
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("minimax", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
				WithMasterKey("master-key").
				Build()
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()

			prx.ProxyRequest(recorder, req)

			require.Equal(t, tt.wantStatus, recorder.Code, recorder.Body.String())
			if tt.wantStatus != http.StatusOK {
				assert.Zero(t, calls.Load(), "a rejected request must not reach the upstream")
				var errorBody struct {
					Error struct {
						Param string `json:"param"`
					} `json:"error"`
				}
				require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &errorBody), recorder.Body.String())
				assert.Equal(t, tt.wantParam, errorBody.Error.Param, recorder.Body.String())
				return
			}
			require.EqualValues(t, 1, calls.Load())
			assert.NotContains(t, outbound, "service_tier")
			if tt.wantThinking == nil {
				assert.NotContains(t, outbound, "thinking")
			} else {
				assert.Equal(t, tt.wantThinking, outbound["thinking"])
			}
			if tt.wantEffort == "" {
				assert.NotContains(t, outbound, "output_config")
			} else {
				assert.Equal(t, map[string]interface{}{"effort": tt.wantEffort}, outbound["output_config"])
			}
			assert.Equal(t, tt.wantMaxTokens, outbound["max_tokens"])
			// Claude-only side effects of thinking stay off.
			assert.Empty(t, outboundBeta)
			assert.NotContains(t, outbound, "anthropic_beta")
			if tt.wantTemp == nil {
				assert.NotContains(t, outbound, "temperature")
			} else {
				assert.Equal(t, tt.wantTemp, outbound["temperature"])
			}
		})
	}
}

// The service tier is checked against the model the credential serves, not the name the
// client sent: behind an alias the router would otherwise strip a priority tier the model
// lacks and run the request at the standard tier without a word.
func TestNativeAdaptiveModelServiceTierBehindAlias(t *testing.T) {
	const alias = "fast-chat"
	tests := []struct {
		name       string
		model      config.ModelRPMConfig
		body       string
		wantStatus int
	}{
		{
			name:       "models real name",
			model:      config.ModelRPMConfig{Name: alias, Model: "MiniMax-M3.1-Flash", RPM: 100, TPM: -1},
			body:       `{"model":"` + alias + `","service_tier":"priority","messages":[{"role":"user","content":"hi"}]}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "per-credential real name, tier in extra_body",
			model:      config.ModelRPMConfig{Name: alias, Model: "MiniMax-M3.1-Flash", Credential: "minimax", RPM: 100, TPM: -1},
			body:       `{"model":"` + alias + `","extra_body":{"service_tier":"flex"},"messages":[{"role":"user","content":"hi"}]}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "standard tier passes",
			model:      config.ModelRPMConfig{Name: alias, Model: "MiniMax-M3.1-Flash", RPM: 100, TPM: -1},
			body:       `{"model":"` + alias + `","max_tokens":1000,"service_tier":"auto","messages":[{"role":"user","content":"hi"}]}`,
			wantStatus: http.StatusOK,
		},
		// Regression guard: an alias whose own name looks native adaptive does not
		// reject a tier for the model it really reaches.
		{
			name:       "alias named like the model",
			model:      config.ModelRPMConfig{Name: "minimax-m3.1-flash-alias", Model: "claude-sonnet-4-6", RPM: 100, TPM: -1},
			body:       `{"model":"minimax-m3.1-flash-alias","max_tokens":1000,"service_tier":"priority","messages":[{"role":"user","content":"hi"}]}`,
			wantStatus: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			var outbound map[string]interface{}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				require.NoError(t, json.NewDecoder(r.Body).Decode(&outbound))
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m",`+
					`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
			}))
			defer upstream.Close()

			modelManager := models.New(testhelpers.NewTestLogger(), 50, []config.ModelRPMConfig{tt.model})
			builder := NewTestProxyBuilder().
				WithSingleCredential("minimax", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
				WithMasterKey("master-key")
			modelManager.LoadModelsFromConfig(builder.config.Credentials)
			builder.config.ModelManager = modelManager
			prx := builder.Build()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tt.body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()

			prx.ProxyRequest(recorder, req)

			require.Equal(t, tt.wantStatus, recorder.Code, recorder.Body.String())
			if tt.wantStatus != http.StatusOK {
				assert.Zero(t, calls.Load(), "a rejected request must not reach the upstream")
				assert.Contains(t, recorder.Body.String(), `"param":"service_tier"`)
				return
			}
			require.EqualValues(t, 1, calls.Load())
			assert.Equal(t, tt.model.Model, outbound["model"])
			assert.NotContains(t, outbound, "service_tier")
		})
	}
}
