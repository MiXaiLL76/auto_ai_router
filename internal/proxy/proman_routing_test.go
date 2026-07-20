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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnsupportedProManRequest(t *testing.T) {
	tests := []struct {
		name  string
		model string
		body  string
		want  string
	}{
		{
			name:  "adaptive thinking nested effort",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"thinking":{"type":"adaptive","effort":"high"}}`,
			want:  "thinking",
		},
		{
			name:  "reasoning effort rejected for unsupported model",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"reasoning_effort":"high"}`,
			want:  "thinking",
		},
		{
			name:  "reasoning effort allowed for supported model",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"reasoning_effort":"high"}`,
			want:  "",
		},
		{
			name:  "responses reasoning effort rejected for unsupported model",
			model: "claude-sonnet-5",
			body:  `{"model":"claude-sonnet-5","input":"hi","reasoning":{"effort":"medium"}}`,
			want:  "thinking",
		},
		{
			name:  "disabled thinking allowed",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"thinking":{"type":"disabled"}}`,
			want:  "",
		},
		{
			name:  "context management",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"context_management":{"edits":[{"type":"compact_20260112"}]}}`,
			want:  "context_management",
		},
		{
			name:  "tool choice none disable parallel",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"tool_choice":{"type":"none","disable_parallel_tool_use":true}}`,
			want:  "tool_choice.none.disable_parallel_tool_use",
		},
		{
			name:  "tool choice none without disable allowed",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"tool_choice":{"type":"none"}}`,
			want:  "",
		},
		{
			name:  "server tool use history",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`,
			want:  "server_tool_use",
		},
		{
			name:  "text plain document",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"text/plain","data":"SGk="}}]}]}`,
			want:  "document.text_plain",
		},
		{
			name:  "assistant prefill rejected for model",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"pre"}]}`,
			want:  "assistant_prefill",
		},
		{
			name:  "assistant prefill allowed for model",
			model: "claude-sonnet-4.5",
			body:  `{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"pre"}]}`,
			want:  "",
		},
		{
			name:  "temperature top p rejected for all known models",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"temperature":0.2,"top_p":0.9}`,
			want:  "temperature+top_p",
		},
		{
			name:  "top p rejected for deprecated sampling model",
			model: "claude-sonnet-5",
			body:  `{"model":"claude-sonnet-5","messages":[],"top_p":0.9}`,
			want:  "top_p",
		},
		{
			name:  "top k rejected for deprecated sampling model",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"top_k":10}`,
			want:  "top_k",
		},
		{
			name:  "temperature one allowed for deprecated sampling model",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"temperature":1}`,
			want:  "",
		},
		{
			name:  "low temperature rejected for deprecated sampling model",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"temperature":0.2}`,
			want:  "temperature",
		},
		{
			name:  "basic request allowed",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"hi"}]}`,
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, unsupportedProManRequest([]byte(tt.body), tt.model))
		})
	}
}

func TestProxyRequest_UnsupportedProManRequestRoutesToNextPrimary(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	var nextPrimaryCalls int32
	nextPrimary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&nextPrimaryCalls, 1)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer next-key", r.Header.Get("Authorization"))
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		assert.Contains(t, string(body), `"reasoning_effort":"high"`)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "chatcmpl-next",
			"object": "chat.completion",
			"choices": []map[string]any{{
				"message": map[string]string{"role": "assistant", "content": "next primary"},
			}},
		})
	}))
	defer nextPrimary.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan, BaseURL: promanUpstream.URL, APIKey: "proman-key", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "next-primary", Type: config.ProviderTypeProxy, BaseURL: nextPrimary.URL, APIKey: "next-key", RPM: 100, TPM: 10000},
		).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "next primary")
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
	assert.Equal(t, int32(1), atomic.LoadInt32(&nextPrimaryCalls))
}

func TestProxyRequest_UnsupportedProManRequestRoutesToFallbackProxy(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	var fallbackCalls int32
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "chatcmpl-fallback",
			"object": "chat.completion",
			"choices": []map[string]any{{
				"message": map[string]string{"role": "assistant", "content": "fallback ok"},
			}},
		})
	}))
	defer fallback.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan, BaseURL: promanUpstream.URL, APIKey: "proman-key", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "fallback", Type: config.ProviderTypeProxy, BaseURL: fallback.URL, APIKey: "fallback-key", RPM: 100, TPM: 10000, IsFallback: true},
		).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"2+2"}],"tool_choice":{"type":"none","disable_parallel_tool_use":true}}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "fallback ok")
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallbackCalls))
}

func TestProxyRequest_UnsupportedProManRequestWithoutFallbackReturnsLocalError(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("proman", config.ProviderTypeProMan, promanUpstream.URL, "proman-key").
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-opus-4-8","messages":[{"role":"user","content":"summarize"}],"context_management":{"edits":[{"type":"compact_20260112"}]}}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), unsupportedProManRequestMessage)
	assert.NotContains(t, strings.ToLower(w.Body.String()), "proman")
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
}
