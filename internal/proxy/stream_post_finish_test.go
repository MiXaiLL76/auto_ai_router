package proxy

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	postFinishTerminal = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"final\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"
	postFinishLate     = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"private-arguments\"}}]},\"finish_reason\":null}]}\n\n"
	postFinishDone     = "data: [DONE]\r\n\r\n"
)

func TestChatPostFinishStreamToClient(t *testing.T) {
	p := NewTestProxyBuilder().Build()
	var logs bytes.Buffer
	p.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn}))
	w := httptest.NewRecorder()
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("ETag", "upstream")
	w.Header().Set("Content-Length", "999")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{RequestID: "req1", Request: req}
	first, rest := postFinishLate[:70], postFinishLate[70:]
	mixed := "event: message\r\ndata: {\"choices\":[\r\ndata: {\"index\":0,\"delta\":{\"content\":\"private-text\"}},{\"index\":1,\"delta\":{\"content\":\"kept\"},\"finish_reason\":\"stop\"}]}\r\n\r\n"
	usage := "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"},{\"index\":1,\"delta\":{\"reasoning_content\":\"private-reasoning\"}}],\"usage\":{\"total_tokens\":9}}\n\n"
	source := io.MultiReader(strings.NewReader(postFinishTerminal+first), strings.NewReader(rest+mixed+usage+postFinishDone))
	var observed bytes.Buffer
	err := p.streamToClient(req.Context(), w, source, "cred", "model", req.URL.Path, http.StatusOK, func(chunk []byte) {
		_, _ = observed.Write(chunk)
	}, nil, logCtx)
	require.NoError(t, err)
	require.Equal(t, postFinishTerminal+
		"event: message\r\ndata: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"kept\"},\"finish_reason\":\"stop\"}]}\r\n\r\n"+
		"data: {\"choices\":[],\"usage\":{\"total_tokens\":9}}\n\n"+postFinishDone, w.Body.String())
	require.Equal(t, postFinishTerminal+postFinishLate+mixed+usage+postFinishDone, observed.String())
	require.Empty(t, w.Result().Header.Get("ETag"))
	require.Empty(t, w.Result().Header.Get("Content-Length"))
	lines := strings.Split(strings.TrimSpace(logs.String()), "\n")
	require.Len(t, lines, 2)
	require.Contains(t, lines[0], `"level":"WARN"`)
	require.Contains(t, lines[0], `"request_id":"req1"`)
	require.Contains(t, lines[0], `"model":"model"`)
	require.Contains(t, lines[0], `"choice_index":0`)
	require.Contains(t, lines[0], `"finish_reason":"tool_calls"`)
	require.Contains(t, lines[0], `"late_fields":["finish_reason","tool_calls"]`)
	require.Contains(t, lines[1], `"choice_index":1`)
	require.Contains(t, lines[1], `"late_fields":["reasoning_content"]`)
	require.NotContains(t, logs.String(), "private-")
}

func TestChatPostFinishFilterScope(t *testing.T) {
	p := NewTestProxyBuilder().Build()
	for _, tt := range []struct {
		name, path, contentType string
		status                  int
	}{
		{"responses", "/v1/responses", "text/event-stream", http.StatusOK},
		{"messages", "/v1/messages", "text/event-stream", http.StatusOK},
		{"non SSE", "/v1/chat/completions", "application/stream+json", http.StatusOK},
		{"non success", "/v1/chat/completions", "text/event-stream", http.StatusBadRequest},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tt.path, nil)
			w := httptest.NewRecorder()
			w.Header().Set("Content-Type", tt.contentType)
			body := postFinishTerminal + postFinishLate + postFinishDone
			require.NoError(t, p.streamToClient(req.Context(), w, strings.NewReader(body), "cred", "model", req.URL.Path, tt.status, nil, nil, nil))
			require.Equal(t, body, w.Body.String())
		})
	}
}

type nonFlushingChatWriter struct{ http.ResponseWriter }

func TestChatPostFinishProxyStreamingResponse(t *testing.T) {
	p := NewTestProxyBuilder().Build()
	for _, name := range []string{"flushing", "non-flushing", "tool replay"} {
		t.Run(name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			recorder := httptest.NewRecorder()
			var w http.ResponseWriter = recorder
			body := postFinishTerminal + postFinishLate + postFinishDone
			if name == "tool replay" {
				p.replayToolStream(w, req, "model", nil, []byte(body))
			} else {
				if name == "non-flushing" {
					w = nonFlushingChatWriter{w}
				}
				resp := &ProxyResponse{
					Headers:    http.Header{"Content-Type": {"text/event-stream"}},
					StatusCode: http.StatusOK,
					StreamBody: io.NopCloser(strings.NewReader(body)),
				}
				_, err := p.writeProxyStreamingResponseWithTokens(w, resp, req, nil, "model", "model", nil)
				require.NoError(t, err)
			}
			require.Equal(t, postFinishTerminal+postFinishDone, recorder.Body.String())
		})
	}
}

func TestChatPostFinishUsageOrdering(t *testing.T) {
	const usage = "data: {\"choices\":[],\"id\":\"chat-1\",\"usage\":{\"prompt_tokens\":26,\"completion_tokens\":242,\"total_tokens\":268}}\n\n"
	const terminal = "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\",\"delta\":{\"content\":\"last\"}}],\"id\":\"chat-1\"}\n\n"
	const combined = "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\",\"delta\":{\"content\":\"last\"}}],\"id\":\"chat-1\",\"usage\":{\"prompt_tokens\":26,\"completion_tokens\":242,\"total_tokens\":268}}"
	p := NewTestProxyBuilder().Build()
	for _, tt := range []struct{ name, input, want string }{
		{"vLLM separate usage", terminal + usage + postFinishDone, terminal + usage + postFinishDone},
		{"combined", combined + "\n\n" + postFinishLate + postFinishDone, terminal + usage + postFinishDone},
		{"combined EOF tail", combined, terminal + usage},
		{"combined CRLF", combined + "\r\n\r\n" + postFinishDone, strings.ReplaceAll(terminal, "\n", "\r\n") + usage + postFinishDone},
		{"without usage", terminal + postFinishDone, terminal + postFinishDone},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			w := httptest.NewRecorder()
			w.Header().Set("Content-Type", "text/event-stream")
			reader := io.MultiReader(strings.NewReader(tt.input[:55]), strings.NewReader(tt.input[55:]))
			require.NoError(t, p.streamToClient(req.Context(), w, reader, "cred", "model", req.URL.Path, http.StatusOK, nil, nil, nil))
			require.Equal(t, tt.want, w.Body.String())
		})
	}
}
