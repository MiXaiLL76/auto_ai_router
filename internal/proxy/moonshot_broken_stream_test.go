package proxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/toolloop"
)

// A provider's stream can break part-way: the connection dies after the
// model has already streamed most of its answer. Measured in production
// against Moonshot and DeepSeek, whose streams break at a fixed 30 s and
// 45 s (see llm-router-org/infra#1538).
//
// Before the fix the router read the provider's stream whole, threw the
// bytes that had arrived away with the read error, and answered the client
// with a bare 502 -- the answer the model had already produced was lost.
// The client now gets the partial stream, which is what the provider sent.

// brokenStreamHandler answers a chat request with a stream that ends
// mid-way: the chunks are written and the connection is then cut, so the
// client's read fails with an unexpected EOF.
func brokenStreamHandler(t *testing.T, answer string, chunks int) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("the test server does not support flushing")
		}
		for i := 0; i < chunks; i++ {
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"` + answer + `"}}]}` + "\n\n"))
			flusher.Flush()
		}
		// The usage frame never arrives and the connection dies: the
		// client's read fails with an unexpected EOF.
		if hijacker, ok := w.(http.Hijacker); ok {
			conn, _, err := hijacker.Hijack()
			if err == nil {
				_ = conn.Close()
			}
		}
	}
}

// TestMoonshotBrokenStreamGivesTheClientTheAnswerItGot pins the fix for the
// broken provider stream on the Moonshot path: the client gets the text the
// model produced before the connection died, not a bare 502.
func TestMoonshotBrokenStreamGivesTheClientTheAnswerItGot(t *testing.T) {
	fake := &moonshotFake{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			brokenStreamHandler(t, "the answer before the break", 5)(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	if !strings.Contains(out, "the answer before the break") {
		t.Errorf("the client did not get the answer the provider had already sent: %s", out)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
}

// TestMoonshotBrokenStreamWithNoBytesIsAnError covers the other end: a
// provider that breaks before sending anything has no answer to hand over,
// so the client still gets the failure.
func TestMoonshotBrokenStreamWithNoBytesIsAnError(t *testing.T) {
	fake := &moonshotFake{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			_, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if hijacker, ok := w.(http.Hijacker); ok {
				conn, _, err := hijacker.Hijack()
				if err == nil {
					_ = conn.Close()
				}
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, _ := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code == http.StatusOK {
		t.Errorf("a stream that broke before sending anything must not be a 200: %s", rec.Body.String())
	}
}

// TestDeepSeekBrokenStreamGivesTheClientTheAnswerItGot is the DeepSeek
// twin of the test above: the same fix, the other adapter.
func TestDeepSeekBrokenStreamGivesTheClientTheAnswerItGot(t *testing.T) {
	server := httptest.NewServer(brokenStreamHandler(t, "the answer before the break", 5))
	defer server.Close()

	credential := config.CredentialConfig{
		Name:          "deepseek",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       server.URL,
		APIKey:        "ds-key",
		RPM:           100,
		TPM:           100000,
		DeepSeekTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{
		Name:                 "deepseek-flash",
		Model:                "deepseek-flash",
		Credential:           "deepseek",
		PassthroughResponses: &noPassthrough,
	}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "deepseek", Type: config.ProviderTypeOpenAI, APIKey: "ds-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"deepseek-flash": {
		InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6,
	}})

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	if !strings.Contains(out, "the answer before the break") {
		t.Errorf("the client did not get the answer the provider had already sent: %s", out)
	}
}

// TestCollectStreamToolCallsKeepsTheBytesOnABrokenStream pins the change in
// the toolloop package itself: the collector hands the bytes that arrived
// back with the error, so a caller can never mistake a broken stream for an
// empty one.
func TestCollectStreamToolCallsKeepsTheBytesOnABrokenStream(t *testing.T) {
	body := &partialThenBrokenBody{data: []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")}
	resp := &http.Response{Body: body, StatusCode: http.StatusOK}

	raw, calls, _, err := toolloop.CollectStreamToolCalls(resp, nil)
	if err == nil {
		t.Fatal("a broken stream must return an error")
	}
	if len(raw) == 0 {
		t.Error("the bytes that arrived must be returned with the error")
	}
	if len(calls) != 0 {
		t.Errorf("calls = %v, want none: the partial stream carries no tool call", calls)
	}
}

// partialThenBrokenBody serves its bytes and then fails, the way a provider's
// connection dies mid-stream.
type partialThenBrokenBody struct {
	data []byte
	off  int
}

func (b *partialThenBrokenBody) Read(p []byte) (int, error) {
	if b.off >= len(b.data) {
		return 0, io.ErrUnexpectedEOF
	}
	n := copy(p, b.data[b.off:])
	b.off += n
	return n, nil
}

func (b *partialThenBrokenBody) Close() error { return nil }

// TestBrokenStreamIsBilledForTheAnswer pins the billing of a broken stream:
// the client got the answer, so the spend row must carry its tokens. A row of
// zeros would bill the answer for nothing.
func TestBrokenStreamIsBilledForTheAnswer(t *testing.T) {
	server := httptest.NewServer(brokenStreamHandler(t, "the answer before the break", 5))
	defer server.Close()

	credential := config.CredentialConfig{
		Name:          "deepseek",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       server.URL,
		APIKey:        "ds-key",
		RPM:           100,
		TPM:           100000,
		DeepSeekTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{
		Name:                 "deepseek-flash",
		Model:                "deepseek-flash",
		Credential:           "deepseek",
		PassthroughResponses: &noPassthrough,
	}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "deepseek", Type: config.ProviderTypeOpenAI, APIKey: "ds-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"deepseek-flash": {
		InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6,
	}})

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require := len(dbStub.loggedEntries)
	if require != 1 {
		t.Fatalf("got %d spend rows, want 1", require)
	}
	entry := dbStub.loggedEntries[0]
	if entry.PromptTokens == 0 {
		t.Error("the prompt tokens are zero: the request was not billed for its input")
	}
	if entry.CompletionTokens == 0 {
		t.Error("the completion tokens are zero: the answer the client got was billed for nothing")
	}
	if entry.Spend == 0 {
		t.Error("the spend is zero: the answer the client got was free")
	}
}

// TestBrokenStreamBeforeTheToolCallNameIsNotRelayed covers the tool call
// whose name never arrived: the stream broke between the piece carrying the
// name and the pieces carrying its arguments, so the call is nameless. A
// nameless call cannot be told apart from one of the router's own -- and one
// of the router's own must never reach the client, which never declared it.
func TestBrokenStreamBeforeTheToolCallNameIsNotRelayed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("the test server does not support flushing")
		}
		// The tool call without its name, then some text, then the break.
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"arguments":"{\"a\""}}]}}]}` + "\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"partial text"}}]}` + "\n\n"))
		flusher.Flush()
		if hijacker, ok := w.(http.Hijacker); ok {
			conn, _, err := hijacker.Hijack()
			if err == nil {
				_ = conn.Close()
			}
		}
	}))
	defer server.Close()

	credential := config.CredentialConfig{
		Name:          "deepseek",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       server.URL,
		APIKey:        "ds-key",
		RPM:           100,
		TPM:           100000,
		DeepSeekTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{
		Name:                 "deepseek-flash",
		Model:                "deepseek-flash",
		Credential:           "deepseek",
		PassthroughResponses: &noPassthrough,
	}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "deepseek", Type: config.ProviderTypeOpenAI, APIKey: "ds-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"deepseek-flash": {
		InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6,
	}})

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	out := rec.Body.String()
	if strings.Contains(out, "tool_calls") {
		t.Errorf("the client saw a tool call it never declared: %s", out)
	}
	if rec.Code == http.StatusOK {
		t.Errorf("a stream with an unrecognised tool call must not be a 200: %s", out)
	}
}

// TestClientGoneDuringBufferingIsNotAProviderFailure covers the other way the
// read can fail: the client went away. The upstream request rode the client's
// context, so its read fails with the cancellation -- and that is not a
// broken provider, it is nobody left to answer.
func TestClientGoneDuringBufferingIsNotAProviderFailure(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("the test server does not support flushing")
		}
		for i := 0; i < 3; i++ {
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"chunk"}}]}` + "\n\n"))
			flusher.Flush()
		}
		<-release
		if hijacker, ok := w.(http.Hijacker); ok {
			conn, _, err := hijacker.Hijack()
			if err == nil {
				_ = conn.Close()
			}
		}
	}))
	defer server.Close()

	credential := config.CredentialConfig{
		Name:          "deepseek",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       server.URL,
		APIKey:        "ds-key",
		RPM:           100,
		TPM:           100000,
		DeepSeekTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{
		Name:                 "deepseek-flash",
		Model:                "deepseek-flash",
		Credential:           "deepseek",
		PassthroughResponses: &noPassthrough,
	}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "deepseek", Type: config.ProviderTypeOpenAI, APIKey: "ds-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"deepseek-flash": {
		InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6,
	}})

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	go func() {
		time.Sleep(150 * time.Millisecond)
		cancel()
	}()
	time.AfterFunc(300*time.Millisecond, func() { close(release) })

	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if entry.Status != "failure" {
		t.Errorf("status = %q, want failure", entry.Status)
	}
}
