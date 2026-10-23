package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/iotest"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/loopgate"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
)

// loopGateWords returns n distinct words.
func loopGateWords(n int) []string {
	words := make([]string, n)
	for i := range words {
		words[i] = fmt.Sprintf("w%d", i)
	}
	return words
}

// loopGateLoop returns a phrase of 40 distinct words repeated times times.
func loopGateLoop(times int) []string {
	var words []string
	for range times {
		for i := range 40 {
			words = append(words, fmt.Sprintf("loop%d", i))
		}
	}
	return words
}

// newLoopGateVLLM serves a chat completion stream: reasoning words, then content
// words, ten words per chunk, then finish_reason and usage.
func newLoopGateVLLM(t *testing.T, reasoning, content []string, finishReason string) string {
	t.Helper()
	srv := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		emit := func(field string, words []string) {
			for i := 0; i < len(words); i += 10 {
				text, _ := json.Marshal(strings.Join(words[i:min(i+10, len(words))], " ") + " ")
				_, _ = fmt.Fprintf(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{\"%s\":%s},\"finish_reason\":null}]}\n\n", field, text)
			}
		}
		emit("reasoning_content", reasoning)
		emit("content", content)
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}]}\n\n", finishReason)
		_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5000,\"total_tokens\":5010}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func newLoopGateProxy(t *testing.T, credType config.ProviderType, upstreamURL string, credOverride, modelOverride *config.LoopGateOverride) *Proxy {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	creds := []config.CredentialConfig{
		{Name: "lg-cred", Type: credType, BaseURL: upstreamURL, APIKey: "k", RPM: 100, TPM: 10000000, LoopGate: credOverride},
	}
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "qwen", Credential: "lg-cred", RPM: -1, TPM: -1, LoopGate: modelOverride},
	})
	manager.LoadModelsFromConfig(creds)

	gate := config.LoopGateConfig{LoopGateSettings: config.LoopGateSettings{
		Mode: config.LoopGateObserve, ArmWords: config.DefaultLoopGateArmWords,
	}}
	gate.ApplyDefaults()

	builder := NewTestProxyBuilder().WithCredentials(creds...)
	builder.config.ModelManager = manager
	builder.config.Metrics = monitoring.New(true)
	builder.config.LoopGate = gate
	return builder.Build()
}

func resetLoopGateMetrics() {
	monitoring.LoopGateStreamsTotal.Reset()
	monitoring.LoopGateFiringsTotal.Reset()
	monitoring.LoopGateWordsAfterFiringTotal.Reset()
}

const loopGateStreamRequest = `{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`

func TestLoopGate_ObserveFiresOnLoopingReasoningAndLeavesStreamIntact(t *testing.T) {
	resetLoopGateMetrics()
	reasoning := append(loopGateWords(500), loopGateLoop(100)...) // 4500 words
	upstream := newLoopGateVLLM(t, reasoning, loopGateWords(50), "length")
	prx := newLoopGateProxy(t, config.ProviderTypeVLLM, upstream, nil, nil)

	w := postChat(t, prx, loopGateStreamRequest)

	require.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	assert.Contains(t, body, "data: [DONE]", "observe mode never cuts the stream")
	assert.Contains(t, body, `"finish_reason":"length"`)
	assert.Equal(t, 4500/10+50/10, strings.Count(body, `"reasoning_content"`)+strings.Count(body, `"content":"w`),
		"every upstream delta reaches the client")

	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateStreamsTotal.WithLabelValues("lg-cred", "qwen", "observe")))
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "observe")))
	assert.Zero(t, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "content", "observe")))
	assert.Equal(t, float64(4500-config.DefaultLoopGateArmWords),
		testutil.ToFloat64(monitoring.LoopGateWordsAfterFiringTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "observe")),
		"fires on the first armed word, everything after it is counted")
}

func TestLoopGate_HealthyLongAnswerDoesNotFire(t *testing.T) {
	resetLoopGateMetrics()
	upstream := newLoopGateVLLM(t, loopGateWords(4000), loopGateWords(6000), "stop")
	prx := newLoopGateProxy(t, config.ProviderTypeVLLM, upstream, nil, nil)

	w := postChat(t, prx, loopGateStreamRequest)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateStreamsTotal.WithLabelValues("lg-cred", "qwen", "observe")))
	assert.Zero(t, testutil.CollectAndCount(monitoring.LoopGateFiringsTotal))
}

func TestLoopGate_ScopedByProviderAndOverrides(t *testing.T) {
	looping := append(loopGateWords(500), loopGateLoop(100)...)
	on := config.LoopGateObserve
	off := config.LoopGateOff
	cases := []struct {
		name          string
		credType      config.ProviderType
		credOverride  *config.LoopGateOverride
		modelOverride *config.LoopGateOverride
		watched       bool
	}{
		{name: "other provider is not watched", credType: config.ProviderTypeOpenAI},
		{name: "credential mode opts another provider in", credType: config.ProviderTypeOpenAI,
			credOverride: &config.LoopGateOverride{Mode: &on}, watched: true},
		{name: "model switches it off", credType: config.ProviderTypeVLLM,
			modelOverride: &config.LoopGateOverride{Mode: &off}},
		{name: "model wins over credential", credType: config.ProviderTypeVLLM,
			credOverride: &config.LoopGateOverride{Mode: &off}, modelOverride: &config.LoopGateOverride{Mode: &on}, watched: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resetLoopGateMetrics()
			upstream := newLoopGateVLLM(t, looping, nil, "length")
			prx := newLoopGateProxy(t, tc.credType, upstream, tc.credOverride, tc.modelOverride)

			w := postChat(t, prx, loopGateStreamRequest)

			require.Equal(t, http.StatusOK, w.Code)
			want := 0.0
			if tc.watched {
				want = 1
			}
			assert.Equal(t, want, testutil.ToFloat64(monitoring.LoopGateStreamsTotal.WithLabelValues("lg-cred", "qwen", "observe")))
			assert.Equal(t, want, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "observe")))
		})
	}
}

func TestLoopGate_ModelThresholdOverride(t *testing.T) {
	resetLoopGateMetrics()
	// 2000 looping words: below the default arm_words, above the model's.
	upstream := newLoopGateVLLM(t, loopGateLoop(50), nil, "length")
	armWords := 1000
	prx := newLoopGateProxy(t, config.ProviderTypeVLLM, upstream, nil, &config.LoopGateOverride{ArmWords: &armWords})

	postChat(t, prx, loopGateStreamRequest)

	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "observe")))
}

func newLoopGateProxyWithGate(t *testing.T, upstreamURL string, gate config.LoopGateConfig) *Proxy {
	t.Helper()
	return newLoopGateProxyWithModel(t, upstreamURL, gate, config.ModelRPMConfig{})
}

// newLoopGateProxyWithModel is newLoopGateProxyWithGate with extra model
// settings (Name, Credential and Model are filled in).
func newLoopGateProxyWithModel(t *testing.T, upstreamURL string, gate config.LoopGateConfig, model config.ModelRPMConfig) *Proxy {
	t.Helper()
	return newLoopGateProxyWith(t, config.CredentialConfig{BaseURL: upstreamURL}, gate, model, 0)
}

// newLoopGateProxyWith builds the loop gate test proxy from cred (Name, Type
// and limits are filled in) and, when requestTimeout is not 0, that
// request_timeout.
func newLoopGateProxyWith(t *testing.T, cred config.CredentialConfig, gate config.LoopGateConfig, model config.ModelRPMConfig, requestTimeout time.Duration) *Proxy {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	cred.Name, cred.Type, cred.APIKey, cred.RPM, cred.TPM = "lg-cred", config.ProviderTypeVLLM, "k", 100, 10000000
	creds := []config.CredentialConfig{cred}
	model.Name, model.Credential, model.Model, model.RPM, model.TPM = "qwen", "lg-cred", "Qwen/Qwen3.6-35B", -1, -1
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{model})
	manager.LoadModelsFromConfig(creds)
	gate.ApplyDefaults()

	builder := NewTestProxyBuilder().WithCredentials(creds...)
	builder.config.ModelManager = manager
	builder.config.Metrics = monitoring.New(true)
	builder.config.LoopGate = gate
	if requestTimeout != 0 {
		builder.config.RequestTimeout = requestTimeout
	}
	return builder.Build()
}

func enforceGate(cutMinRatio float64, maxAttempts int) config.LoopGateConfig {
	return config.LoopGateConfig{LoopGateSettings: config.LoopGateSettings{
		Mode: config.LoopGateEnforce, ArmWords: config.DefaultLoopGateArmWords, CutMinRatio: cutMinRatio,
		MaxAttempts: maxAttempts,
	}}
}

// loopGateSSEChunks parses the data payloads of a client-visible SSE body.
func loopGateSSEChunks(t *testing.T, body string) (chunks []map[string]any, done bool) {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if payload == "[DONE]" {
			done = true
			continue
		}
		var chunk map[string]any
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk), payload)
		chunks = append(chunks, chunk)
	}
	return chunks, done
}

// loopGateCutTail returns the closing chunk of a cut chat stream, checking
// that the usage chunk (choices: []) follows it.
func loopGateCutTail(t *testing.T, chunks []map[string]any) map[string]any {
	t.Helper()
	require.GreaterOrEqual(t, len(chunks), 2)
	usage := chunks[len(chunks)-1]
	assert.Empty(t, usage["choices"], "the usage chunk follows the closing chunk")
	u, ok := usage["usage"].(map[string]any)
	require.True(t, ok, "usage chunk: %v", usage)
	assert.Positive(t, u["completion_tokens"])
	assert.Equal(t, u["prompt_tokens"].(float64)+u["completion_tokens"].(float64), u["total_tokens"])
	return chunks[len(chunks)-2]
}

func TestLoopGate_EnforceCutsLoopingStream(t *testing.T) {
	resetLoopGateMetrics()
	monitoring.LoopGateCutsTotal.Reset()
	reasoning := append(loopGateWords(500), loopGateLoop(100)...) // 4500 words
	upstream := newLoopGateVLLM(t, reasoning, loopGateWords(50), "stop")
	prx := newLoopGateProxyWithGate(t, upstream, enforceGate(0, 1))

	w := postChat(t, prx, loopGateStreamRequest)

	require.Equal(t, http.StatusOK, w.Code)
	chunks, done := loopGateSSEChunks(t, w.Body.String())
	assert.True(t, done, "the cut stream still ends with [DONE]")
	require.NotEmpty(t, chunks)

	// Everything up to and including the chunk that reached the threshold, then
	// the closing chunk: no further reasoning, no content.
	assert.Len(t, chunks, config.DefaultLoopGateArmWords/10+2)
	assert.NotContains(t, w.Body.String(), `"content":"w`)

	last := loopGateCutTail(t, chunks)
	assert.Equal(t, "c1", last["id"])
	assert.Equal(t, "qwen", last["model"], "the closing chunk carries the client-visible model")
	choices := last["choices"].([]any)
	require.Len(t, choices, 1)
	assert.Equal(t, "length", choices[0].(map[string]any)["finish_reason"])
	marker := last["air_loop_gate"].(map[string]any)
	assert.Equal(t, "cut", marker["action"])
	assert.Equal(t, "reasoning", marker["channel"])
	assert.Equal(t, float64(config.DefaultLoopGateArmWords), marker["words"])

	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "enforce")))
	assert.Zero(t, testutil.ToFloat64(monitoring.LoopGateWordsAfterFiringTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "enforce")),
		"cut at the firing: nothing generated after it reaches the client")
}

func TestLoopGate_EnforceCutRatioAboveFiring(t *testing.T) {
	resetLoopGateMetrics()
	monitoring.LoopGateCutsTotal.Reset()
	reasoning := append(loopGateWords(500), loopGateLoop(200)...) // 8500 words
	upstream := newLoopGateVLLM(t, reasoning, nil, "length")
	prx := newLoopGateProxyWithGate(t, upstream, enforceGate(0.9, 1))

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	marker := loopGateCutTail(t, chunks)["air_loop_gate"].(map[string]any)
	cutWords := int(marker["words"].(float64))
	assert.Greater(t, cutWords, config.DefaultLoopGateArmWords, "fires at min_ratio, cuts only at cut_min_ratio")
	assert.Less(t, cutWords, 8500)
	assert.GreaterOrEqual(t, marker["ratio"].(float64), 0.9)
	assert.Equal(t, float64(cutWords-config.DefaultLoopGateArmWords),
		testutil.ToFloat64(monitoring.LoopGateWordsAfterFiringTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "enforce")))
}

// The cut must abort the upstream generation, not just hide it from the client.
func TestLoopGate_EnforceAbortsUpstreamGeneration(t *testing.T) {
	resetLoopGateMetrics()
	aborted := make(chan int, 1)
	srv := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		const maxChunks = 200000 // an engine stuck until max_tokens
		for i := range maxChunks {
			if r.Context().Err() != nil {
				aborted <- i
				return
			}
			if _, err := fmt.Fprintf(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s \"},\"finish_reason\":null}]}\n\n", strings.Join(loopGateLoop(1)[:10], " ")); err != nil {
				aborted <- i
				return
			}
			flusher.Flush()
		}
		aborted <- maxChunks
	}))
	t.Cleanup(srv.Close)
	prx := newLoopGateProxyWithGate(t, srv.URL, enforceGate(0, 1))

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	assert.Equal(t, "content", loopGateCutTail(t, chunks)["air_loop_gate"].(map[string]any)["channel"])
	sent := <-aborted
	assert.Less(t, sent, 200000, "the engine stops generating once the stream is cut")
}

func TestLoopGate_EnforceLeavesHealthyStreamIntact(t *testing.T) {
	resetLoopGateMetrics()
	upstream := newLoopGateVLLM(t, loopGateWords(4000), loopGateWords(6000), "stop")
	prx := newLoopGateProxyWithGate(t, upstream, enforceGate(0, 1))

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	assert.True(t, done)
	assert.Len(t, chunks, 4000/10+6000/10+2, "every delta, the finish chunk and the usage chunk")
	assert.NotContains(t, w.Body.String(), "air_loop_gate")
}

func newTestLoopGateObserver(t *testing.T, mode config.LoopGateMode) *loopGateObserver {
	t.Helper()
	gate := config.LoopGateConfig{LoopGateSettings: config.LoopGateSettings{Mode: mode, ArmWords: 100}}
	gate.ApplyDefaults()
	settings, err := gate.Resolve(config.ProviderTypeVLLM)
	require.NoError(t, err)
	o := &loopGateObserver{
		p:        &Proxy{logger: testhelpers.NewTestLogger(), metrics: monitoring.New(false)},
		ctx:      t.Context(),
		settings: settings,
		format:   loopgate.FormatChat,
		now:      time.Now,
		start:    time.Now(),
	}
	o.detector = loopgate.NewDetector(loopgate.Config{
		NGram: settings.NGram, ArmWords: settings.ArmWords, MinRepeat: settings.MinRepeat, MinRatio: settings.MinRatio,
	}, o.start)
	return o
}

func loopGateSSE(words []string) string {
	var b strings.Builder
	for i := 0; i < len(words); i += 10 {
		text, _ := json.Marshal(strings.Join(words[i:min(i+10, len(words))], " ") + " ")
		fmt.Fprintf(&b, "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%s}}]}\n\n", text)
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestLoopGateObserver_ObservePassesBytesThroughOneByteReads(t *testing.T) {
	input := loopGateSSE(loopGateLoop(20))
	o := newTestLoopGateObserver(t, config.LoopGateObserve)

	out, err := io.ReadAll(iotest.OneByteReader(o.wrap(io.NopCloser(strings.NewReader(input)))))

	require.NoError(t, err)
	assert.Equal(t, input, string(out))
	require.NotNil(t, o.detector.Content().Fired(), "lines split across reads are reassembled")
	assert.Equal(t, 800, o.detector.Content().Words())
}

func TestLoopGateObserver_EnforceCutsAtLineBoundaryOneByteReads(t *testing.T) {
	input := loopGateSSE(loopGateLoop(20))
	o := newTestLoopGateObserver(t, config.LoopGateEnforce)
	body := &closeTrackingBody{Reader: strings.NewReader(input)}

	out, err := io.ReadAll(iotest.OneByteReader(o.wrap(body)))

	require.NoError(t, err)
	require.NotNil(t, o.cut)
	assert.True(t, body.closed, "the upstream body is closed at the cut")
	head, tail, found := strings.Cut(string(out), "\n\ndata: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\"")
	require.True(t, found, string(out))
	assert.True(t, strings.HasPrefix(input, head+"\n"), "everything before the closing chunk is an untouched prefix of the upstream")
	fired := o.detector.Content().Fired()
	require.NotNil(t, fired)
	assert.Equal(t, fired.Words, o.cut.words, "with cut_min_ratio = min_ratio the stream is cut at the firing")
	assert.Equal(t, fired.Words, o.detector.Content().Words(), "nothing past the cut line is read")
	tokens := strconv.Itoa(o.cut.words * 4 / 3)
	assert.True(t, strings.HasSuffix(tail, "\"finish_reason\":\"length\"}],\"air_loop_gate\":{\"action\":\"cut\",\"channel\":\"content\",\"words\":"+strconv.Itoa(o.cut.words)+",\"ratio\":"+strconv.FormatFloat(o.cut.ratio, 'g', -1, 64)+",\"attempts\":1}}"+
		"\n\ndata: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":"+tokens+",\"total_tokens\":"+tokens+"}}"+
		"\n\ndata: [DONE]\n\n"), tail)
}

func TestLoopGateObserver_SkipsOversizedLine(t *testing.T) {
	huge := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + strings.Repeat("x ", loopGateMaxLine) + "\"}}]}\n\n"
	input := huge + loopGateSSE(loopGateWords(30))
	o := newTestLoopGateObserver(t, config.LoopGateObserve)

	out, err := io.ReadAll(o.wrap(io.NopCloser(strings.NewReader(input))))

	require.NoError(t, err)
	assert.Equal(t, input, string(out))
	assert.Equal(t, 30, o.detector.Content().Words(), "the oversized line is skipped, the next lines are counted")
}

type closeTrackingBody struct {
	io.Reader
	closed bool
}

func (b *closeTrackingBody) Close() error {
	b.closed = true
	return nil
}

// scriptedVLLM answers the n-th request (0-based) with script(n, w, r) and
// keeps every request body and whether the reader went away mid-response
// (script returns true).
type scriptedVLLM struct {
	mu      sync.Mutex
	bodies  []map[string]any
	aborted []bool
}

func newScriptedVLLM(t *testing.T, script func(n int, w http.ResponseWriter, r *http.Request) bool) (*scriptedVLLM, string) {
	t.Helper()
	sv := &scriptedVLLM{}
	srv := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		sv.mu.Lock()
		n := len(sv.bodies)
		sv.bodies = append(sv.bodies, body)
		sv.aborted = append(sv.aborted, false)
		sv.mu.Unlock()
		if script(n, w, r) {
			sv.mu.Lock()
			sv.aborted[n] = true
			sv.mu.Unlock()
		}
	}))
	t.Cleanup(srv.Close)
	return sv, srv.URL
}

func (sv *scriptedVLLM) snapshot() ([]map[string]any, []bool) {
	sv.mu.Lock()
	defer sv.mu.Unlock()
	return append([]map[string]any(nil), sv.bodies...), append([]bool(nil), sv.aborted...)
}

// writeLoopGateStream writes reasoning and content deltas (ten words a chunk)
// with id, then finish_reason and usage when finish is not "". It stops early
// when the reader goes away, like an engine whose client disconnected, and
// then returns true.
func writeLoopGateStream(w http.ResponseWriter, r *http.Request, id string, reasoning, content []string, finish string) bool {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher := w.(http.Flusher)
	emit := func(field string, words []string) bool {
		for i := 0; i < len(words); i += 10 {
			if r.Context().Err() != nil {
				return false
			}
			text, _ := json.Marshal(strings.Join(words[i:min(i+10, len(words))], " ") + " ")
			if _, err := fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{\"%s\":%s},\"finish_reason\":null}]}\n\n", id, field, text); err != nil {
				return false
			}
			flusher.Flush()
		}
		return true
	}
	if !emit("reasoning_content", reasoning) || !emit("content", content) {
		return true
	}
	if finish == "" {
		return false
	}
	_, _ = fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}]}\n\n", id, finish)
	_, _ = fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":700,\"total_tokens\":710}}\n\n", id)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	return false
}

func resetLoopGateEnforceMetrics() {
	resetLoopGateMetrics()
	monitoring.LoopGateCutsTotal.Reset()
	monitoring.LoopGateRestartsTotal.Reset()
	monitoring.LoopGateRecoveredTotal.Reset()
}

// writeLoopGateEndless streams 500 distinct reasoning words, then the same
// phrase until the reader goes away (an engine stuck until a huge max_tokens).
// Returns true when the reader went away.
func writeLoopGateEndless(w http.ResponseWriter, r *http.Request, id string) bool {
	return writeLoopGateEndlessField(w, r, id, "reasoning_content")
}

// writeLoopGateEndlessField is writeLoopGateEndless on the given delta field.
func writeLoopGateEndlessField(w http.ResponseWriter, r *http.Request, id, field string) bool {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher := w.(http.Flusher)
	words := loopGateWords(500)
	for i := 0; i < 1_000_000; i += 10 {
		if r.Context().Err() != nil {
			return true
		}
		var chunk []string
		if i < len(words) {
			chunk = words[i : i+10]
		} else {
			chunk = loopGateLoop(1)[i%40 : i%40+10]
		}
		text, _ := json.Marshal(strings.Join(chunk, " ") + " ")
		if _, err := fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{%q:%s},\"finish_reason\":null}]}\n\n", id, field, text); err != nil {
			return true
		}
		flusher.Flush()
	}
	return false
}

func TestLoopGate_EnforceRestartsLoopingReasoningInSameStream(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", loopGateWords(200), loopGateWords(300), "stop")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))
	tpmBefore := prx.rateLimiter.GetCurrentTPM("lg-cred")

	w := postChat(t, prx, `{"model":"qwen","stream":true,"temperature":0.6,"messages":[{"role":"user","content":"hi"}]}`)

	require.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	chunks, done := loopGateSSEChunks(t, body)
	require.True(t, done)
	assert.Equal(t, 1, strings.Count(body, "data: [DONE]"))
	assert.NotContains(t, body, "air_loop_gate", "a recovered stream is not marked")

	// Attempt 1 up to the line that tipped the loop, then all of attempt 2.
	assert.Len(t, chunks, config.DefaultLoopGateArmWords/10+200/10+300/10+2)
	assert.Equal(t, "c1", chunks[0]["id"])
	assert.Equal(t, "c2", chunks[len(chunks)-1]["id"])
	assert.Equal(t, "qwen", chunks[len(chunks)-1]["model"], "attempt 2 goes through the same model rewrite")
	assert.Contains(t, body, `"finish_reason":"stop"`)
	assert.Contains(t, body, `w299 "`)

	assert.Eventually(t, func() bool { _, aborted := sv.snapshot(); return aborted[0] }, 5*time.Second, 10*time.Millisecond,
		"the looping generation is aborted upstream")
	bodies, aborted := sv.snapshot()
	require.Len(t, bodies, 2)
	assert.False(t, aborted[1])
	assert.NotContains(t, bodies[0], "seed")
	assert.Equal(t, 1.0, bodies[1]["presence_penalty"])
	assert.IsType(t, float64(0), bodies[1]["seed"])
	for _, key := range []string{"model", "messages", "temperature", "stream", "stream_options"} {
		assert.Equal(t, bodies[0][key], bodies[1][key], key)
	}

	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRestartsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRecoveredTotal.WithLabelValues("lg-cred", "qwen")))
	assert.Zero(t, testutil.CollectAndCount(monitoring.LoopGateCutsTotal))
	assert.GreaterOrEqual(t, prx.rateLimiter.GetCurrentTPM("lg-cred")-tpmBefore, 710+config.DefaultLoopGateArmWords*4/3,
		"the restarted attempt counts against the credential TPM on top of the final usage")
}

func TestLoopGate_EnforceCutsWhenRestartsRunOut(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return writeLoopGateEndless(w, r, fmt.Sprintf("c%d", n+1))
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	assert.Len(t, chunks, 3*config.DefaultLoopGateArmWords/10+2)
	last := loopGateCutTail(t, chunks)
	assert.Equal(t, "c3", last["id"])
	marker := last["air_loop_gate"].(map[string]any)
	assert.Equal(t, "cut", marker["action"])
	assert.Equal(t, float64(3), marker["attempts"])

	assert.Eventually(t, func() bool {
		_, aborted := sv.snapshot()
		return len(aborted) == 3 && aborted[0] && aborted[1] && aborted[2]
	}, 5*time.Second, 10*time.Millisecond, "every looping attempt is aborted upstream")
	bodies, _ := sv.snapshot()
	require.Len(t, bodies, 3, "max_attempts counts the first attempt")
	assert.Equal(t, 1.0, bodies[1]["presence_penalty"])
	assert.Equal(t, 1.5, bodies[2]["presence_penalty"])
	assert.Equal(t, 2.0, testutil.ToFloat64(monitoring.LoopGateRestartsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
	assert.Zero(t, testutil.CollectAndCount(monitoring.LoopGateRecoveredTotal))
}

func TestLoopGate_EnforceNeverRestartsOnceTheAnswerStarted(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return writeLoopGateStream(w, r, "c1", loopGateWords(100), append(loopGateWords(500), loopGateLoop(200)...), "length")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	assert.Equal(t, "content", loopGateCutTail(t, chunks)["air_loop_gate"].(map[string]any)["channel"])
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 1, "the client already reads the answer: cut, no restart")
	assert.Zero(t, testutil.CollectAndCount(monitoring.LoopGateRestartsTotal))
}

func TestLoopGate_EnforceCutsWhenRestartFails(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":"overloaded"}`)
		return false
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	marker := loopGateCutTail(t, chunks)["air_loop_gate"].(map[string]any)
	assert.Equal(t, float64(1), marker["attempts"])
	assert.Equal(t, "c1", loopGateCutTail(t, chunks)["id"])
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 2)
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
}

func TestLoopGateRestartBody(t *testing.T) {
	got, err := loopGateRestartBody([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`), 1.5, 1, 42)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","seed":42,"presence_penalty":1.5,"messages":[{"role":"user","content":"hi"}]}`, string(got),
		"no client seed: the random one")

	got, err = loopGateRestartBody([]byte(`{"model":"m","seed":7,"presence_penalty":0.2}`), 1.5, 2, 42)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","seed":9,"presence_penalty":1.5}`, string(got), "client seed + restart; the gate's higher penalty")

	got, err = loopGateRestartBody([]byte(`{"model":"m","seed":42,"presence_penalty":1.8}`), 1.0, 1, 5)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","seed":43,"presence_penalty":1.8}`, string(got), "a stronger client penalty is kept")

	got, err = loopGateRestartBody([]byte(`{"model":"m","seed":null,"presence_penalty":null}`), 1.0, 1, 5)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","seed":5,"presence_penalty":1}`, string(got), "null counts as unset")

	_, err = loopGateRestartBody([]byte(`[1]`), 1, 1, 1)
	assert.Error(t, err)
	_, err = loopGateRestartBody([]byte(`null`), 1, 1, 1)
	assert.Error(t, err)
}

func TestLoopGate_EnforceNoRestartPastDeadline(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return writeLoopGateEndless(w, r, "c1")
	})
	gate := enforceGate(0, 3)
	gate.RetryDeadline = time.Nanosecond
	prx := newLoopGateProxyWithGate(t, url, gate)

	w := postChat(t, prx, loopGateStreamRequest)

	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	assert.Equal(t, float64(1), loopGateCutTail(t, chunks)["air_loop_gate"].(map[string]any)["attempts"])
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 1)
}

const loopGateNonStreamRequest = `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"temperature":0.6}`

func decodeCompletion(t *testing.T, w *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	var out map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out), w.Body.String())
	return out
}

func completionChoice0(t *testing.T, out map[string]any) (message map[string]any, finish string) {
	t.Helper()
	choices := out["choices"].([]any)
	require.Len(t, choices, 1)
	c := choices[0].(map[string]any)
	return c["message"].(map[string]any), c["finish_reason"].(string)
}

func TestLoopGate_NonStreamIsStreamedUpstreamAndRestarted(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", loopGateWords(200), loopGateWords(300), "stop")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postChat(t, prx, loopGateNonStreamRequest)

	out := decodeCompletion(t, w)
	assert.Equal(t, "chat.completion", out["object"])
	assert.Equal(t, "c2", out["id"])
	assert.Equal(t, "qwen", out["model"])
	assert.NotContains(t, out, "air_loop_gate")
	msg, finish := completionChoice0(t, out)
	assert.Equal(t, "stop", finish)
	assert.Equal(t, strings.Join(loopGateWords(300), " ")+" ", msg["content"])
	assert.Equal(t, strings.Join(loopGateWords(200), " ")+" ", msg["reasoning_content"], "only the returned attempt's reasoning")
	usage := out["usage"].(map[string]any)
	assert.Equal(t, float64(700), usage["completion_tokens"])

	bodies, _ := sv.snapshot()
	require.Len(t, bodies, 2)
	for i, b := range bodies {
		assert.Equal(t, true, b["stream"], "attempt %d goes upstream as a stream", i+1)
		assert.Equal(t, map[string]any{"include_usage": true, "continuous_usage_stats": true}, b["stream_options"],
			"vLLM reports the running usage, so a cut attempt is still accounted")
	}
	assert.Equal(t, 1.0, bodies[1]["presence_penalty"])
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRestartsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRecoveredTotal.WithLabelValues("lg-cred", "qwen")))
	assert.Zero(t, testutil.CollectAndCount(monitoring.LoopGateCutsTotal))
}

func TestLoopGate_NonStreamRestartsAnswerLoopsToo(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndlessField(w, r, "c1", "content")
		}
		return writeLoopGateStream(w, r, "c2", nil, loopGateWords(30), "stop")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	out := decodeCompletion(t, postChat(t, prx, loopGateNonStreamRequest))

	msg, finish := completionChoice0(t, out)
	assert.Equal(t, "stop", finish)
	assert.Equal(t, strings.Join(loopGateWords(30), " ")+" ", msg["content"], "the client saw nothing of the looping answer")
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 2)
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRestartsTotal.WithLabelValues("lg-cred", "qwen", "content")))
}

func TestLoopGate_NonStreamReturnsTrimmedAttemptWhenRestartsRunOut(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return writeLoopGateEndlessField(w, r, fmt.Sprintf("c%d", n+1), "content")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	out := decodeCompletion(t, postChat(t, prx, loopGateNonStreamRequest))

	msg, finish := completionChoice0(t, out)
	assert.Equal(t, "length", finish)
	content := msg["content"].(string)
	assert.True(t, strings.HasPrefix(content, strings.Join(loopGateWords(500), " ")+" loop"), "the start of the answer is kept")
	assert.Equal(t, 1, strings.Count(content, "loop0 "), "one round of the loop is kept, the looped tail is trimmed")
	marker := out["air_loop_gate"].(map[string]any)
	assert.Equal(t, "cut", marker["action"])
	assert.Equal(t, "content", marker["channel"])
	assert.Equal(t, float64(3), marker["attempts"])
	usage := out["usage"].(map[string]any)
	assert.Equal(t, float64(3000*4/3), usage["completion_tokens"], "no upstream usage: the streamed words as tokens")
	assert.Positive(t, usage["prompt_tokens"], "the local prompt estimate")
	assert.Equal(t, usage["prompt_tokens"].(float64)+usage["completion_tokens"].(float64), usage["total_tokens"])

	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 3)
	assert.Equal(t, 2.0, testutil.ToFloat64(monitoring.LoopGateRestartsTotal.WithLabelValues("lg-cred", "qwen", "content")))
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "content")))
}

func TestLoopGate_NonStreamErrorInsideStreamKeepsItsStatus(t *testing.T) {
	resetLoopGateEnforceMetrics()
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"error\":{\"object\":\"error\",\"message\":\"This model's maximum context length is 8192 tokens\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}}\n\ndata: [DONE]\n\n")
		return false
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postChat(t, prx, loopGateNonStreamRequest)

	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
}

func TestLoopGate_NonStreamObserveDoesNotStreamUpstream(t *testing.T) {
	for name, gate := range map[string]config.LoopGateConfig{
		"observe": {LoopGateSettings: config.LoopGateSettings{Mode: config.LoopGateObserve, ArmWords: config.DefaultLoopGateArmWords}},
		"enforce without force_stream": func() config.LoopGateConfig {
			g := enforceGate(0, 3)
			off := false
			g.ForceStream = &off
			return g
		}(),
	} {
		t.Run(name, func(t *testing.T) {
			resetLoopGateEnforceMetrics()
			looping := strings.Join(append(loopGateWords(500), loopGateLoop(100)...), " ")
			sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
				w.Header().Set("Content-Type", "application/json")
				text, _ := json.Marshal(looping)
				_, _ = fmt.Fprintf(w, `{"id":"c1","object":"chat.completion","created":1,"model":"Qwen/Qwen3.6-35B","choices":[{"index":0,"message":{"role":"assistant","content":%s},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":6000,"total_tokens":6001}}`, text)
				return false
			})
			prx := newLoopGateProxyWithGate(t, url, gate)

			out := decodeCompletion(t, postChat(t, prx, loopGateNonStreamRequest))

			msg, _ := completionChoice0(t, out)
			assert.Equal(t, looping, msg["content"], "the response is not changed")
			bodies, _ := sv.snapshot()
			require.Len(t, bodies, 1)
			assert.NotContains(t, bodies[0], "stream")
			mode := string(gate.Mode)
			assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "content", mode)))
			assert.InDelta(t, float64(4500-config.DefaultLoopGateArmWords),
				testutil.ToFloat64(monitoring.LoopGateWordsAfterFiringTotal.WithLabelValues("lg-cred", "qwen", "content", mode)), 10,
				"fires about where a stream would have, so words_after_firing estimates what enforce would save")
		})
	}
}

func TestLoopGate_NonStreamResponsesAPIOverVLLM(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", nil, []string{"final", "answer"}, "stop")
	})
	passthrough := false
	prx := newLoopGateProxyWithModel(t, url, enforceGate(0, 3), config.ModelRPMConfig{PassthroughResponses: &passthrough})

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"qwen","input":"hi"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Contains(t, w.Body.String(), `"object":"response"`)
	assert.Contains(t, w.Body.String(), "final answer")
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 2, "the Responses API request is converted to a chat completion and gated the same way")
}

func TestLoopGate_NonStreamTrimmedAttemptCarriesRunningUsage(t *testing.T) {
	resetLoopGateEnforceMetrics()
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		phrase := strings.Join(loopGateLoop(1)[:10], " ") + " "
		for i := 1; i <= 100000; i++ {
			if r.Context().Err() != nil {
				return true
			}
			if _, err := fmt.Fprintf(w, "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n", phrase, i*12, 10+i*12); err != nil {
				return true
			}
			flusher.Flush()
		}
		return false
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 2))
	tpmBefore := prx.rateLimiter.GetCurrentTPM("lg-cred")

	out := decodeCompletion(t, postChat(t, prx, loopGateNonStreamRequest))

	_, finish := completionChoice0(t, out)
	assert.Equal(t, "length", finish)
	usage := out["usage"].(map[string]any)
	cutAt := float64(config.DefaultLoopGateArmWords / 10 * 12)
	assert.Equal(t, cutAt, usage["completion_tokens"], "the running usage at the cut")
	// Both attempts reach the engine: the returned one via its usage, the
	// other via the loop gate.
	assert.GreaterOrEqual(t, prx.rateLimiter.GetCurrentTPM("lg-cred")-tpmBefore, 2*int(10+cutAt))
}

func TestLoopGateStreamBody(t *testing.T) {
	got, err := loopGateStreamBody([]byte(`{"model":"m","stream":false,"stream_options":{"x":1},"messages":[]}`), false)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[]}`, string(got))
	assert.Equal(t, 1, strings.Count(string(got), `"stream"`), "replaced, not duplicated")
	_, err = loopGateStreamBody([]byte(`"x"`), false)
	assert.Error(t, err)
}

func TestLoopGate_NonStreamMessagesAPIOverVLLM(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", nil, []string{"final", "answer"}, "stop")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"qwen","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Contains(t, w.Body.String(), `"type":"message"`)
	assert.Contains(t, w.Body.String(), "final answer")
	bodies, _ := sv.snapshot()
	require.Len(t, bodies, 2, "the Messages API request is converted to a chat completion and gated the same way")
	assert.Equal(t, true, bodies[0]["stream"])
}

// writeLoopGateNativeEndless streams events produced by event(i) for word
// chunk i (500 distinct words, then the same phrase) until the reader goes
// away, after the given opening events. Returns true when the reader went away.
func writeLoopGateNativeEndless(w http.ResponseWriter, r *http.Request, opening []string, event func(i int, text string) string) bool {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher := w.(http.Flusher)
	for _, e := range opening {
		_, _ = io.WriteString(w, e)
	}
	words := loopGateWords(500)
	for i := 0; i < 1_000_000; i += 10 {
		if r.Context().Err() != nil {
			return true
		}
		var chunk []string
		if i < len(words) {
			chunk = words[i : i+10]
		} else {
			chunk = loopGateLoop(1)[i%40 : i%40+10]
		}
		text, _ := json.Marshal(strings.Join(chunk, " ") + " ")
		if _, err := io.WriteString(w, event(i/10, string(text))); err != nil {
			return true
		}
		flusher.Flush()
	}
	return false
}

func responsesReasoningLoop(w http.ResponseWriter, r *http.Request) bool {
	return writeLoopGateNativeEndless(w, r, []string{
		"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":5,\"model\":\"Qwen/Qwen3.6-35B\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
		"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
	}, func(i int, text string) string {
		return fmt.Sprintf("event: response.reasoning_text.delta\ndata: {\"type\":\"response.reasoning_text.delta\",\"sequence_number\":%d,\"item_id\":\"rs_1\",\"output_index\":0,\"content_index\":0,\"delta\":%s}\n\n", i+2, text)
	})
}

func messagesThinkingLoop(w http.ResponseWriter, r *http.Request) bool {
	return writeLoopGateNativeEndless(w, r, []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"Qwen/Qwen3.6-35B\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
	}, func(_ int, text string) string {
		return fmt.Sprintf("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":%s}}\n\n", text)
	})
}

func postLoopGate(t *testing.T, prx *Proxy, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// sseEventTypes returns the "type" of every data payload in an SSE body.
func sseEventTypes(t *testing.T, body string) []string {
	t.Helper()
	var types []string
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var ev struct {
			Type string `json:"type"`
		}
		require.NoError(t, json.Unmarshal([]byte(payload), &ev), payload)
		types = append(types, ev.Type)
	}
	return types
}

func lastSSEPayload(t *testing.T, body, eventType string) map[string]any {
	t.Helper()
	var found map[string]any
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var ev map[string]any
		require.NoError(t, json.Unmarshal([]byte(payload), &ev))
		if ev["type"] == eventType {
			found = ev
		}
	}
	require.NotNil(t, found, "no %s event in %s", eventType, body)
	return found
}

func countOf(items []string, want string) int {
	n := 0
	for _, it := range items {
		if it == want {
			n++
		}
	}
	return n
}

func TestLoopGate_ResponsesPassthroughStreamIsCut(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return responsesReasoningLoop(w, r)
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postLoopGate(t, prx, "/v1/responses", `{"model":"qwen","input":"hi","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	types := sseEventTypes(t, w.Body.String())
	require.NotEmpty(t, types)
	assert.Equal(t, "response.incomplete", types[len(types)-1])
	assert.Equal(t, 1, countOf(types, "response.created"))
	assert.Equal(t, config.DefaultLoopGateArmWords/10, countOf(types, "response.reasoning_text.delta"))
	last := lastSSEPayload(t, w.Body.String(), "response.incomplete")
	resp := last["response"].(map[string]any)
	assert.Equal(t, "resp_1", resp["id"])
	assert.Equal(t, "incomplete", resp["status"])
	assert.Equal(t, "max_output_tokens", resp["incomplete_details"].(map[string]any)["reason"])
	assert.Equal(t, []string{"response.reasoning_text.done", "response.output_item.done", "response.incomplete"}, types[len(types)-3:],
		"the open reasoning item is closed before the terminal event")
	assert.Equal(t, float64(config.DefaultLoopGateArmWords/10+4), last["sequence_number"])
	streamed := strings.Join(loopGateWords(500), " ") + " "
	done := lastSSEPayload(t, w.Body.String(), "response.reasoning_text.done")
	assert.Equal(t, "rs_1", done["item_id"])
	assert.True(t, strings.HasPrefix(done["text"].(string), streamed), "the done event carries the whole streamed text")
	output := resp["output"].([]any)
	require.Len(t, output, 1, "the terminal event carries the output, not []")
	item := output[0].(map[string]any)
	assert.Equal(t, "reasoning", item["type"])
	assert.Equal(t, "rs_1", item["id"])
	assert.Equal(t, "incomplete", item["status"])
	assert.Equal(t, []any{}, item["summary"])
	assert.Equal(t, done["text"], item["content"].([]any)[0].(map[string]any)["text"])
	usage := resp["usage"].(map[string]any)
	assert.Equal(t, float64(config.DefaultLoopGateArmWords*4/3), usage["output_tokens"])
	assert.Equal(t, usage["output_tokens"], usage["output_tokens_details"].(map[string]any)["reasoning_tokens"])
	assert.Positive(t, usage["input_tokens"], "the local prompt estimate")
	marker := last["air_loop_gate"].(map[string]any)
	assert.Equal(t, "reasoning", marker["channel"])
	assert.Equal(t, float64(1), marker["attempts"], "a Responses stream is cut, never restarted")

	assert.Eventually(t, func() bool { _, a := sv.snapshot(); return len(a) == 1 && a[0] }, 5*time.Second, 10*time.Millisecond,
		"the upstream generation is aborted")
	bodies, _ := sv.snapshot()
	assert.Equal(t, "hi", bodies[0]["input"], "the request went to vLLM's own /v1/responses")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
}

func TestLoopGate_ResponsesPassthroughCutClosesOpenMessage(t *testing.T) {
	resetLoopGateEnforceMetrics()
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return writeLoopGateNativeEndless(w, r, []string{
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":5,\"model\":\"Qwen/Qwen3.6-35B\",\"status\":\"in_progress\",\"instructions\":\"be brief\",\"output\":[],\"usage\":{\"input_tokens\":42,\"output_tokens\":0,\"total_tokens\":42}}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"think\"}]}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":3,\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
			"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":4,\"item_id\":\"msg_1\",\"output_index\":1,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}\n\n",
		}, func(i int, text string) string {
			return fmt.Sprintf("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":%d,\"item_id\":\"msg_1\",\"output_index\":1,\"content_index\":0,\"delta\":%s}\n\n", i+5, text)
		})
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postLoopGate(t, prx, "/v1/responses", `{"model":"qwen","input":"hi","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	body := w.Body.String()
	types := sseEventTypes(t, body)
	assert.Equal(t, []string{"response.output_text.done", "response.content_part.done", "response.output_item.done", "response.incomplete"}, types[len(types)-4:])
	textDone := lastSSEPayload(t, body, "response.output_text.done")
	text := textDone["text"].(string)
	assert.True(t, strings.HasPrefix(text, strings.Join(loopGateWords(500), " ")+" loop"))
	assert.Equal(t, float64(1), textDone["output_index"])
	assert.Equal(t, float64(0), textDone["content_index"])
	partDone := lastSSEPayload(t, body, "response.content_part.done")
	assert.Equal(t, map[string]any{"type": "output_text", "text": text, "annotations": []any{}}, partDone["part"])

	last := lastSSEPayload(t, body, "response.incomplete")
	resp := last["response"].(map[string]any)
	assert.Equal(t, "be brief", resp["instructions"], "the response object of response.created is kept")
	assert.Equal(t, "incomplete", resp["status"])
	output := resp["output"].([]any)
	require.Len(t, output, 2)
	assert.Equal(t, "think", output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"], "a finished item is kept as sent")
	msg := output[1].(map[string]any)
	assert.Equal(t, "msg_1", msg["id"])
	assert.Equal(t, "assistant", msg["role"])
	assert.Equal(t, "incomplete", msg["status"])
	assert.Equal(t, []any{partDone["part"]}, msg["content"])
	assert.Equal(t, float64(42), resp["usage"].(map[string]any)["input_tokens"], "input_tokens of response.created")

	// Sequence numbers keep counting after the last relayed event.
	var seqs []float64
	for _, typ := range types[len(types)-4:] {
		seqs = append(seqs, lastSSEPayload(t, body, typ)["sequence_number"].(float64))
	}
	assert.Equal(t, []float64{seqs[0], seqs[0] + 1, seqs[0] + 2, seqs[0] + 3}, seqs)

	// What the OpenAI SDK and the response store read from the terminal event.
	var typed struct {
		Response responses.Response `json:"response"`
	}
	payload, err := json.Marshal(last)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(payload, &typed))
	require.Len(t, typed.Response.Output, 2)
	require.NotNil(t, typed.Response.Usage)
	assert.Positive(t, typed.Response.Usage.OutputTokens)
}

func TestLoopGate_ResponsesPassthroughStreamObserve(t *testing.T) {
	resetLoopGateEnforceMetrics()
	reasoning := append(loopGateWords(500), loopGateLoop(100)...)
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n")
		for i := 0; i < len(reasoning); i += 10 {
			text, _ := json.Marshal(strings.Join(reasoning[i:i+10], " ") + " ")
			_, _ = fmt.Fprintf(w, "event: response.reasoning_text.delta\ndata: {\"type\":\"response.reasoning_text.delta\",\"delta\":%s}\n\n", text)
		}
		_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n")
		return false
	})
	prx := newLoopGateProxyWithGate(t, url, config.LoopGateConfig{LoopGateSettings: config.LoopGateSettings{Mode: config.LoopGateObserve, ArmWords: config.DefaultLoopGateArmWords}})

	w := postLoopGate(t, prx, "/v1/responses", `{"model":"qwen","input":"hi","stream":true}`)

	types := sseEventTypes(t, w.Body.String())
	assert.Equal(t, "response.completed", types[len(types)-1], "observe mode leaves the stream intact")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "observe")))
}

func TestLoopGate_MessagesPassthroughStreamIsCut(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return messagesThinkingLoop(w, r)
	})
	passthrough := true
	prx := newLoopGateProxyWithModel(t, url, enforceGate(0, 3), config.ModelRPMConfig{PassthroughMessages: &passthrough})

	w := postLoopGate(t, prx, "/v1/messages", `{"model":"qwen","max_tokens":100000,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	types := sseEventTypes(t, w.Body.String())
	require.GreaterOrEqual(t, len(types), 4)
	assert.Equal(t, []string{"content_block_stop", "message_delta", "message_stop"}, types[len(types)-3:])
	assert.Equal(t, 1, countOf(types, "message_start"))
	delta := lastSSEPayload(t, w.Body.String(), "message_delta")
	assert.Equal(t, "max_tokens", delta["delta"].(map[string]any)["stop_reason"])
	assert.Equal(t, "reasoning", delta["air_loop_gate"].(map[string]any)["channel"])
	stop := lastSSEPayload(t, w.Body.String(), "content_block_stop")
	assert.Equal(t, float64(0), stop["index"])

	assert.Eventually(t, func() bool { _, a := sv.snapshot(); return len(a) == 1 && a[0] }, 5*time.Second, 10*time.Millisecond)
	bodies, _ := sv.snapshot()
	assert.Contains(t, bodies[0], "max_tokens", "the request went to vLLM's own /v1/messages")
	assert.NotContains(t, bodies[0], "stream_options")
}

func TestLoopGate_ConvertedMessagesStreamIsRestarted(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", loopGateWords(20), []string{"final", "answer"}, "stop")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postLoopGate(t, prx, "/v1/messages", `{"model":"qwen","max_tokens":100000,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	types := sseEventTypes(t, w.Body.String())
	assert.Equal(t, 1, countOf(types, "message_start"), "the restart continues the same message")
	assert.Equal(t, 1, countOf(types, "message_stop"))
	assert.Equal(t, "message_stop", types[len(types)-1])
	assert.Contains(t, w.Body.String(), "final answer")
	assert.Equal(t, "end_turn", lastSSEPayload(t, w.Body.String(), "message_delta")["delta"].(map[string]any)["stop_reason"])
	bodies, _ := sv.snapshot()
	require.Len(t, bodies, 2)
	assert.Contains(t, bodies[0], "messages", "converted to a chat completion")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRestartsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
}

func TestLoopGate_ConvertedResponsesStreamIsRestarted(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", loopGateWords(20), []string{"final", "answer"}, "stop")
	})
	passthrough := false
	prx := newLoopGateProxyWithModel(t, url, enforceGate(0, 3), config.ModelRPMConfig{PassthroughResponses: &passthrough})

	w := postLoopGate(t, prx, "/v1/responses", `{"model":"qwen","input":"hi","stream":true}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	types := sseEventTypes(t, w.Body.String())
	assert.Equal(t, 1, countOf(types, "response.created"), "the restart continues the same response")
	assert.Equal(t, 1, countOf(types, "response.completed"))
	assert.Equal(t, "response.completed", types[len(types)-1])
	assert.Contains(t, w.Body.String(), "final answer")
	bodies, _ := sv.snapshot()
	require.Len(t, bodies, 2)
	assert.Contains(t, bodies[0], "messages", "converted to a chat completion")
}

func TestLoopGate_ConvertedMessagesStreamCutOnAnswerLoop(t *testing.T) {
	resetLoopGateEnforceMetrics()
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		return writeLoopGateEndlessField(w, r, "c1", "content")
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postLoopGate(t, prx, "/v1/messages", `{"model":"qwen","max_tokens":100000,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	types := sseEventTypes(t, w.Body.String())
	assert.Equal(t, "message_stop", types[len(types)-1])
	assert.Equal(t, "max_tokens", lastSSEPayload(t, w.Body.String(), "message_delta")["delta"].(map[string]any)["stop_reason"],
		"the chat finish_reason length becomes the Messages max_tokens stop")
}

func TestLoopGate_NonStreamPassthroughIsObserved(t *testing.T) {
	looping := strings.Join(append(loopGateWords(500), loopGateLoop(100)...), " ")
	text, _ := json.Marshal(looping)
	for name, tc := range map[string]struct {
		path, request, response string
		model                   config.ModelRPMConfig
	}{
		"responses": {
			path: "/v1/responses", request: `{"model":"qwen","input":"hi"}`,
			response: `{"id":"resp_1","object":"response","status":"completed","model":"Qwen/Qwen3.6-35B","output":[{"type":"reasoning","id":"rs","summary":[],"content":[{"type":"reasoning_text","text":` + string(text) + `}]},{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":6000,"total_tokens":6001}}`,
		},
		"messages": {
			path: "/v1/messages", request: `{"model":"qwen","max_tokens":100000,"messages":[{"role":"user","content":"hi"}]}`,
			response: `{"id":"msg_1","type":"message","role":"assistant","model":"Qwen/Qwen3.6-35B","content":[{"type":"thinking","thinking":` + string(text) + `,"signature":""},{"type":"text","text":"ok"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":6000}}`,
			model:    config.ModelRPMConfig{PassthroughMessages: func() *bool { v := true; return &v }()},
		},
	} {
		t.Run(name, func(t *testing.T) {
			resetLoopGateEnforceMetrics()
			_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
				return false
			})
			prx := newLoopGateProxyWithModel(t, url, enforceGate(0, 3), tc.model)

			w := postLoopGate(t, prx, tc.path, tc.request)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Contains(t, w.Body.String(), "loop0", "the response is not changed")
			assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateFiringsTotal.WithLabelValues("lg-cred", "qwen", "reasoning", "enforce")))
		})
	}
}

func TestLoopGate_RestartGoesThroughCredentialProxy(t *testing.T) {
	resetLoopGateEnforceMetrics()
	// The scripted server is the credential's forward proxy: the upstream host
	// does not resolve, so a request that skips proxy_url fails.
	sv, proxyURL := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		return writeLoopGateStream(w, r, "c2", loopGateWords(200), loopGateWords(300), "stop")
	})
	cred := config.CredentialConfig{BaseURL: "http://upstream.invalid", ProxyURL: proxyURL}
	prx := newLoopGateProxyWith(t, cred, enforceGate(0, 3), config.ModelRPMConfig{}, 0)

	w := postChat(t, prx, loopGateStreamRequest)

	body := w.Body.String()
	assert.NotContains(t, body, "air_loop_gate", "the restart reached the upstream through proxy_url")
	assert.Contains(t, body, `"finish_reason":"stop"`)
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 2)
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateRecoveredTotal.WithLabelValues("lg-cred", "qwen")))
}

func TestLoopGate_NonStreamHungUpstreamTimesOut(t *testing.T) {
	resetLoopGateEnforceMetrics()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		writeLoopGateStream(w, r, "c1", loopGateWords(20), nil, "")
		<-r.Context().Done() // the engine hangs mid-generation
		return true
	})
	prx := newLoopGateProxyWith(t, config.CredentialConfig{BaseURL: url}, enforceGate(0, 3), config.ModelRPMConfig{}, 300*time.Millisecond)

	start := time.Now()
	w := postChat(t, prx, loopGateNonStreamRequest)

	assert.Less(t, time.Since(start), 5*time.Second, "each forced-stream attempt is bounded by request_timeout")
	assert.Equal(t, http.StatusRequestTimeout, w.Code, w.Body.String())
	assert.Eventually(t, func() bool { _, aborted := sv.snapshot(); return len(aborted) > 0 && aborted[0] }, 5*time.Second, 10*time.Millisecond,
		"the hung generation is aborted upstream")
}

func TestLoopGate_NonStreamFailedRestartStillCountsTheCut(t *testing.T) {
	resetLoopGateEnforceMetrics()
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndlessField(w, r, "c1", "content")
		}
		writeLoopGateStream(w, r, "c2", nil, loopGateWords(30), "")
		panic(http.ErrAbortHandler) // the restart's connection drops midway
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	out := decodeCompletion(t, postChat(t, prx, loopGateNonStreamRequest))

	assert.Equal(t, "c1", out["id"], "answered with the trimmed first attempt")
	marker := out["air_loop_gate"].(map[string]any)
	assert.Equal(t, float64(2), marker["attempts"], "the failed restart is an attempt too")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "content")))
}

func TestLoopGateObserver_FinishClosesRestartedUpstream(t *testing.T) {
	o := newTestLoopGateObserver(t, config.LoopGateEnforce)
	var reasoning []string
	reasoning = append(reasoning, loopGateWords(100)...)
	reasoning = append(reasoning, loopGateLoop(20)...)
	input := strings.ReplaceAll(loopGateSSE(reasoning), `"content"`, `"reasoning_content"`)
	restarted := &closeTrackingBody{Reader: strings.NewReader(loopGateSSE(loopGateWords(1000)))}
	o.respawn = func(int, float64) (io.ReadCloser, error) { return restarted, nil }
	r := o.wrap(io.NopCloser(strings.NewReader(input)))

	buf := make([]byte, 64)
	for o.restarts == 0 {
		_, err := r.Read(buf)
		require.NoError(t, err)
	}
	o.finish() // the client went away mid-stream

	assert.True(t, restarted.closed, "the restarted attempt's body is closed when the handler returns")
}

func TestLoopGateObserver_LastLineWithoutNewlineIsObserved(t *testing.T) {
	input := strings.TrimSuffix(loopGateSSE(loopGateWords(30)), "data: [DONE]\n\n")
	input = strings.TrimSuffix(input, "\n\n")
	o := newTestLoopGateObserver(t, config.LoopGateObserve)

	out, err := io.ReadAll(o.wrap(io.NopCloser(strings.NewReader(input))))

	require.NoError(t, err)
	assert.Equal(t, input, string(out))
	o.finishAttempt("")
	assert.Equal(t, 30, o.detector.Content().Words())
}

func TestProxy_BodyReadFailureOnLastCredentialIsNotEmpty200(t *testing.T) {
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"id":"x",`)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		return true
	})
	prx := newLoopGateProxyWith(t, config.CredentialConfig{BaseURL: url}, config.LoopGateConfig{}, config.ModelRPMConfig{}, 300*time.Millisecond)

	w := postChat(t, prx, loopGateNonStreamRequest)

	assert.Equal(t, http.StatusBadGateway, w.Code, "a body lost mid-read on the last credential is a transport error, not an empty 200")
}

func TestLoopGateObserver_CutUsageTakesTheRunningUsage(t *testing.T) {
	var b strings.Builder
	for i, line := range strings.Split(strings.TrimSuffix(loopGateSSE(loopGateLoop(20)), "data: [DONE]\n\n"), "\n\n") {
		if line == "" {
			continue
		}
		// vLLM's continuous_usage_stats: the running usage on every chunk.
		fmt.Fprintf(&b, "%s,\"usage\":{\"prompt_tokens\":77,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n", strings.TrimSuffix(line, "}"), 13*(i+1), 77+13*(i+1))
	}
	o := newTestLoopGateObserver(t, config.LoopGateEnforce)

	out, err := io.ReadAll(o.wrap(io.NopCloser(strings.NewReader(b.String()))))

	require.NoError(t, err)
	require.NotNil(t, o.cut)
	chunks, done := loopGateSSEChunks(t, string(out))
	require.True(t, done)
	loopGateCutTail(t, chunks)
	u := chunks[len(chunks)-1]["usage"].(map[string]any)
	cutChunks := o.cut.words / 10
	assert.Equal(t, float64(77), u["prompt_tokens"])
	assert.Equal(t, float64(13*cutChunks), u["completion_tokens"], "the engine's count at the cut, not an estimate")
	assert.Equal(t, float64(77+13*cutChunks), u["total_tokens"])
}

func TestLoopGate_RestartCountsLikeARequest(t *testing.T) {
	resetLoopGateEnforceMetrics()
	monitoring.CredentialErrorsTotal.Reset()
	_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if n == 0 {
			return writeLoopGateEndless(w, r, "c1")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":"overloaded"}`)
		return false
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 2))
	rpmBefore := prx.rateLimiter.GetCurrentRPM("lg-cred")

	w := postChat(t, prx, loopGateStreamRequest)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 2, prx.rateLimiter.GetCurrentRPM("lg-cred")-rpmBefore, "the restart counts toward the credential RPM")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.CredentialErrorsTotal.WithLabelValues("lg-cred")), "the failed restart is a failed attempt")
}

func TestLoopGate_OffLeavesRequestsAndStreamsUntouched(t *testing.T) {
	resetLoopGateEnforceMetrics()
	// A looping stream: with the gate on it would be restarted or cut.
	rec := httptest.NewRecorder()
	writeLoopGateStream(rec, httptest.NewRequest(http.MethodPost, "/", nil), "c1", loopGateLoop(100), loopGateWords(20), "stop")
	stream := rec.Body.String()
	sv, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		if r.Header.Get("X-Test-Stream") == "" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, vllmOKBody)
			return false
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, stream)
		return false
	})
	prx := newLoopGateProxyWithGate(t, url, config.LoopGateConfig{}) // no loop_gate section: mode off

	w := postChat(t, prx, loopGateNonStreamRequest)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(loopGateStreamRequest))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("X-Test-Stream", "1")
	ws := httptest.NewRecorder()
	prx.ProxyRequest(ws, req)

	bodies, _ := sv.snapshot()
	require.Len(t, bodies, 2)
	assert.NotContains(t, bodies[0], "stream", "a non-streamed request is not streamed upstream")
	assert.NotContains(t, bodies[0], "stream_options")
	assert.NotContains(t, bodies[1], "seed")
	assert.NotContains(t, ws.Body.String(), "air_loop_gate", "a looping stream is not cut")
	assert.Equal(t, strings.Count(stream, "data: "), strings.Count(ws.Body.String(), "data: "), "every upstream event reaches the client")
	assert.Zero(t, testutil.CollectAndCount(monitoring.LoopGateStreamsTotal))
}
