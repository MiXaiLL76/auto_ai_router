package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	logger := testhelpers.NewTestLogger()
	creds := []config.CredentialConfig{
		{Name: "lg-cred", Type: config.ProviderTypeVLLM, BaseURL: upstreamURL, APIKey: "k", RPM: 100, TPM: 10000000},
	}
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "qwen", Credential: "lg-cred", RPM: -1, TPM: -1, Model: "Qwen/Qwen3.6-35B"},
	})
	manager.LoadModelsFromConfig(creds)
	gate.ApplyDefaults()

	builder := NewTestProxyBuilder().WithCredentials(creds...)
	builder.config.ModelManager = manager
	builder.config.Metrics = monitoring.New(true)
	builder.config.LoopGate = gate
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
	assert.Len(t, chunks, config.DefaultLoopGateArmWords/10+1)
	assert.NotContains(t, w.Body.String(), `"content":"w`)

	last := chunks[len(chunks)-1]
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
	marker := chunks[len(chunks)-1]["air_loop_gate"].(map[string]any)
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
	assert.Equal(t, "content", chunks[len(chunks)-1]["air_loop_gate"].(map[string]any)["channel"])
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
	assert.True(t, strings.HasSuffix(tail, "\"finish_reason\":\"length\"}],\"air_loop_gate\":{\"action\":\"cut\",\"channel\":\"content\",\"words\":"+strconv.Itoa(o.cut.words)+",\"ratio\":"+strconv.FormatFloat(o.cut.ratio, 'g', -1, 64)+",\"attempts\":1}}\n\ndata: [DONE]\n\n"), tail)
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
		if _, err := fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":%s},\"finish_reason\":null}]}\n\n", id, text); err != nil {
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
	assert.Len(t, chunks, 3*config.DefaultLoopGateArmWords/10+1)
	last := chunks[len(chunks)-1]
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
	assert.Equal(t, "content", chunks[len(chunks)-1]["air_loop_gate"].(map[string]any)["channel"])
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
	marker := chunks[len(chunks)-1]["air_loop_gate"].(map[string]any)
	assert.Equal(t, float64(1), marker["attempts"])
	assert.Equal(t, "c1", chunks[len(chunks)-1]["id"])
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 2)
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.LoopGateCutsTotal.WithLabelValues("lg-cred", "qwen", "reasoning")))
}

func TestLoopGateRestartBody(t *testing.T) {
	got, err := loopGateRestartBody([]byte(`{"model":"m","seed":1,"presence_penalty":0.2,"messages":[{"role":"user","content":"hi"}]}`), 1.5, 42)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"m","seed":42,"presence_penalty":1.5,"messages":[{"role":"user","content":"hi"}]}`, string(got))

	_, err = loopGateRestartBody([]byte(`[1]`), 1, 1)
	assert.Error(t, err)
	_, err = loopGateRestartBody([]byte(`null`), 1, 1)
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
	assert.Equal(t, float64(1), chunks[len(chunks)-1]["air_loop_gate"].(map[string]any)["attempts"])
	bodies, _ := sv.snapshot()
	assert.Len(t, bodies, 1)
}
