package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
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
				_, _ = fmt.Fprintf(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"%s\":%s},\"finish_reason\":null}]}\n\n", field, text)
			}
		}
		emit("reasoning_content", reasoning)
		emit("content", content)
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}]}\n\n", finishReason)
		_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"qwen\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5000,\"total_tokens\":5010}}\n\n")
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
