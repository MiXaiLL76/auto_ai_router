package proxy

import (
	"bytes"
	"compress/gzip"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsBedrockDailyTokenQuotaError(t *testing.T) {
	body := []byte(`{"__type":"ThrottlingException","message":"Too many tokens per day"}`)

	assert.True(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusTooManyRequests, body))
	assert.True(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusTooManyRequests,
		[]byte(`THROTTLINGEXCEPTION: TOO MANY TOKENS PER DAY`)))

	assert.False(t, isBedrockDailyTokenQuotaError(config.ProviderTypeAnthropic, http.StatusTooManyRequests, body))
	assert.False(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusBadRequest, body))
	assert.False(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusTooManyRequests,
		[]byte(`{"__type":"ThrottlingException","message":"Rate exceeded"}`)))
	assert.False(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusTooManyRequests,
		[]byte(`{"message":"Too many tokens per day"}`)))
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, time.January, 15, 12, 0, 0, 0, time.UTC)

	deadline, ok := parseRetryAfter("120", now)
	require.True(t, ok)
	assert.Equal(t, now.Add(2*time.Minute), deadline)

	wantDate := now.Add(3 * time.Hour)
	deadline, ok = parseRetryAfter(wantDate.Format(http.TimeFormat), now)
	require.True(t, ok)
	assert.Equal(t, wantDate, deadline)

	deadline, ok = parseRetryAfter("999999999999", now)
	require.True(t, ok)
	assert.Equal(t, now.Add(bedrockDailyQuotaRetryAfterMax), deadline)

	deadline, ok = parseRetryAfter(now.Add(7*24*time.Hour).Format(http.TimeFormat), now)
	require.True(t, ok)
	assert.Equal(t, now.Add(bedrockDailyQuotaRetryAfterMax), deadline)

	for _, value := range []string{"", "invalid", "-1", "0", now.Add(-time.Minute).Format(http.TimeFormat)} {
		_, ok = parseRetryAfter(value, now)
		assert.False(t, ok, value)
	}
}

func TestBedrockDailyQuotaTrackerResetWindow(t *testing.T) {
	tests := []struct {
		name string
		now  time.Time
		want time.Time
	}{
		{
			name: "before UTC reset",
			now:  time.Date(2026, time.January, 15, 0, 0, 30, 0, time.UTC),
			want: time.Date(2026, time.January, 15, 0, 1, 0, 0, time.UTC),
		},
		{
			name: "before eastern daylight midnight",
			now:  time.Date(2026, time.January, 15, 2, 0, 0, 0, time.UTC),
			want: time.Date(2026, time.January, 15, 4, 1, 0, 0, time.UTC),
		},
		{
			name: "between eastern checkpoints",
			now:  time.Date(2026, time.January, 15, 4, 30, 0, 0, time.UTC),
			want: time.Date(2026, time.January, 15, 5, 1, 0, 0, time.UTC),
		},
		{
			name: "before pacific standard midnight",
			now:  time.Date(2026, time.January, 15, 7, 30, 0, 0, time.UTC),
			want: time.Date(2026, time.January, 15, 8, 1, 0, 0, time.UTC),
		},
		{
			name: "first error after US reset window",
			now:  time.Date(2026, time.January, 15, 12, 0, 0, 0, time.UTC),
			want: time.Date(2026, time.January, 16, 0, 1, 0, 0, time.UTC),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := newBedrockDailyQuotaTracker()
			decision := tracker.nextBan("cred", "claude", tt.now, "")
			assert.Equal(t, tt.want, decision.BanUntil)
			assert.Equal(t, bedrockDailyQuotaPhaseResetWindow, decision.Phase)
			assert.Equal(t, bedrockDailyQuotaDeadlineHeuristic, decision.Source)
		})
	}
}

func TestBedrockDailyQuotaTrackerBackoffAndNewUTCWindow(t *testing.T) {
	tracker := newBedrockDailyQuotaTracker()
	keyCred, keyModel := "cred", "claude"

	first := time.Date(2026, time.January, 15, 12, 0, 0, 0, time.UTC)
	decision := tracker.nextBan(keyCred, keyModel, first, "")
	assert.Equal(t, time.Date(2026, time.January, 16, 0, 1, 0, 0, time.UTC), decision.BanUntil)

	// UTC reset did not help: walk the plausible US local-midnight checkpoints.
	for _, checkpoint := range []struct{ hour, wantHour int }{{0, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}} {
		now := time.Date(2026, time.January, 16, checkpoint.hour, 1, 0, 0, time.UTC)
		decision = tracker.nextBan(keyCred, keyModel, now, "")
		assert.Equal(t, time.Date(2026, time.January, 16, checkpoint.wantHour, 1, 0, 0, time.UTC), decision.BanUntil)
		assert.Equal(t, bedrockDailyQuotaPhaseResetWindow, decision.Phase)
	}

	// After UTC-8 has been covered, use 1h, 2h, 4h and cap at 4h.
	decision = tracker.nextBan(keyCred, keyModel, time.Date(2026, time.January, 16, 8, 1, 0, 0, time.UTC), "")
	assert.Equal(t, time.Date(2026, time.January, 16, 9, 1, 0, 0, time.UTC), decision.BanUntil)
	assert.Equal(t, 1, decision.Attempt)

	decision = tracker.nextBan(keyCred, keyModel, decision.BanUntil, "")
	assert.Equal(t, time.Date(2026, time.January, 16, 11, 1, 0, 0, time.UTC), decision.BanUntil)
	assert.Equal(t, 2, decision.Attempt)

	decision = tracker.nextBan(keyCred, keyModel, decision.BanUntil, "")
	assert.Equal(t, time.Date(2026, time.January, 16, 15, 1, 0, 0, time.UTC), decision.BanUntil)
	assert.Equal(t, 3, decision.Attempt)

	decision = tracker.nextBan(keyCred, keyModel, decision.BanUntil, "")
	assert.Equal(t, time.Date(2026, time.January, 16, 19, 1, 0, 0, time.UTC), decision.BanUntil)
	assert.Equal(t, 4, decision.Attempt)

	// Never sleep past the next UTC reset checkpoint.
	decision = tracker.nextBan(keyCred, keyModel, time.Date(2026, time.January, 16, 23, 0, 0, 0, time.UTC), "")
	assert.Equal(t, time.Date(2026, time.January, 17, 0, 1, 0, 0, time.UTC), decision.BanUntil)
	decision = tracker.nextBan(keyCred, keyModel, decision.BanUntil, "")
	assert.Equal(t, time.Date(2026, time.January, 17, 4, 1, 0, 0, time.UTC), decision.BanUntil)
	assert.Equal(t, bedrockDailyQuotaPhaseResetWindow, decision.Phase)
}

func TestBedrockDailyQuotaTrackerRetryAfterDuplicateAndReset(t *testing.T) {
	tracker := newBedrockDailyQuotaTracker()
	now := time.Date(2026, time.January, 15, 12, 0, 0, 0, time.UTC)

	decision := tracker.nextBan("cred-a", "claude", now, "600")
	assert.Equal(t, now.Add(10*time.Minute), decision.BanUntil)
	assert.Equal(t, bedrockDailyQuotaPhaseRetryAfter, decision.Phase)
	assert.Equal(t, bedrockDailyQuotaDeadlineRetryAfter, decision.Source)

	// Concurrent duplicate failures in the same window must not advance it.
	duplicate := tracker.nextBan("cred-a", "claude", now.Add(time.Second), "")
	assert.Equal(t, decision, duplicate)

	// Another pair is isolated.
	other := tracker.nextBan("cred-b", "claude", now, "")
	assert.NotEqual(t, decision.BanUntil, other.BanUntil)

	tracker.reset("cred-a", "claude")
	afterSuccess := tracker.nextBan("cred-a", "claude", now.Add(time.Hour), "")
	assert.Equal(t, time.Date(2026, time.January, 16, 0, 1, 0, 0, time.UTC), afterSuccess.BanUntil)
}

func TestDecodeResponseBodyPrefixLimitsCompressedBody(t *testing.T) {
	body := []byte(`{"__type":"ThrottlingException","message":"Too many tokens per day"}` + strings.Repeat("x", bedrockDailyQuotaBodyScanLimit*8))
	var compressed bytes.Buffer
	gz := gzip.NewWriter(&compressed)
	_, err := gz.Write(body)
	require.NoError(t, err)
	require.NoError(t, gz.Close())

	decoded := decodeResponseBodyPrefix(compressed.Bytes(), "gzip", bedrockDailyQuotaBodyScanLimit)
	require.Len(t, decoded, bedrockDailyQuotaBodyScanLimit)
	assert.True(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusTooManyRequests, decoded))

	lateMarker := []byte(strings.Repeat("x", bedrockDailyQuotaBodyScanLimit) + `{"__type":"ThrottlingException","message":"Too many tokens per day"}`)
	compressed.Reset()
	gz = gzip.NewWriter(&compressed)
	_, err = gz.Write(lateMarker)
	require.NoError(t, err)
	require.NoError(t, gz.Close())

	decoded = decodeResponseBodyPrefix(compressed.Bytes(), "gzip", bedrockDailyQuotaBodyScanLimit)
	require.Len(t, decoded, bedrockDailyQuotaBodyScanLimit)
	assert.False(t, isBedrockDailyTokenQuotaError(config.ProviderTypeBedrock, http.StatusTooManyRequests, decoded))
}

func TestBedrockDailyQuotaTrackerConcurrentDuplicatesShareDeadline(t *testing.T) {
	tracker := newBedrockDailyQuotaTracker()
	now := time.Date(2026, time.January, 15, 12, 0, 0, 0, time.UTC)

	const goroutines = 64
	decisions := make(chan bedrockDailyQuotaDecision, goroutines)
	var wg sync.WaitGroup
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			decisions <- tracker.nextBan("cred", "claude", now, "")
		}()
	}
	wg.Wait()
	close(decisions)

	var first bedrockDailyQuotaDecision
	for decision := range decisions {
		if first.BanUntil.IsZero() {
			first = decision
		}
		assert.Equal(t, first, decision)
	}
}

func TestRecordProviderResponseBansOnlyBedrockCredentialModelPair(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "bedrock-a", Type: config.ProviderTypeBedrock, BaseURL: "https://bedrock-a.example", APIKey: "a", RPM: 100},
		{Name: "bedrock-b", Type: config.ProviderTypeBedrock, BaseURL: "https://bedrock-b.example", APIKey: "b", RPM: 100},
	}
	proxy := NewTestProxyBuilder().WithCredentials(credentials...).Build()
	body := []byte(`{"__type":"ThrottlingException","message":"Too many tokens per day"}`)

	matched := proxy.recordProviderResponse(context.Background(), &credentials[0], "claude-opus", "anthropic.claude-opus-v1:0", http.StatusTooManyRequests,
		http.Header{"Retry-After": []string{"3600"}}, body)

	assert.True(t, matched)
	assert.True(t, proxy.balancer.IsBanned("bedrock-a", "claude-opus"))
	assert.False(t, proxy.balancer.IsBanned("bedrock-a", "claude-sonnet"))
	assert.False(t, proxy.balancer.IsBanned("bedrock-b", "claude-opus"))

	selected, err := proxy.balancer.NextForModel("claude-opus")
	require.NoError(t, err)
	assert.Equal(t, "bedrock-b", selected.Name)

	selected, err = proxy.balancer.NextSpecific("bedrock-a", "claude-sonnet")
	require.NoError(t, err)
	assert.Equal(t, "bedrock-a", selected.Name)

	// A regular Bedrock 429 follows the existing generic rules and is not
	// mistaken for the daily token quota error.
	matched = proxy.recordProviderResponse(context.Background(), &credentials[1], "claude-sonnet", "anthropic.claude-sonnet-v1:0", http.StatusTooManyRequests,
		nil, []byte(`{"__type":"ThrottlingException","message":"Rate exceeded"}`))
	assert.False(t, matched)
	assert.False(t, proxy.balancer.IsBanned("bedrock-b", "claude-sonnet"))
}

func TestRecordProviderResponseBansAliasesForSameBedrockModelID(t *testing.T) {
	credential := config.CredentialConfig{
		Name: "bedrock-a", Type: config.ProviderTypeBedrock, BaseURL: "https://bedrock-a.example", APIKey: "a", RPM: 100,
	}
	builder := NewTestProxyBuilder().WithCredentials(credential)
	builder.config.ModelManager = models.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "opus-primary", Model: "anthropic.claude-opus-v1:0", Credential: "bedrock-a"},
		{Name: "opus-secondary", Model: "anthropic.claude-opus-v1:0", Credential: "bedrock-a"},
		{Name: "sonnet", Model: "anthropic.claude-sonnet-v1:0", Credential: "bedrock-a"},
	})
	builder.config.ModelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	proxy := builder.Build()

	matched := proxy.recordProviderResponse(
		context.Background(),
		&credential,
		"opus-primary",
		"anthropic.claude-opus-v1:0",
		http.StatusTooManyRequests,
		http.Header{"Retry-After": []string{"3600"}},
		[]byte(`{"__type":"ThrottlingException","message":"Too many tokens per day"}`),
	)

	require.True(t, matched)
	assert.True(t, proxy.balancer.IsBanned("bedrock-a", "opus-primary"))
	assert.True(t, proxy.balancer.IsBanned("bedrock-a", "opus-secondary"))
	assert.False(t, proxy.balancer.IsBanned("bedrock-a", "sonnet"))

	decision := proxy.bedrockDailyQuota.nextBan(
		"bedrock-a",
		"anthropic.claude-opus-v1:0",
		time.Now().UTC(),
		"",
	)
	assert.Equal(t, bedrockDailyQuotaPhaseRetryAfter, decision.Phase)
}

func TestProxyRequest_BedrockDailyQuotaStreamingErrorBansPairAndRetries(t *testing.T) {
	var throttledCalls int32
	throttled := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&throttledCalls, 1)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.Header().Set("Retry-After", "3600")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"__type":"ThrottlingException","message":"Too many tokens per day"}`))
	}))
	defer throttled.Close()

	var healthyCalls int32
	healthy := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&healthyCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_1",
			"type":"message",
			"role":"assistant",
			"model":"claude-opus",
			"stop_reason":"end_turn",
			"content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
	defer healthy.Close()

	credentials := []config.CredentialConfig{
		{Name: "bedrock-a", Type: config.ProviderTypeBedrock, BaseURL: throttled.URL, APIKey: "a", RPM: 100},
		{Name: "bedrock-b", Type: config.ProviderTypeBedrock, BaseURL: healthy.URL, APIKey: "b", RPM: 100},
	}
	builder := NewTestProxyBuilder().
		WithCredentials(credentials...).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1)
	builder.config.ModelManager = models.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "claude", Model: "anthropic.claude-opus-v1:0", Credential: "bedrock-a"},
		{Name: "claude", Model: "anthropic.claude-opus-v1:0", Credential: "bedrock-b"},
	})
	builder.config.ModelManager.LoadModelsFromConfig(credentials)
	proxy := builder.Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"claude","messages":[{"role":"user","content":"hi"}],"stream":true}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), atomic.LoadInt32(&throttledCalls))
	assert.Equal(t, int32(1), atomic.LoadInt32(&healthyCalls))
	assert.True(t, proxy.balancer.IsBanned("bedrock-a", "claude"))
	assert.False(t, proxy.balancer.IsBanned("bedrock-a", "another-model"))
	assert.Contains(t, w.Body.String(), "ok")

	_, health := proxy.HealthCheck()
	stats := health.Models["bedrock-a:claude"]
	assert.Equal(t, bedrockDailyQuotaProviderError, stats.ProviderError)
	assert.NotNil(t, stats.BanUntil)
}
