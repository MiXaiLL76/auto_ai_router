package httputil

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// recordedTrace is a request's trace in a test: a root span in ctx, as the
// router's server span is, and every span that ends beneath it.
type recordedTrace struct {
	ctx      context.Context
	root     trace.Span
	recorder *tracetest.SpanRecorder
}

func newRecordedTrace(t *testing.T) *recordedTrace {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	ctx, root := provider.Tracer("test").Start(context.Background(), "request")
	return &recordedTrace{ctx: ctx, root: root, recorder: recorder}
}

// attempts are the client spans, in the order they began.
func (r *recordedTrace) attempts() []sdktrace.ReadOnlySpan {
	var spans []sdktrace.ReadOnlySpan
	for _, span := range r.recorder.Ended() {
		if span.SpanKind() == trace.SpanKindClient {
			spans = append(spans, span)
		}
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].StartTime().Before(spans[j].StartTime()) })
	return spans
}

// steps are the spans beneath a client span, in the order they began.
func (r *recordedTrace) steps(parent sdktrace.ReadOnlySpan) []sdktrace.ReadOnlySpan {
	var spans []sdktrace.ReadOnlySpan
	for _, span := range r.recorder.Ended() {
		if span.Parent().SpanID() == parent.SpanContext().SpanID() {
			spans = append(spans, span)
		}
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].StartTime().Before(spans[j].StartTime()) })
	return spans
}

func names(spans []sdktrace.ReadOnlySpan) []string {
	out := make([]string, 0, len(spans))
	for _, span := range spans {
		out = append(out, span.Name())
	}
	return out
}

// ours are a span's aar.* attributes.
func ours(span sdktrace.ReadOnlySpan) map[string]any {
	out := map[string]any{}
	for _, kv := range span.Attributes() {
		if strings.HasPrefix(string(kv.Key), "aar.") {
			out[string(kv.Key)] = kv.Value.AsInterface()
		}
	}
	return out
}

// exitRTT takes the measured round trip to the exit out of a span's
// attributes, for the rest to be compared whole.
func exitRTT(t *testing.T, attrs map[string]any) (map[string]any, int64) {
	t.Helper()
	rtt, ok := attrs["aar.exit_rtt_ms"].(int64)
	require.True(t, ok, "aar.exit_rtt_ms in %v", attrs)
	delete(attrs, "aar.exit_rtt_ms")
	return attrs, rtt
}

func okServer(t *testing.T, h2 bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	srv.EnableHTTP2 = h2
	srv.StartTLS()
	t.Cleanup(srv.Close)
	return srv
}

func roundTripOK(ctx context.Context, t *testing.T, client *http.Client, target string) {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, http.NoBody)
	require.NoError(t, err)
	resp, err := client.Do(req)
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
}

// The egress path: a tunnel through a proxy that names itself. The attempt
// that opens the tunnel has every step; the next, on the pooled tunnel, has
// the same exit and only its wait for the first byte.
func TestAttemptTransport_ANewTunnelThenAPooledOne(t *testing.T) {
	for _, proto := range []string{"HTTP/1.1", "HTTP/2"} {
		t.Run(proto, func(t *testing.T) {
			h2 := proto == "HTTP/2"
			srv := okServer(t, h2)
			proxy := newConnectProxy(t)
			proxy.answer("HTTP/1.1 exit-a, 1.1 nearer-proxy", 0)
			// the exit took 30 ms of the reply's 50 to connect to the provider
			proxy.times(`cache;dur=1, connect;desc="tcp";dur=30`, 50*time.Millisecond)
			transport := tlsBuild(srv.Certificate(), h2)()
			defer transport.CloseIdleConnections()
			client := &http.Client{Transport: instrument(transport)}

			recorded := newRecordedTrace(t)
			ctx := WithProxyURL(WithAttempts(recorded.ctx), "http://"+proxy.listener.Addr().String())
			attempts := AttemptsFrom(ctx)

			attempts.Begin("cred-a")
			roundTripOK(ctx, t, client, srv.URL)
			attempts.Next(AttemptRetry, "rate_limit")
			attempts.Begin("cred-b")
			roundTripOK(ctx, t, client, srv.URL)
			recorded.root.End()

			spans := recorded.attempts()
			require.Len(t, spans, 2)
			require.EqualValues(t, 1, proxy.total.Load(), "one tunnel for both")

			first := map[string]any{
				"aar.credential": "cred-a", "aar.attempt.number": int64(1), "aar.attempt.kind": AttemptFirst,
				"aar.connection.reused": false, "aar.exit": "exit-a", "aar.exit_connect_ms": int64(30),
			}
			attrs, rtt := exitRTT(t, ours(spans[0]))
			assert.Equal(t, first, attrs)
			assert.GreaterOrEqual(t, rtt, int64(20), "the reply's 50 ms less the exit's 30")
			assert.Less(t, rtt, int64(2000))
			steps := recorded.steps(spans[0])
			require.Equal(t, []string{stepConnect, stepTunnel, stepTLS, stepTTFB}, names(steps))
			for i, step := range steps {
				attrs, stepRTT := exitRTT(t, ours(step))
				assert.Equal(t, first, attrs, "%s carries the attempt's attributes", step.Name())
				assert.Equal(t, rtt, stepRTT, step.Name())
				assert.Equal(t, codes.Unset, step.Status().Code, step.Name())
				assert.False(t, step.EndTime().Before(step.StartTime()), step.Name())
				assert.False(t, step.StartTime().Before(spans[0].StartTime()), step.Name())
				if i > 0 {
					assert.False(t, step.StartTime().Before(steps[i-1].EndTime()), "%s begins after %s ends", step.Name(), steps[i-1].Name())
				}
			}

			second := map[string]any{
				"aar.credential": "cred-b", "aar.attempt.number": int64(2), "aar.attempt.kind": AttemptRetry,
				"aar.attempt.reason": "rate_limit", "aar.connection.reused": true, "aar.exit": "exit-a",
				"aar.exit_connect_ms": int64(30),
			}
			attrs, pooledRTT := exitRTT(t, ours(spans[1]))
			assert.Equal(t, second, attrs)
			assert.Equal(t, rtt, pooledRTT, "a pooled connection carries what its opening measured")
			assert.Equal(t, []string{stepTTFB}, names(recorded.steps(spans[1])))

			assert.Equal(t, 2, attempts.Count())
			assert.Equal(t, "exit-a", attempts.Exit("cred-a"))
			assert.Equal(t, "exit-a", attempts.Exit("cred-b"))

			opened, ok := attempts.Stages("cred-a")
			require.True(t, ok)
			pooled, ok := attempts.Stages("cred-b")
			require.True(t, ok)
			require.NotNil(t, opened.ConnWait)
			require.NotNil(t, pooled.ConnWait)
			assert.Less(t, *pooled.ConnWait, *opened.ConnWait, "a pooled connection is had sooner than a tunnel is opened")
			require.NotNil(t, opened.TTFB)
			require.NotNil(t, pooled.TTFB)
			assert.Positive(t, *pooled.TTFB)
			require.NotNil(t, opened.Reused)
			require.NotNil(t, pooled.Reused)
			assert.False(t, *opened.Reused, "it had the tunnel opened")
			assert.True(t, *pooled.Reused)
			require.NotNil(t, opened.ExitConnect)
			assert.Equal(t, 30*time.Millisecond, *opened.ExitConnect)
			require.NotNil(t, opened.ExitRTT)
			assert.GreaterOrEqual(t, *opened.ExitRTT, 20*time.Millisecond)
			assert.Equal(t, opened.ExitRTT, pooled.ExitRTT, "the connection's, not the request's")
			assert.Equal(t, opened.ExitConnect, pooled.ExitConnect)
		})
	}
}

// The steps of opening the connection are spans as soon as the request has
// it: a provider that answers a whole completion later must not make them
// old news to whatever reads the spans as they arrive.
func TestAttemptTransport_TheDialsStepsDoNotWaitForTheAnswer(t *testing.T) {
	arrived, release := make(chan struct{}), make(chan struct{})
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(arrived)
		<-release
		_, _ = io.WriteString(w, "ok")
	}))
	srv.StartTLS()
	defer srv.Close()
	proxy := newConnectProxy(t)
	proxy.answer("HTTP/1.1 exit-a", 0)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	ctx := WithProxyURL(WithAttempts(recorded.ctx), "http://"+proxy.listener.Addr().String())
	AttemptsFrom(ctx).Begin("cred-a")
	failed := make(chan error, 1)
	go func() {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, http.NoBody)
		if err != nil {
			failed <- err
			return
		}
		resp, err := client.Do(req)
		if err != nil {
			failed <- err
			return
		}
		_, err = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		failed <- err
	}()

	<-arrived
	opened := recorded.recorder.Ended()
	sort.Slice(opened, func(i, j int) bool { return opened[i].StartTime().Before(opened[j].StartTime()) })
	assert.Equal(t, []string{stepConnect, stepTunnel, stepTLS}, names(opened), "written while the provider has not answered")
	for _, step := range opened {
		assert.Equal(t, "exit-a", ours(step)["aar.exit"], step.Name())
		assert.Equal(t, false, ours(step)["aar.connection.reused"], step.Name())
	}

	close(release)
	require.NoError(t, <-failed)
	recorded.root.End()
	spans := recorded.attempts()
	require.Len(t, spans, 1)
	assert.Equal(t, []string{stepConnect, stepTunnel, stepTLS, stepTTFB}, names(recorded.steps(spans[0])), "each step once")
}

// A direct connection has no tunnel and no exit.
func TestAttemptTransport_ADirectConnection(t *testing.T) {
	srv := okServer(t, false)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	ctx := WithAttempts(recorded.ctx)
	AttemptsFrom(ctx).Begin("cred-a")
	roundTripOK(ctx, t, client, srv.URL)
	recorded.root.End()

	spans := recorded.attempts()
	require.Len(t, spans, 1)
	assert.Equal(t, map[string]any{
		"aar.credential": "cred-a", "aar.attempt.number": int64(1), "aar.attempt.kind": AttemptFirst,
		"aar.connection.reused": false,
	}, ours(spans[0]))
	assert.Equal(t, []string{stepConnect, stepTLS, stepTTFB}, names(recorded.steps(spans[0])))
	assert.Empty(t, AttemptsFrom(ctx).Exit("cred-a"))
}

// A proxy that refuses the tunnel still names itself: the attempt has the
// exit, and its tunnel step failed.
func TestAttemptTransport_ARefusedTunnel(t *testing.T) {
	srv := okServer(t, false)
	proxy := newConnectProxy(t)
	proxy.answer("HTTP/1.1 exit-b", http.StatusBadGateway)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	ctx := WithProxyURL(WithAttempts(recorded.ctx), "http://"+proxy.listener.Addr().String())
	AttemptsFrom(ctx).Begin("cred-a")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, http.NoBody)
	require.NoError(t, err)
	resp, err := client.Do(req)
	if err == nil {
		_ = resp.Body.Close()
	}
	require.Error(t, err)
	recorded.root.End()

	spans := recorded.attempts()
	require.Len(t, spans, 1)
	assert.Equal(t, map[string]any{
		"aar.credential": "cred-a", "aar.attempt.number": int64(1), "aar.attempt.kind": AttemptFirst,
		"aar.exit": "exit-b",
	}, ours(spans[0]), "no connection, so nothing about its reuse")
	steps := recorded.steps(spans[0])
	require.Equal(t, []string{stepConnect, stepTunnel}, names(steps))
	assert.Equal(t, codes.Unset, steps[0].Status().Code)
	assert.Equal(t, codes.Error, steps[1].Status().Code)
	assert.Contains(t, steps[1].Status().Description, "502")
	assert.Equal(t, "exit-b", AttemptsFrom(ctx).Exit("cred-a"))
	stages, ok := AttemptsFrom(ctx).Stages("cred-a")
	require.True(t, ok)
	assert.Nil(t, stages.ConnWait, "it got no connection")
	assert.Nil(t, stages.TTFB)
}

// A connection nothing answers on: the step under way when the attempt
// failed ends there, failed.
func TestAttemptTransport_AFailedConnect(t *testing.T) {
	transport := newTransport(nil)
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	ctx := WithAttempts(recorded.ctx)
	AttemptsFrom(ctx).Begin("cred-a")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://127.0.0.1:1/", http.NoBody)
	require.NoError(t, err)
	resp, err := client.Do(req)
	if err == nil {
		_ = resp.Body.Close()
	}
	require.Error(t, err)
	recorded.root.End()

	spans := recorded.attempts()
	require.Len(t, spans, 1)
	assert.Equal(t, map[string]any{
		"aar.credential": "cred-a", "aar.attempt.number": int64(1), "aar.attempt.kind": AttemptFirst,
	}, ours(spans[0]), "no connection, no exit")
	steps := recorded.steps(spans[0])
	require.Equal(t, []string{stepConnect}, names(steps))
	assert.Equal(t, codes.Error, steps[0].Status().Code)
}

// A request that is not an attempt (the router's own: a price list, a token)
// keeps its one client span: it says how it left, and has no steps.
func TestAttemptTransport_ARequestThatIsNotAnAttempt(t *testing.T) {
	srv := okServer(t, false)
	proxy := newConnectProxy(t)
	proxy.answer("HTTP/1.1 exit-a", 0)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	recorded := newRecordedTrace(t)
	ctx := WithProxyURL(recorded.ctx, "http://"+proxy.listener.Addr().String())
	roundTripOK(ctx, t, client, srv.URL)
	// with a request's Attempts but none begun: the same
	roundTripOK(WithAttempts(ctx), t, client, srv.URL)
	recorded.root.End()

	spans := recorded.attempts()
	require.Len(t, spans, 2)
	assert.Equal(t, map[string]any{"aar.connection.reused": false, "aar.exit": "exit-a"}, ours(spans[0]))
	assert.Equal(t, map[string]any{"aar.connection.reused": true, "aar.exit": "exit-a"}, ours(spans[1]))
	assert.Empty(t, recorded.steps(spans[0]))
	assert.Empty(t, recorded.steps(spans[1]))
}

// Without tracing there are no spans to describe, and the exit and the waits
// still reach the request's Attempts, for the spend event.
func TestAttemptTransport_TheExitWithoutTracing(t *testing.T) {
	srv := okServer(t, false)
	proxy := newConnectProxy(t)
	proxy.answer("2 exit-c", 0)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	ctx := WithProxyURL(WithAttempts(context.Background()), "http://"+proxy.listener.Addr().String())
	attempts := AttemptsFrom(ctx)
	attempts.Begin("cred-a")
	roundTripOK(ctx, t, client, srv.URL)
	attempts.Begin("cred-b")
	roundTripOK(ctx, t, client, srv.URL)

	assert.Equal(t, "exit-c", attempts.Exit("cred-a"))
	assert.Equal(t, "exit-c", attempts.Exit("cred-b"), "from the pooled tunnel")
	stages, ok := attempts.Stages("cred-b")
	require.True(t, ok)
	assert.NotNil(t, stages.ConnWait, "the waits are measured without tracing too")
	assert.NotNil(t, stages.TTFB)
	require.NotNil(t, stages.Reused)
	assert.True(t, *stages.Reused)
	assert.Nil(t, stages.ExitRTT, "the proxy gave no time")
	assert.Nil(t, stages.ExitConnect)
}

// A proxy that gives its time and no name, on a way so short that its whole
// milliseconds exceed what the round trip took.
func TestAttemptTransport_TheExitsTimeWithoutItsName(t *testing.T) {
	srv := okServer(t, false)
	proxy := newConnectProxy(t)
	proxy.times("connect;dur=5000", 0)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: instrument(transport)}

	ctx := WithProxyURL(WithAttempts(context.Background()), "http://"+proxy.listener.Addr().String())
	attempts := AttemptsFrom(ctx)
	attempts.Begin("cred-a")
	roundTripOK(ctx, t, client, srv.URL)

	assert.Empty(t, attempts.Exit("cred-a"))
	stages, ok := attempts.Stages("cred-a")
	require.True(t, ok)
	require.NotNil(t, stages.ExitConnect)
	assert.Equal(t, 5*time.Second, *stages.ExitConnect)
	require.NotNil(t, stages.ExitRTT)
	assert.Zero(t, *stages.ExitRTT, "never negative")
}

func TestServerTimingConnect(t *testing.T) {
	ms := func(v float64) *time.Duration {
		d := time.Duration(v * float64(time.Millisecond))
		return &d
	}
	for _, tc := range []struct {
		lines []string
		want  *time.Duration
	}{
		{nil, nil},
		{[]string{"connect;dur=98"}, ms(98)},
		{[]string{`cache;desc="hit";dur=0.2, Connect ; desc="tcp" ; DUR="12.5"`}, ms(12.5)},
		{[]string{"cache;dur=1", "connect;dur=7, connect;dur=900"}, ms(7)},
		{[]string{"connect;dur=0"}, ms(0)},
		{[]string{"connect"}, nil},
		{[]string{"connect;dur=-"}, nil},
		{[]string{"connect;dur=-5"}, nil},
		{[]string{"connect;dur=NaN"}, nil},
		{[]string{"connect;dur=1e12"}, nil},
		{[]string{"connected;dur=5"}, nil},
	} {
		header := http.Header{}
		for _, line := range tc.lines {
			header.Add("Server-Timing", line)
		}
		assert.Equal(t, tc.want, serverTimingConnect(header), "Server-Timing: %q", tc.lines)
	}
}

func TestViaReceivedBy(t *testing.T) {
	for via, want := range map[string]string{
		"":                                   "",
		"HTTP/2 de-ber-01":                   "de-ber-01",
		"2 de-ber-01":                        "de-ber-01",
		"1.1 exit.example:3128 (Envoy)":      "exit.example:3128",
		"  1.1   far-proxy , 1.1 near-proxy": "far-proxy",
		"nameless":                           "",
		"1.1 " + strings.Repeat("x", 300):    "",
	} {
		header := http.Header{}
		if via != "" {
			header.Set("Via", via)
		}
		assert.Equal(t, want, viaReceivedBy(header), "Via: %q", via)
	}
}
