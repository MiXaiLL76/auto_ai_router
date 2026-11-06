package httputil

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	tracerName = "github.com/mixaill76/auto_ai_router/internal/httputil"

	// The steps of an upstream attempt, as spans beneath its client span.
	stepConnect = "upstream.connect" // DNS and the TCP connect, to the proxy when there is one
	stepTunnel  = "upstream.tunnel"  // the connection made -> the proxy's reply to CONNECT
	stepTLS     = "upstream.tls"     // the TLS handshake with the provider
	stepTTFB    = "upstream.ttfb"    // the request written -> the first byte of the response

	// maxExitLen bounds a name that comes from a response header.
	maxExitLen = 255
	// maxExitConnect bounds a time that comes from one.
	maxExitConnect = 10 * time.Minute
)

// instrument wraps a transport with otelhttp's client span and attemptTransport.
func instrument(base http.RoundTripper) http.RoundTripper {
	return otelhttp.NewTransport(&attemptTransport{base: base}, otelhttp.WithSpanNameFormatter(spanName))
}

// attemptTransport sits between otelhttp and the transport. It describes the
// round trip on otelhttp's client span (which attempt, the connection, its
// exit), adds a span per step that took time, and gives the request's
// Attempts what a spend row keeps.
type attemptTransport struct {
	base http.RoundTripper
}

func (t *attemptTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx := req.Context()
	span := trace.SpanFromContext(ctx)
	attempts := AttemptsFrom(ctx)
	number, current := attempts.current()

	// Only an attempt gets step spans, not the router's own requests.
	recording := span.IsRecording()
	rt := &roundTrip{timed: recording && number > 0}
	// What the spans say of the attempt: complete once it has a connection.
	describe := func() []attribute.KeyValue {
		way, reused := rt.connection()
		attrs := make([]attribute.KeyValue, 0, 8)
		if number > 0 {
			attrs = append(attrs,
				attribute.String("aar.credential", current.credential),
				attribute.Int("aar.attempt.number", number),
				attribute.String("aar.attempt.kind", current.kind),
			)
			if current.reason != "" {
				attrs = append(attrs, attribute.String("aar.attempt.reason", current.reason))
			}
		}
		if reused != nil {
			attrs = append(attrs, attribute.Bool("aar.connection.reused", *reused))
		}
		if way.exit != "" {
			attrs = append(attrs, attribute.String("aar.exit", way.exit))
		}
		if way.exitRTT != nil && way.exitConnect != nil {
			attrs = append(attrs,
				attribute.Int64("aar.exit_rtt_ms", way.exitRTT.Milliseconds()),
				attribute.Int64("aar.exit_connect_ms", way.exitConnect.Milliseconds()),
			)
		}
		return attrs
	}
	if rt.timed {
		secure := req.URL.Scheme == "https"
		proxy, _ := proxyFromRequest(req)
		rt.secure = secure
		rt.tunnels = secure && proxy != nil && (proxy.Scheme == "http" || proxy.Scheme == "https")
		// Steps are written with explicit times once they are over. The dial's
		// are written as soon as the request has its connection: the provider
		// may answer a whole completion later.
		tracer := span.TracerProvider().Tracer(tracerName)
		parent := trace.ContextWithSpan(context.Background(), span)
		rt.write = func(steps []step) {
			attrs := describe()
			for _, s := range steps {
				_, child := tracer.Start(parent, s.name, trace.WithTimestamp(s.start), trace.WithAttributes(attrs...))
				if s.err != nil {
					child.SetStatus(codes.Error, s.err.Error())
				}
				child.End(trace.WithTimestamp(s.end))
			}
		}
	}

	ctx = httptrace.WithClientTrace(context.WithValue(ctx, roundTripKey{}, rt), rt.clientTrace())
	resp, err := t.base.RoundTrip(req.WithContext(ctx))
	end := time.Now()

	way, reused := rt.connection()
	connWait, ttfb := rt.waits()
	attempts.observed(number, way, reused, connWait, ttfb)
	if recording {
		span.SetAttributes(describe()...)
		if resp != nil {
			answered(span, resp)
		}
	}
	if rt.timed {
		rt.write(rt.stepsAtReturn(end, err))
	}
	return resp, err
}

// tunnel is what a proxy's reply to CONNECT says of a connection's way.
// Zero without a proxy, or when the proxy says nothing.
type tunnel struct {
	// exit is the name the farthest proxy gave itself (Via).
	exit string
	// exitConnect is the proxy's connect to the target (Server-Timing), exitRTT
	// the rest of the CONNECT's round trip. Both nil without that time.
	exitRTT     *time.Duration
	exitConnect *time.Duration
}

// tunnelConn is a dialed connection and its way, kept on it for the pooled
// and multiplexed requests that never see the CONNECT.
type tunnelConn struct {
	net.Conn
	opened atomic.Pointer[tunnel]
}

func (c *tunnelConn) way() tunnel {
	if c == nil {
		return tunnel{}
	}
	if opened := c.opened.Load(); opened != nil {
		return *opened
	}
	return tunnel{}
}

// tunnelOf finds the dialed connection under the transport's TLS layers.
func tunnelOf(conn net.Conn) *tunnelConn {
	for conn != nil {
		switch c := conn.(type) {
		case *tunnelConn:
			return c
		case *tls.Conn:
			conn = c.NetConn()
		default:
			return nil
		}
	}
	return nil
}

// upstreamDialer is the dialer a Transport without a DialContext uses.
var upstreamDialer net.Dialer

// dialContext is Transport.DialContext: the default dial, returning a tunnelConn.
func dialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	rt := roundTripFrom(ctx)
	d := rt.dialStarted(ctx)
	conn, err := upstreamDialer.DialContext(ctx, network, addr)
	if err != nil {
		rt.dialEnded(d, nil, err)
		return nil, err
	}
	dialed := &tunnelConn{Conn: conn}
	rt.dialEnded(d, dialed, nil)
	return dialed, nil
}

// onProxyConnectResponse is Transport.OnProxyConnectResponse.
func onProxyConnectResponse(ctx context.Context, _ *url.URL, _ *http.Request, res *http.Response) error {
	roundTripFrom(ctx).tunnelReplied(ctx, viaReceivedBy(res.Header), serverTimingConnect(res.Header), res.StatusCode)
	return nil
}

// viaReceivedBy is the received-by of the first element of Via (RFC 9110,
// 7.6.3): the farthest proxy's name. "" without one.
func viaReceivedBy(header http.Header) string {
	first, _, _ := strings.Cut(header.Get("Via"), ",")
	fields := strings.Fields(first)
	if len(fields) < 2 || len(fields[1]) > maxExitLen {
		return ""
	}
	return fields[1]
}

// serverTimingConnect is the dur, in milliseconds, of the first metric
// named connect in Server-Timing. Nil without one.
func serverTimingConnect(header http.Header) *time.Duration {
	for _, line := range header.Values("Server-Timing") {
		for _, metric := range strings.Split(line, ",") {
			name, params, _ := strings.Cut(metric, ";")
			if !strings.EqualFold(strings.TrimSpace(name), "connect") {
				continue
			}
			for _, param := range strings.Split(params, ";") {
				key, value, _ := strings.Cut(param, "=")
				if !strings.EqualFold(strings.TrimSpace(key), "dur") {
					continue
				}
				ms, err := strconv.ParseFloat(strings.Trim(strings.TrimSpace(value), `"`), 64)
				// written so that a NaN fails it too
				if err != nil || !(ms >= 0 && ms <= float64(maxExitConnect/time.Millisecond)) {
					return nil
				}
				took := time.Duration(ms * float64(time.Millisecond))
				return &took
			}
			return nil
		}
	}
	return nil
}

type roundTripKey struct{}

// roundTrip is what one round trip learns about the connection it is sent
// on, from the transport's dial hooks and its ClientTrace.
type roundTrip struct {
	timed   bool         // an attempt whose span is recording: keep the steps' times
	tunnels bool         // the request goes through a proxy's CONNECT
	secure  bool         // the request is https
	write   func([]step) // writes steps as spans

	mu        sync.Mutex
	dials     []*dial     // the dials this round trip's context started: none on a pooled connection, rarely two
	tlsStart  time.Time   // the last TLS handshake of those dials
	tlsDone   time.Time   //
	tlsErr    error       //
	asked     time.Time   // a connection asked for
	conn      *tunnelConn // the connection the request was sent on
	gotConn   bool        //
	gotAt     time.Time   //
	reused    bool        //
	wrote     time.Time   // the request written
	firstByte time.Time   // the first byte of the response
}

// dial is one connection a round trip started. Go passes the dial's own
// context to DialContext and to its proxy hooks: that identity ties the
// CONNECT's reply to the connection.
type dial struct {
	ctx       context.Context //nolint:containedctx // kept for its identity only
	start     time.Time
	connected time.Time
	err       error
	conn      *tunnelConn
	asked     time.Time // the CONNECT about to be written
	tunneled  time.Time // the proxy's reply to CONNECT
	refused   error     // that reply, when it was not a 200
	written   bool      // its steps are spans already
}

func roundTripFrom(ctx context.Context) *roundTrip {
	rt, _ := ctx.Value(roundTripKey{}).(*roundTrip)
	return rt
}

func (rt *roundTrip) dialStarted(ctx context.Context) *dial {
	if rt == nil {
		return nil
	}
	d := &dial{ctx: ctx, start: time.Now()}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.dials = append(rt.dials, d)
	return d
}

func (rt *roundTrip) dialEnded(d *dial, conn *tunnelConn, err error) {
	if rt == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	d.connected, d.conn, d.err = time.Now(), conn, err
}

// dialOf is the dial whose hooks run with ctx. Should Go stop passing the
// dial's own context, it is the only dial awaiting a reply. The caller
// holds rt.mu.
func (rt *roundTrip) dialOf(ctx context.Context) *dial {
	var awaiting *dial
	waiting := 0
	for _, d := range rt.dials {
		if d.ctx == ctx {
			return d
		}
		if d.conn != nil && d.tunneled.IsZero() {
			awaiting = d
			waiting++
		}
	}
	if waiting != 1 {
		return nil
	}
	return awaiting
}

// connectAsked stamps the CONNECT's start: Go asks for its headers right
// before it writes it.
func (rt *roundTrip) connectAsked(ctx context.Context) {
	if rt == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if d := rt.dialOf(ctx); d != nil {
		d.asked = time.Now()
	}
}

// tunnelReplied records the proxy's reply to the CONNECT.
func (rt *roundTrip) tunnelReplied(ctx context.Context, exit string, exitConnect *time.Duration, status int) {
	if rt == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	replied := rt.dialOf(ctx)
	if replied == nil {
		return
	}
	replied.tunneled = time.Now()
	if status != http.StatusOK {
		replied.refused = fmt.Errorf("the proxy answered CONNECT with %d", status)
	}
	opened := tunnel{exit: exit}
	if exitConnect != nil && !replied.asked.IsZero() {
		// The exit's time is whole milliseconds and can round above the round trip.
		rtt := max(replied.tunneled.Sub(replied.asked)-*exitConnect, 0)
		opened.exitRTT, opened.exitConnect = &rtt, exitConnect
	}
	if opened != (tunnel{}) && replied.conn != nil {
		replied.conn.opened.Store(&opened)
	}
}

func (rt *roundTrip) clientTrace() *httptrace.ClientTrace {
	stamp := func(at *time.Time) {
		rt.mu.Lock()
		defer rt.mu.Unlock()
		*at = time.Now()
	}
	clientTrace := &httptrace.ClientTrace{
		// once: the transport asks again when a pooled connection turns out dead
		GetConn: func(string) {
			rt.mu.Lock()
			defer rt.mu.Unlock()
			if rt.asked.IsZero() {
				rt.asked = time.Now()
			}
		},
		GotConn: func(info httptrace.GotConnInfo) {
			rt.mu.Lock()
			rt.conn, rt.gotConn, rt.gotAt, rt.reused = tunnelOf(info.Conn), true, time.Now(), info.Reused
			var opened []step
			if rt.timed {
				opened = rt.dialSteps(time.Time{}, nil)
			}
			rt.mu.Unlock()
			if len(opened) > 0 {
				rt.write(opened)
			}
		},
	}
	clientTrace.WroteRequest = func(httptrace.WroteRequestInfo) { stamp(&rt.wrote) }
	clientTrace.GotFirstResponseByte = func() { stamp(&rt.firstByte) }
	if !rt.timed {
		return clientTrace
	}
	clientTrace.TLSHandshakeStart = func() {
		rt.mu.Lock()
		defer rt.mu.Unlock()
		rt.tlsStart, rt.tlsDone, rt.tlsErr = time.Now(), time.Time{}, nil
	}
	clientTrace.TLSHandshakeDone = func(_ tls.ConnectionState, err error) {
		rt.mu.Lock()
		defer rt.mu.Unlock()
		rt.tlsDone, rt.tlsErr = time.Now(), err
	}
	return clientTrace
}

// waits are the round trip's waits for a connection and for the first
// byte. Nil for one that did not end.
func (rt *roundTrip) waits() (connWait, ttfb *time.Duration) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.gotConn && !rt.asked.IsZero() {
		d := rt.gotAt.Sub(rt.asked)
		connWait = &d
	}
	if !rt.wrote.IsZero() && !rt.firstByte.Before(rt.wrote) {
		d := rt.firstByte.Sub(rt.wrote)
		ttfb = &d
	}
	return connWait, ttfb
}

// connection is the way of the connection the request was sent on and
// whether it was pooled: nil when it got none, then the way is its dial's.
func (rt *roundTrip) connection() (way tunnel, reused *bool) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.gotConn {
		pooled := rt.reused
		return rt.conn.way(), &pooled
	}
	if d := rt.ownDial(); d != nil {
		return d.conn.way(), nil
	}
	return tunnel{}, nil
}

// ownDial is the dial whose connection the request was sent on, or the last
// started when it got none. The caller holds rt.mu.
func (rt *roundTrip) ownDial() *dial {
	if !rt.gotConn {
		if len(rt.dials) == 0 {
			return nil
		}
		return rt.dials[len(rt.dials)-1]
	}
	for _, d := range rt.dials {
		if d.conn != nil && d.conn == rt.conn {
			return d
		}
	}
	return nil
}

type step struct {
	name       string
	start, end time.Time
	err        error
}

// dialSteps are the steps of the round trip's own dial, once. On a failure
// before a connection, the step under way ends at end. The caller holds
// rt.mu.
func (rt *roundTrip) dialSteps(end time.Time, failure error) []step {
	d := rt.ownDial()
	if d == nil || d.written {
		return nil
	}
	d.written = true
	var steps []step
	add := stepAdder(&steps, end, failure)
	add(stepConnect, d.start, d.connected, d.err)
	if d.conn != nil && rt.tunnels {
		add(stepTunnel, d.connected, d.tunneled, d.refused)
	}
	// With a tunnel the provider's handshake comes after the proxy's reply;
	// one before it was an https proxy's own, part of the tunnel.
	if d.conn != nil && rt.secure && (!rt.tunnels || (!d.tunneled.IsZero() && !rt.tlsStart.Before(d.tunneled))) {
		add(stepTLS, rt.tlsStart, rt.tlsDone, rt.tlsErr)
	}
	return steps
}

// stepsAtReturn are the steps not yet written when the round trip ended.
func (rt *roundTrip) stepsAtReturn(end time.Time, failure error) []step {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	steps := rt.dialSteps(end, failure)
	stepAdder(&steps, end, failure)(stepTTFB, rt.wrote, rt.firstByte, nil)
	return steps
}

// stepAdder appends a step that began. One not finished is left out, unless
// the round trip failed: then it ends at end, with that failure.
func stepAdder(steps *[]step, end time.Time, failure error) func(name string, start, done time.Time, err error) {
	return func(name string, start, done time.Time, err error) {
		if start.IsZero() {
			return
		}
		if done.Before(start) {
			if failure == nil {
				return
			}
			done, err = end, failure
		}
		*steps = append(*steps, step{name: name, start: start, end: done, err: err})
	}
}
