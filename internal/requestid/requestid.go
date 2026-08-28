// Package requestid assigns one ID per incoming HTTP request that is used
// as: the OTel trace_id for that request's server span (see
// internal/telemetry's IDGenerator), the value logged as request_id
// throughout internal/proxy, and the X-Request-Id header echoed to the
// client — so a client can hand support a single ID that resolves to the
// full trace end-to-end.
package requestid

import (
	"context"
	"crypto/rand"
	"net/http"

	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// Header is the response header carrying the request ID back to the client.
const Header = "X-Request-Id"

type contextKey struct{}

// New generates a fresh request ID: 16 uniformly random bytes rendered as a
// lowercase 32-hex string via trace.TraceID.
//
// The bytes must be uniformly random (not a UUIDv4, whose version/variant
// nibbles are pinned) because this ID is reused verbatim as the OTel trace
// ID (see internal/telemetry's IDGenerator), and sdktrace's
// TraceIDRatioBased sampler reads TraceID[8:16] to make its decision — fixed
// bits there bias or, for common ratios, completely break ratio sampling.
func New() string {
	var b trace.TraceID
	_, _ = rand.Read(b[:])
	return b.String()
}

// WithID returns a context carrying id, retrievable via FromContext.
func WithID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, contextKey{}, id)
}

// FromContext returns the request ID stored in ctx, or "" if none.
func FromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	id, _ := ctx.Value(contextKey{}).(string)
	return id
}

// Valid reports whether id is a valid lowercase 32-hex trace ID.
func Valid(id string) bool {
	return Canonical(id) != ""
}

// Canonical returns id as a normalized trace ID, or "" when invalid.
func Canonical(id string) string {
	tid, err := trace.TraceIDFromHex(id)
	if err != nil || !tid.IsValid() {
		return ""
	}
	canonical := tid.String()
	if canonical != id {
		return ""
	}
	return canonical
}

var traceparentPropagator = propagation.TraceContext{}

// fromTraceparent extracts the trace ID from a trusted inbound W3C
// traceparent header. When present, otelhttp creates a *child* span
// continuing that trace instead of asking the IDGenerator for a new one, so
// reusing it as our request ID is the only way to keep request_id and
// trace_id aligned for that request; minting an unrelated ID here would
// silently break the correlation the feature exists to provide.
func fromTraceparent(r *http.Request) string {
	sc := trace.SpanContextFromContext(traceparentPropagator.Extract(r.Context(), propagation.HeaderCarrier(r.Header)))
	if !sc.TraceID().IsValid() {
		return ""
	}
	return sc.TraceID().String()
}

// Middleware assigns every incoming request an ID, makes it available to
// handlers (and to the OTel IDGenerator) via the request context, and
// echoes it back to the client via the X-Request-Id header.
//
// trustIncomingTraceparent must mirror the effective
// otel.trust_incoming_traceparent setting (i.e. also false when tracing
// itself is disabled): when true, a trusted inbound traceparent's trace ID
// is reused as the request ID instead of minting a new, unrelated one — see
// fromTraceparent.
//
// The returned middleware must wrap the otelhttp handler (run before it),
// not be wrapped by it, so the ID is already in context by the time otelhttp
// starts the server span.
func Middleware(trustIncomingTraceparent bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := ""
			if trustIncomingTraceparent {
				id = fromTraceparent(r)
			}
			if id == "" {
				id = New()
			}
			w.Header().Set(Header, id)
			next.ServeHTTP(w, r.WithContext(WithID(r.Context(), id)))
		})
	}
}
