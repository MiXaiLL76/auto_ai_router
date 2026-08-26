// Package requestid assigns one UUID per incoming HTTP request that is used
// as: the OTel trace_id for that request's server span (see
// internal/telemetry's IDGenerator), the value logged as request_id
// throughout internal/proxy, and the X-Request-Id header echoed to the
// client — so a client can hand support a single ID that resolves to the
// full trace end-to-end.
package requestid

import (
	"context"
	"net/http"

	"github.com/google/uuid"
)

// Header is the response header carrying the request ID back to the client.
const Header = "X-Request-Id"

type contextKey struct{}

// New generates a fresh request ID.
func New() string {
	return uuid.NewString()
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

// Middleware assigns every incoming request a fresh ID, makes it available
// to handlers (and to the OTel IDGenerator) via the request context, and
// echoes it back to the client via the X-Request-Id header. It must wrap the
// otelhttp handler (run before it), not be wrapped by it, so the ID is
// already in context by the time otelhttp starts the server span.
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := New()
		w.Header().Set(Header, id)
		next.ServeHTTP(w, r.WithContext(WithID(r.Context(), id)))
	})
}
