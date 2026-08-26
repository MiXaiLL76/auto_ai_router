package telemetry

import (
	"context"
	"crypto/rand"

	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"go.opentelemetry.io/otel/trace"
)

// requestIDGenerator makes a root span's trace_id equal to the
// application's request_id (internal/requestid), so the ID a client sees in
// X-Request-Id resolves directly to its trace in the tracing backend.
//
// request_id is produced by requestid.New as 16 uniformly random bytes, so
// reusing it verbatim as the trace ID is safe for sdktrace's
// TraceIDRatioBased sampler, which reads TraceID[8:16] to make its decision.
//
// When no request_id is present in ctx (background jobs, startup spans) it
// falls back to a random trace ID, matching the SDK's default IDGenerator.
type requestIDGenerator struct{}

func (requestIDGenerator) NewIDs(ctx context.Context) (trace.TraceID, trace.SpanID) {
	var tid trace.TraceID
	if id := requestid.FromContext(ctx); id != "" {
		if parsed, err := trace.TraceIDFromHex(id); err == nil {
			tid = parsed
		}
	}
	if !tid.IsValid() {
		_, _ = rand.Read(tid[:])
	}
	return tid, randomSpanID()
}

func (requestIDGenerator) NewSpanID(_ context.Context, _ trace.TraceID) trace.SpanID {
	return randomSpanID()
}

func randomSpanID() trace.SpanID {
	var sid trace.SpanID
	_, _ = rand.Read(sid[:])
	return sid
}
