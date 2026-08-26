package telemetry

import (
	"context"
	"crypto/rand"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"go.opentelemetry.io/otel/trace"
)

// requestIDGenerator makes a root span's trace_id equal to the
// application's request_id (internal/requestid), so the ID a client sees in
// X-Request-Id resolves directly to its trace in the tracing backend.
//
// uuid.UUID and trace.TraceID are both [16]byte, so the request UUID is used
// verbatim as the trace ID — no hashing or truncation needed. When no
// request_id is present in ctx (background jobs, startup spans, requests
// that continue an inbound traceparent) it falls back to a random trace ID,
// matching the SDK's default IDGenerator.
type requestIDGenerator struct{}

func (requestIDGenerator) NewIDs(ctx context.Context) (trace.TraceID, trace.SpanID) {
	var tid trace.TraceID
	if id := requestid.FromContext(ctx); id != "" {
		if u, err := uuid.Parse(id); err == nil {
			tid = trace.TraceID(u)
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
