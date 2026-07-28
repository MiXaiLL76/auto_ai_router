package ratelimit

import (
	"context"
	"testing"
)

// BenchmarkLocalCheckRPM_Unlimited exercises the hot path used by the balancer's
// precheck (canAllowRPM) for a credential with no RPM limit. Regardless of how
// many requests have already been recorded for the key (b.N grows across runs),
// each call should cost the same small constant amount of work, since cleanup
// is bounded by the fixed number of one-second buckets rather than the number
// of requests seen so far.
func BenchmarkLocalCheckRPM_Unlimited(b *testing.B) {
	backend := newLocalBackend()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backend.canAllowRPM(ctx, "hot-key", -1)
	}
}

// BenchmarkLocalTryAllowAll_ManyCandidates simulates the balancer's per-request
// admission loop: one credential-level + one model-level check, repeated as if
// scanning several candidate credentials before picking one — the exact shape
// that dominated the profiled CPU trace before the fix.
func BenchmarkLocalTryAllowAll_ManyCandidates(b *testing.B) {
	backend := newLocalBackend()
	ctx := context.Background()
	const numCandidates = 10

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for c := 0; c < numCandidates; c++ {
			backend.canAllowRPM(ctx, "cred", -1)
			backend.canAllowRPM(ctx, "model", -1)
		}
		backend.tryAllowAll(ctx, "cred", -1, -1, "model", -1, -1)
	}
}

// BenchmarkLocalCheckRPM_Parallel hammers a single hot key from many goroutines
// concurrently, matching production traffic against one popular credential.
func BenchmarkLocalCheckRPM_Parallel(b *testing.B) {
	backend := newLocalBackend()
	ctx := context.Background()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			backend.tryAllowRPM(ctx, "hot-key", -1)
		}
	})
}
