package telemetry

import (
	"context"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func TestRequestIDGeneratorUsesRequestIDAsTraceID(t *testing.T) {
	id := requestid.New()
	ctx := requestid.WithID(context.Background(), id)

	tid, sid := requestIDGenerator{}.NewIDs(ctx)

	require.True(t, tid.IsValid())
	assert.Equal(t, id, tid.String())
	assert.True(t, sid.IsValid())
}

func TestRequestIDGeneratorFallsBackToRandomWithoutRequestID(t *testing.T) {
	tid, sid := requestIDGenerator{}.NewIDs(context.Background())

	assert.True(t, tid.IsValid())
	assert.True(t, sid.IsValid())
}

func TestRequestIDGeneratorNewSpanIDIsRandom(t *testing.T) {
	var tid [16]byte
	sid1 := requestIDGenerator{}.NewSpanID(context.Background(), tid)
	sid2 := requestIDGenerator{}.NewSpanID(context.Background(), tid)

	assert.True(t, sid1.IsValid())
	assert.NotEqual(t, sid1, sid2)
}

// Regression test: a UUIDv4 has its version/variant nibbles fixed inside
// TraceID[8:16], which is exactly the range sdktrace.TraceIDRatioBased reads
// to make its decision — using one verbatim as the trace ID silently drops
// every trace for any ratio <= 0.5. requestid.New's bytes must stay
// uniformly random so ratio sampling keeps working at realistic ratios.
func TestRequestIDGeneratorTraceIDsSurviveRatioSampling(t *testing.T) {
	const ratio = 0.1
	const trials = 2000
	sampler := sdktrace.TraceIDRatioBased(ratio)

	sampled := 0
	for range trials {
		tid, _ := requestIDGenerator{}.NewIDs(requestid.WithID(context.Background(), requestid.New()))
		result := sampler.ShouldSample(sdktrace.SamplingParameters{TraceID: tid})
		if result.Decision == sdktrace.RecordAndSample {
			sampled++
		}
	}

	// Expect roughly ratio*trials (~200); a tight range would be flaky, but
	// the old bug produced exactly 0 for any ratio <= 0.5, so a low floor is
	// enough to catch a regression without being sensitive to randomness.
	assert.Greater(t, sampled, trials/20, "ratio sampling should not silently drop every trace")
}
