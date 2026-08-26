package telemetry

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequestIDGeneratorUsesRequestIDAsTraceID(t *testing.T) {
	id := uuid.NewString()
	ctx := requestid.WithID(context.Background(), id)

	tid, sid := requestIDGenerator{}.NewIDs(ctx)

	require.True(t, tid.IsValid())
	assert.Equal(t, id, uuid.UUID(tid).String())
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
