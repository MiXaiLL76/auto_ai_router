package kafkalog

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSpendEvent_Key(t *testing.T) {
	e := &SpendEvent{RequestID: "req-123"}
	assert.Equal(t, []byte("req-123"), e.Key())

	var nilEvent *SpendEvent
	assert.Nil(t, nilEvent.Key())
}

func TestSpendEvent_JSONMarshal_OmitsNilOptionalFields(t *testing.T) {
	e := &SpendEvent{
		RequestID: "req-123",
		StartTime: time.Date(2026, 7, 15, 10, 0, 0, 0, time.UTC),
		EndTime:   time.Date(2026, 7, 15, 10, 0, 1, 0, time.UTC),
		Status:    "success",
	}

	data, err := json.Marshal(e)
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(data, &raw))

	// Nullable fields must be absent when not set (streaming didn't happen).
	_, hasCompletionStart := raw["completion_start_time"]
	assert.False(t, hasCompletionStart)
	_, hasTTFT := raw["ttft_ms"]
	assert.False(t, hasTTFT)

	// Body placeholder fields are always present with zero values.
	assert.Equal(t, false, raw["body_captured"])
	assert.Equal(t, float64(0), raw["body_request_bytes"])
	assert.Equal(t, float64(0), raw["body_response_bytes"])
}

func TestSpendEvent_JSONMarshal_IncludesTTFTWhenSet(t *testing.T) {
	completionStart := time.Date(2026, 7, 15, 10, 0, 0, 300, time.UTC)
	ttft := int64(300)
	e := &SpendEvent{
		RequestID:           "req-123",
		CompletionStartTime: &completionStart,
		TTFTMs:              &ttft,
	}

	data, err := json.Marshal(e)
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(data, &raw))

	assert.Equal(t, float64(300), raw["ttft_ms"])
	assert.Contains(t, raw, "completion_start_time")
}
