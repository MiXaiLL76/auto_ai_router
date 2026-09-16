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

func TestRawBodyEvent_Key(t *testing.T) {
	e := &RawBodyEvent{RequestID: "req-123"}
	assert.Equal(t, []byte("req-123"), e.Key())

	var nilEvent *RawBodyEvent
	assert.Nil(t, nilEvent.Key())
}

func TestRawBodyEvent_JSONMarshal_OmitsEmptyOptionalFields(t *testing.T) {
	startTime := time.Date(2026, 7, 15, 10, 0, 0, 0, time.UTC)
	endTime := time.Date(2026, 7, 15, 10, 0, 0, 250, time.UTC)
	e := &RawBodyEvent{
		RequestID:      "req-123",
		ServerRouterID: "air-ru02-abc123",
		StartTime:      startTime,
		EndTime:        endTime,
		HTTPStatus:     429,
	}

	data, err := json.Marshal(e)
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(data, &raw))

	assert.Equal(t, "req-123", raw["request_id"])
	assert.Equal(t, "air-ru02-abc123", raw["server_router_id"])
	assert.Contains(t, raw, "start_time")
	assert.Contains(t, raw, "end_time", "EndTime has no omitempty -- always present, same as SpendEvent.EndTime")
	assert.Equal(t, float64(429), raw["http_status"])
	_, hasErrorClass := raw["error_class"]
	assert.False(t, hasErrorClass)
	_, hasResponseBody := raw["response_body"]
	assert.False(t, hasResponseBody)
}

// TestRawBodyEvent_RequestBodyOmittedWhenEmpty locks in that RequestBody
// (only ever populated when KafkaRawBodiesConfig.StoreRawBody is
// explicitly enabled) doesn't appear in the JSON at all when the capture
// site left it unset, same omitempty behavior as ResponseBody.
func TestRawBodyEvent_RequestBodyOmittedWhenEmpty(t *testing.T) {
	e := RawBodyEvent{}
	data, err := json.Marshal(&e)
	require.NoError(t, err)
	assert.NotContains(t, string(data), "request_body")
}

func TestRawBodyEvent_JSONMarshal_IncludesRequestBodyWhenSet(t *testing.T) {
	e := &RawBodyEvent{
		RequestID:   "req-123",
		RequestBody: `{"messages":[{"role":"user","content":"hello"}]}`,
	}

	data, err := json.Marshal(e)
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(data, &raw))
	assert.Equal(t, e.RequestBody, raw["request_body"])
}

func TestRawBodyEvent_JSONMarshal_IncludesBodiesWhenSet(t *testing.T) {
	e := &RawBodyEvent{
		RequestID:          "req-123",
		ErrorClass:         "RateLimitError",
		ResponseBody:       `{"error":{"message":"rate limited"}}`,
		ClientResponseBody: `{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","param":null,"code":"rate_limit_error"}}`,
	}

	data, err := json.Marshal(e)
	require.NoError(t, err)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(data, &raw))

	assert.Equal(t, "RateLimitError", raw["error_class"])
	assert.Equal(t, `{"error":{"message":"rate limited"}}`, raw["response_body"])
	assert.Equal(t, e.ClientResponseBody, raw["client_response_body"])
	assert.NotEqual(t, raw["response_body"], raw["client_response_body"], "the whole point of this field is that it usually differs from response_body")
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
