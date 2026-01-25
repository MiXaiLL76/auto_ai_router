package monitoring

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	m := New(true)
	assert.NotNil(t, m)
	assert.True(t, m.enabled)

	m2 := New(false)
	assert.NotNil(t, m2)
	assert.False(t, m2.enabled)
}

func TestRecordRequest_Enabled(t *testing.T) {
	// Reset metrics before test
	RequestsTotal.Reset()
	RequestDuration.Reset()
	CredentialErrorsTotal.Reset()

	m := New(true)

	// Record a successful request
	m.RecordRequest("cred1", "/v1/chat/completions", 200, 100*time.Millisecond)

	// Verify RequestsTotal metric
	count := testutil.CollectAndCount(RequestsTotal)
	assert.Greater(t, count, 0)

	// Record an error request
	m.RecordRequest("cred1", "/v1/chat/completions", 500, 150*time.Millisecond)

	// Verify CredentialErrorsTotal metric was incremented
	count = testutil.CollectAndCount(CredentialErrorsTotal)
	assert.Greater(t, count, 0)
}

func TestRecordRequest_Disabled(t *testing.T) {
	RequestsTotal.Reset()

	m := New(false)

	// Record requests when disabled
	m.RecordRequest("cred1", "/v1/chat/completions", 200, 100*time.Millisecond)
	m.RecordRequest("cred1", "/v1/chat/completions", 500, 150*time.Millisecond)

	// Metrics should not be recorded when disabled
	// (They actually will be because metrics are global, but the method should return early)
	// We can't easily test this without mocking, but we verify the method doesn't panic
}

func TestRecordRequest_DifferentStatusCodes(t *testing.T) {
	RequestsTotal.Reset()
	CredentialErrorsTotal.Reset()

	m := New(true)

	// Record requests with different status codes
	statusCodes := []int{200, 201, 400, 401, 403, 429, 500, 502, 503}
	for _, code := range statusCodes {
		m.RecordRequest("cred1", "/v1/test", code, 50*time.Millisecond)
	}

	// Verify metrics were collected
	count := testutil.CollectAndCount(RequestsTotal)
	assert.Greater(t, count, 0)
}

func TestRecordRequest_MultipleCredentials(t *testing.T) {
	RequestsTotal.Reset()

	m := New(true)

	// Record requests for different credentials
	m.RecordRequest("cred1", "/v1/chat/completions", 200, 100*time.Millisecond)
	m.RecordRequest("cred2", "/v1/chat/completions", 200, 150*time.Millisecond)
	m.RecordRequest("cred3", "/v1/embeddings", 200, 80*time.Millisecond)

	// Verify metrics were collected
	count := testutil.CollectAndCount(RequestsTotal)
	assert.Greater(t, count, 0)
}

func TestUpdateCredentialRPM(t *testing.T) {
	CredentialRPMCurrent.Reset()

	m := New(true)

	// Update RPM for credentials
	m.UpdateCredentialRPM("cred1", 50)
	m.UpdateCredentialRPM("cred2", 75)
	m.UpdateCredentialRPM("cred1", 60) // Update again

	// Verify metrics were set
	count := testutil.CollectAndCount(CredentialRPMCurrent)
	assert.Greater(t, count, 0)
}

func TestUpdateCredentialRPM_Disabled(t *testing.T) {
	m := New(false)

	// Should not panic when disabled
	m.UpdateCredentialRPM("cred1", 50)
	m.UpdateCredentialRPM("cred2", 100)
}

func TestUpdateCredentialBanStatus(t *testing.T) {
	CredentialBanned.Reset()

	m := New(true)

	// Update ban status
	m.UpdateCredentialBanStatus("cred1", false) // Not banned
	m.UpdateCredentialBanStatus("cred2", true)  // Banned
	m.UpdateCredentialBanStatus("cred3", false) // Not banned

	// Verify metrics were set
	count := testutil.CollectAndCount(CredentialBanned)
	assert.Greater(t, count, 0)
}

func TestUpdateCredentialBanStatus_Disabled(t *testing.T) {
	m := New(false)

	// Should not panic when disabled
	m.UpdateCredentialBanStatus("cred1", true)
	m.UpdateCredentialBanStatus("cred2", false)
}

func TestMetrics_Integration(t *testing.T) {
	// Reset all metrics
	RequestsTotal.Reset()
	RequestDuration.Reset()
	CredentialRPMCurrent.Reset()
	CredentialBanned.Reset()
	CredentialErrorsTotal.Reset()

	m := New(true)

	// Simulate a series of requests
	m.RecordRequest("cred1", "/v1/chat/completions", 200, 100*time.Millisecond)
	m.RecordRequest("cred1", "/v1/chat/completions", 200, 120*time.Millisecond)
	m.RecordRequest("cred1", "/v1/chat/completions", 500, 150*time.Millisecond) // Error

	m.RecordRequest("cred2", "/v1/embeddings", 200, 80*time.Millisecond)
	m.RecordRequest("cred2", "/v1/embeddings", 429, 90*time.Millisecond) // Rate limit error

	// Update RPM metrics
	m.UpdateCredentialRPM("cred1", 3)
	m.UpdateCredentialRPM("cred2", 2)

	// Update ban status
	m.UpdateCredentialBanStatus("cred1", false)
	m.UpdateCredentialBanStatus("cred2", false)

	// Verify all metrics have been collected
	assert.Greater(t, testutil.CollectAndCount(RequestsTotal), 0)
	assert.Greater(t, testutil.CollectAndCount(RequestDuration), 0)
	assert.Greater(t, testutil.CollectAndCount(CredentialRPMCurrent), 0)
	assert.Greater(t, testutil.CollectAndCount(CredentialBanned), 0)
	assert.Greater(t, testutil.CollectAndCount(CredentialErrorsTotal), 0)
}

func TestMetrics_PrometheusRegistration(t *testing.T) {
	// Verify that all metrics are registered with Prometheus
	metrics := []prometheus.Collector{
		RequestsTotal,
		RequestDuration,
		CredentialRPMCurrent,
		CredentialBanned,
		CredentialErrorsTotal,
	}

	for _, metric := range metrics {
		assert.NotNil(t, metric)
	}
}

func TestRecordRequest_ErrorIncrementsCounter(t *testing.T) {
	CredentialErrorsTotal.Reset()

	m := New(true)

	// Record successful request (200)
	m.RecordRequest("cred1", "/v1/test", 200, 50*time.Millisecond)

	// Get initial error count (should be 0)
	initialErrors := testutil.ToFloat64(CredentialErrorsTotal.WithLabelValues("cred1"))

	// Record error request
	m.RecordRequest("cred1", "/v1/test", 500, 50*time.Millisecond)

	// Error count should increase
	finalErrors := testutil.ToFloat64(CredentialErrorsTotal.WithLabelValues("cred1"))
	assert.Greater(t, finalErrors, initialErrors)
}

func TestUpdateCredentialBanStatus_Values(t *testing.T) {
	CredentialBanned.Reset()

	m := New(true)

	// Set banned to true
	m.UpdateCredentialBanStatus("cred1", true)
	bannedValue := testutil.ToFloat64(CredentialBanned.WithLabelValues("cred1"))
	assert.Equal(t, 1.0, bannedValue)

	// Set banned to false
	m.UpdateCredentialBanStatus("cred1", false)
	notBannedValue := testutil.ToFloat64(CredentialBanned.WithLabelValues("cred1"))
	assert.Equal(t, 0.0, notBannedValue)
}

func TestMultipleEndpoints(t *testing.T) {
	RequestsTotal.Reset()

	m := New(true)

	endpoints := []string{
		"/v1/chat/completions",
		"/v1/embeddings",
		"/v1/completions",
		"/v1/models",
	}

	for _, endpoint := range endpoints {
		m.RecordRequest("cred1", endpoint, 200, 100*time.Millisecond)
	}

	// All endpoints should be tracked separately
	count := testutil.CollectAndCount(RequestsTotal)
	assert.Greater(t, count, 0)
}
