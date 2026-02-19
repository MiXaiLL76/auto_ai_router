package proxyhealth

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsUnhealthy_UnknownProxy_ReturnsFalse(t *testing.T) {
	tracker := NewTracker()

	// Unknown proxy should be assumed healthy (fail-open behavior)
	result := tracker.IsUnhealthy("unknown-proxy")
	assert.False(t, result, "unknown proxy should not be reported as unhealthy (fail-open)")
}

func TestIsHealthy_UnknownProxy_ReturnsFalse(t *testing.T) {
	tracker := NewTracker()

	// Unknown proxy has no entry in healthStatus map, so IsHealthy returns false
	result := tracker.IsHealthy("unknown-proxy")
	assert.False(t, result, "unknown proxy should not be reported as healthy (no entry exists)")
}

func TestRecordFailure_MarksUnhealthy(t *testing.T) {
	tracker := NewTracker()
	proxyName := "test-proxy"

	// Initially unknown, not unhealthy (fail-open)
	assert.False(t, tracker.IsUnhealthy(proxyName))

	// Record a failure
	tracker.RecordFailure(proxyName, errors.New("connection refused"))

	// Now should be unhealthy
	assert.True(t, tracker.IsUnhealthy(proxyName), "proxy should be unhealthy after failure")
	assert.False(t, tracker.IsHealthy(proxyName), "proxy should not be healthy after failure")
	assert.Equal(t, 1, tracker.GetFailureCount(proxyName), "failure count should be 1")

	// Record another failure, verify count increments
	tracker.RecordFailure(proxyName, errors.New("timeout"))
	assert.True(t, tracker.IsUnhealthy(proxyName), "proxy should remain unhealthy after second failure")
	assert.Equal(t, 2, tracker.GetFailureCount(proxyName), "failure count should be 2")
}

func TestRecordSuccess_MarksHealthy(t *testing.T) {
	tracker := NewTracker()
	proxyName := "test-proxy"

	// First mark as unhealthy via failure
	tracker.RecordFailure(proxyName, errors.New("connection refused"))
	assert.True(t, tracker.IsUnhealthy(proxyName), "proxy should be unhealthy after failure")

	// Now record success
	tracker.RecordSuccess(proxyName)

	// Should be healthy again
	assert.True(t, tracker.IsHealthy(proxyName), "proxy should be healthy after success")
	assert.False(t, tracker.IsUnhealthy(proxyName), "proxy should not be unhealthy after success")
	assert.Equal(t, 0, tracker.GetFailureCount(proxyName), "failure count should be reset to 0")
}
