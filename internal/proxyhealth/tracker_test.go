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

func TestGetFailedNames(t *testing.T) {
	tracker := NewTracker()

	// Two failed proxies
	tracker.RecordFailure("proxy-a", errors.New("err"))
	tracker.RecordFailure("proxy-b", errors.New("err"))
	// One healthy proxy
	tracker.RecordSuccess("proxy-c")

	failed := tracker.GetFailedNames()
	assert.Len(t, failed, 2)
	assert.Contains(t, failed, "proxy-a")
	assert.Contains(t, failed, "proxy-b")
}

func TestGetRecoveredNames(t *testing.T) {
	tracker := NewTracker()

	// Fail, then succeed → recovered
	tracker.RecordFailure("proxy-a", errors.New("err"))
	tracker.RecordSuccess("proxy-a")

	// First call should return the recovered proxy
	recovered := tracker.GetRecoveredNames()
	assert.Len(t, recovered, 1)
	assert.Contains(t, recovered, "proxy-a")

	// Second call should return empty (consumed)
	recovered2 := tracker.GetRecoveredNames()
	assert.Empty(t, recovered2, "recovered should be consumed after first call")
}

func TestGetStatus_And_GetAllStatuses(t *testing.T) {
	tracker := NewTracker()

	// Setup 3 proxies in different states
	tracker.RecordSuccess("proxy-healthy")
	tracker.RecordFailure("proxy-failed", errors.New("err"))
	tracker.RecordFailure("proxy-failed", errors.New("err")) // 2 failures
	// proxy-unknown is not recorded at all

	// GetStatus
	healthy, failCount := tracker.GetStatus("proxy-healthy")
	assert.True(t, healthy)
	assert.Equal(t, 0, failCount)

	healthy, failCount = tracker.GetStatus("proxy-failed")
	assert.False(t, healthy)
	assert.Equal(t, 2, failCount)

	healthy, failCount = tracker.GetStatus("proxy-unknown")
	assert.False(t, healthy) // no entry
	assert.Equal(t, 0, failCount)

	// GetAllStatuses
	all := tracker.GetAllStatuses()
	assert.True(t, all["proxy-healthy"])
	assert.False(t, all["proxy-failed"])

	// Verify it returns a copy (modifying snapshot doesn't affect tracker)
	all["proxy-healthy"] = false
	allAgain := tracker.GetAllStatuses()
	assert.True(t, allAgain["proxy-healthy"], "original tracker should not be affected")

	// ResetFailureCount
	tracker.ResetFailureCount("proxy-failed")
	assert.Equal(t, 0, tracker.GetFailureCount("proxy-failed"))
}
