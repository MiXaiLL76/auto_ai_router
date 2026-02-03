package fail2ban

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	f2b := New(3, 5*time.Minute, []int{401, 403, 500})

	assert.NotNil(t, f2b)
	assert.Equal(t, 3, f2b.maxAttempts)
	assert.Equal(t, 5*time.Minute, f2b.banDuration)
	assert.True(t, f2b.errorCodes[401])
	assert.True(t, f2b.errorCodes[403])
	assert.True(t, f2b.errorCodes[500])
	assert.False(t, f2b.errorCodes[200])
}

func TestRecordResponse_Success(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record success - should not increment failures
	f2b.RecordResponse("cred1", 200)

	assert.Equal(t, 0, f2b.GetFailureCount("cred1"))
	assert.False(t, f2b.IsBanned("cred1"))
}

func TestRecordResponse_Error(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record error
	f2b.RecordResponse("cred1", 401)

	assert.Equal(t, 1, f2b.GetFailureCount("cred1"))
	assert.False(t, f2b.IsBanned("cred1"))
}

func TestRecordResponse_NonTrackedError(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record error code that's not tracked (404)
	f2b.RecordResponse("cred1", 404)

	assert.Equal(t, 0, f2b.GetFailureCount("cred1"))
	assert.False(t, f2b.IsBanned("cred1"))
}

func TestRecordResponse_BanAfterMaxAttempts(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record 2 errors of same type - not banned yet
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	assert.False(t, f2b.IsBanned("cred1"))
	assert.Equal(t, 2, f2b.GetFailureCount("cred1"))

	// 3rd error of same type - should ban
	f2b.RecordResponse("cred1", 401)
	assert.True(t, f2b.IsBanned("cred1"))
	assert.Equal(t, 3, f2b.GetFailureCount("cred1"))
}

func TestRecordResponse_SuccessResetCounter(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record 2 errors
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 403)
	assert.Equal(t, 2, f2b.GetFailureCount("cred1"))

	// Success should reset counter
	f2b.RecordResponse("cred1", 200)
	assert.Equal(t, 0, f2b.GetFailureCount("cred1"))
	assert.False(t, f2b.IsBanned("cred1"))
}

func TestIsBanned_NotBanned(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Credential never recorded
	assert.False(t, f2b.IsBanned("unknown_cred"))

	// Credential with less than max attempts
	f2b.RecordResponse("cred1", 401)
	assert.False(t, f2b.IsBanned("cred1"))
}

func TestIsBanned_PermanentBan(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	assert.True(t, f2b.IsBanned("cred1"))

	// Should still be banned (permanent)
	time.Sleep(100 * time.Millisecond)
	assert.True(t, f2b.IsBanned("cred1"))
}

func TestIsBanned_TemporaryBan_NotExpired(t *testing.T) {
	f2b := New(3, 200*time.Millisecond, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	assert.True(t, f2b.IsBanned("cred1"))

	// Check immediately - should still be banned
	time.Sleep(50 * time.Millisecond)
	assert.True(t, f2b.IsBanned("cred1"))
}

func TestIsBanned_TemporaryBan_Expired(t *testing.T) {
	f2b := New(3, 100*time.Millisecond, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	assert.True(t, f2b.IsBanned("cred1"))

	// Wait for ban to expire
	time.Sleep(150 * time.Millisecond)

	// Should be auto-unbanned
	assert.False(t, f2b.IsBanned("cred1"))
	assert.Equal(t, 0, f2b.GetFailureCount("cred1"))
}

func TestUnban(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	assert.True(t, f2b.IsBanned("cred1"))
	assert.Equal(t, 3, f2b.GetFailureCount("cred1"))

	// Manual unban
	f2b.Unban("cred1")

	assert.False(t, f2b.IsBanned("cred1"))
	assert.Equal(t, 0, f2b.GetFailureCount("cred1"))
}

func TestGetFailureCount(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	assert.Equal(t, 0, f2b.GetFailureCount("cred1"))

	f2b.RecordResponse("cred1", 401)
	assert.Equal(t, 1, f2b.GetFailureCount("cred1"))

	f2b.RecordResponse("cred1", 500)
	assert.Equal(t, 2, f2b.GetFailureCount("cred1"))
}

func TestGetBannedCredentials(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// No banned credentials initially
	banned := f2b.GetBannedCredentials()
	assert.Len(t, banned, 0)

	// Ban cred1
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	// Ban cred2
	f2b.RecordResponse("cred2", 500)
	f2b.RecordResponse("cred2", 500)
	f2b.RecordResponse("cred2", 500)

	banned = f2b.GetBannedCredentials()
	assert.Len(t, banned, 2)
	assert.Contains(t, banned, "cred1")
	assert.Contains(t, banned, "cred2")
}

func TestRecordResponse_IgnoreIfAlreadyBanned(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Ban credential
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	assert.True(t, f2b.IsBanned("cred1"))

	// Try to record more responses - should be ignored
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 200)

	// Failure count should remain at 3
	assert.Equal(t, 3, f2b.GetFailureCount("cred1"))
	assert.True(t, f2b.IsBanned("cred1"))
}

func TestConcurrency(t *testing.T) {
	f2b := New(100, 0, []int{401, 403, 500})

	var wg sync.WaitGroup
	numGoroutines := 50
	requestsPerGoroutine := 20

	// Concurrent RecordResponse
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(credID int) {
			defer wg.Done()
			credName := "cred" + string(rune('0'+credID%10))
			for j := 0; j < requestsPerGoroutine; j++ {
				if j%2 == 0 {
					f2b.RecordResponse(credName, 401)
				} else {
					f2b.RecordResponse(credName, 200)
				}
			}
		}(i)
	}

	// Concurrent IsBanned checks
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(credID int) {
			defer wg.Done()
			credName := "cred" + string(rune('0'+credID%10))
			for j := 0; j < requestsPerGoroutine; j++ {
				_ = f2b.IsBanned(credName)
				_ = f2b.GetFailureCount(credName)
			}
		}(i)
	}

	wg.Wait()

	// Verify no race conditions occurred (test passes if no panic)
	_ = f2b.GetBannedCredentials()
}

func TestMultipleCredentials(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record errors for multiple credentials
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred2", 403)
	f2b.RecordResponse("cred3", 500)

	assert.Equal(t, 1, f2b.GetFailureCount("cred1"))
	assert.Equal(t, 1, f2b.GetFailureCount("cred2"))
	assert.Equal(t, 1, f2b.GetFailureCount("cred3"))

	// Ban only cred1
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	assert.True(t, f2b.IsBanned("cred1"))
	assert.False(t, f2b.IsBanned("cred2"))
	assert.False(t, f2b.IsBanned("cred3"))
}

func TestRecordResponse_429_ImmediateBan(t *testing.T) {
	// Create fail2ban with default max_attempts=3, but 429 has max_attempts=1
	rules := []ErrorCodeRule{
		{
			Code:        429,
			MaxAttempts: 1,
			BanDuration: 10 * time.Second,
		},
	}

	f2b := NewWithRules(3, 0, []int{401, 403, 429, 500}, rules)

	// First 429 should immediately ban
	f2b.RecordResponse("vertex_v1", 429)
	assert.True(t, f2b.IsBanned("vertex_v1"))
	assert.Equal(t, 1, f2b.GetFailureCount("vertex_v1"))
}

func TestRecordResponse_429_TemporaryBan(t *testing.T) {
	// Create fail2ban with 429 ban duration = 100ms
	rules := []ErrorCodeRule{
		{
			Code:        429,
			MaxAttempts: 1,
			BanDuration: 100 * time.Millisecond,
		},
	}

	f2b := NewWithRules(3, 0, []int{401, 403, 429, 500}, rules)

	// First 429 should ban
	f2b.RecordResponse("vertex_v1", 429)
	assert.True(t, f2b.IsBanned("vertex_v1"))

	// Wait for ban to expire
	time.Sleep(150 * time.Millisecond)

	// Should be auto-unbanned
	assert.False(t, f2b.IsBanned("vertex_v1"))
	assert.Equal(t, 0, f2b.GetFailureCount("vertex_v1"))
}

func TestRecordResponse_OtherCodes_Unaffected(t *testing.T) {
	// Create fail2ban with 429 special rule (max_attempts=1)
	// but other codes still use default (max_attempts=3)
	rules := []ErrorCodeRule{
		{
			Code:        429,
			MaxAttempts: 1,
			BanDuration: 10 * time.Second,
		},
	}

	f2b := NewWithRules(3, 0, []int{401, 429, 500}, rules)

	// Record 1 error 500 - should not ban
	f2b.RecordResponse("cred1", 500)
	assert.False(t, f2b.IsBanned("cred1"))

	// Record 2 more errors 500 - should not ban yet (max_attempts=3 for 500)
	f2b.RecordResponse("cred1", 500)
	f2b.RecordResponse("cred1", 500)
	assert.True(t, f2b.IsBanned("cred1"))

	// Different credential: 429 should ban immediately (max_attempts=1 for 429)
	f2b.RecordResponse("cred2", 429)
	assert.True(t, f2b.IsBanned("cred2"))

	// Third error 500 on cred1 is ignored since already banned
	f2b.RecordResponse("cred1", 500)
	assert.True(t, f2b.IsBanned("cred1"))
}

func TestBackwardCompatibility(t *testing.T) {
	// Old-style Fail2Ban without error_code_rules should still work
	f2b := New(3, 5*time.Minute, []int{401, 403, 429, 500})

	// All error codes use same max_attempts
	f2b.RecordResponse("cred1", 429)
	f2b.RecordResponse("cred1", 429)
	f2b.RecordResponse("cred1", 429)

	// 429 should ban after 3 attempts (not 1)
	assert.True(t, f2b.IsBanned("cred1"))

	// New credential: 500 also uses max_attempts=3
	f2b.RecordResponse("cred2", 500)
	assert.False(t, f2b.IsBanned("cred2"))
	f2b.RecordResponse("cred2", 500)
	f2b.RecordResponse("cred2", 500)
	assert.True(t, f2b.IsBanned("cred2"))
}

func TestNewWithRules(t *testing.T) {
	rules := []ErrorCodeRule{
		{Code: 429, MaxAttempts: 1, BanDuration: 10 * time.Second},
		{Code: 503, MaxAttempts: 2, BanDuration: 30 * time.Second},
	}

	f2b := NewWithRules(3, 0, []int{401, 429, 503, 500}, rules)

	assert.NotNil(t, f2b)
	assert.Equal(t, 3, f2b.maxAttempts)

	// 429 uses rule (max_attempts=1)
	f2b.RecordResponse("cred1", 429)
	assert.True(t, f2b.IsBanned("cred1"))

	// 503 uses rule (max_attempts=2)
	f2b.RecordResponse("cred2", 503)
	assert.False(t, f2b.IsBanned("cred2"))
	f2b.RecordResponse("cred2", 503)
	assert.True(t, f2b.IsBanned("cred2"))

	// 500 uses default (max_attempts=3)
	f2b.RecordResponse("cred3", 500)
	f2b.RecordResponse("cred3", 500)
	assert.False(t, f2b.IsBanned("cred3"))
	f2b.RecordResponse("cred3", 500)
	assert.True(t, f2b.IsBanned("cred3"))
}

func TestMixedErrorCodesWithPerCodeRules(t *testing.T) {
	rules := []ErrorCodeRule{
		{Code: 429, MaxAttempts: 1, BanDuration: 10 * time.Second},
		{Code: 503, MaxAttempts: 2, BanDuration: 30 * time.Second},
	}

	f2b := NewWithRules(3, 0, []int{401, 429, 503, 500}, rules)

	// Verify that each error code has its own failure counter
	// but once a credential is banned (for ANY error code), it stays banned

	// 429 bans after 1 attempt
	f2b.RecordResponse("cred1", 429)
	assert.True(t, f2b.IsBanned("cred1"))

	// Once cred1 is banned, further errors don't change anything
	// (the ban is per-credential, not per-error-code)
	f2b.RecordResponse("cred1", 503)
	assert.True(t, f2b.IsBanned("cred1")) // Still banned from 429

	// Different credential: 503 needs 2 attempts to ban
	f2b.RecordResponse("cred2", 503)
	assert.False(t, f2b.IsBanned("cred2"))
	f2b.RecordResponse("cred2", 503)
	assert.True(t, f2b.IsBanned("cred2"))

	// Another credential: 500 (default rule) needs 3 attempts to ban
	f2b.RecordResponse("cred3", 500)
	assert.False(t, f2b.IsBanned("cred3"))
	f2b.RecordResponse("cred3", 500)
	assert.False(t, f2b.IsBanned("cred3"))
	f2b.RecordResponse("cred3", 500)
	assert.True(t, f2b.IsBanned("cred3"))
}
