package fail2ban

import (
	"sort"
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
	f2b.RecordResponse("cred1", "gpt-4", 200)

	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "gpt-4"))
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestRecordResponse_Error(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record error
	f2b.RecordResponse("cred1", "gpt-4", 401)

	assert.Equal(t, 1, f2b.GetFailureCount("cred1", "gpt-4"))
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestRecordResponse_NonTrackedError(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record error code that's not tracked (404)
	f2b.RecordResponse("cred1", "gpt-4", 404)

	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "gpt-4"))
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestRecordResponse_BanAfterMaxAttempts(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record 2 errors of same type - not banned yet
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
	assert.Equal(t, 2, f2b.GetFailureCount("cred1", "gpt-4"))

	// 3rd error of same type - should ban
	f2b.RecordResponse("cred1", "gpt-4", 401)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
	assert.Equal(t, 3, f2b.GetFailureCount("cred1", "gpt-4"))
}

func TestRecordResponse_SuccessResetCounter(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record 2 errors
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 403)
	assert.Equal(t, 2, f2b.GetFailureCount("cred1", "gpt-4"))

	// Success should reset counter
	f2b.RecordResponse("cred1", "gpt-4", 200)
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "gpt-4"))
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestIsBanned_NotBanned(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Credential never recorded
	assert.False(t, f2b.IsBanned("unknown_cred", "gpt-4"))

	// Credential with less than max attempts
	f2b.RecordResponse("cred1", "gpt-4", 401)
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestIsBanned_PermanentBan(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// Should still be banned (permanent)
	time.Sleep(100 * time.Millisecond)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestIsBanned_TemporaryBan_NotExpired(t *testing.T) {
	f2b := New(3, 200*time.Millisecond, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// Check immediately - should still be banned
	time.Sleep(50 * time.Millisecond)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestIsBanned_TemporaryBan_Expired(t *testing.T) {
	f2b := New(3, 100*time.Millisecond, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// Wait for ban to expire
	time.Sleep(150 * time.Millisecond)

	// Should be auto-unbanned
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "gpt-4"))
}

func TestUnban(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Trigger ban
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
	assert.Equal(t, 3, f2b.GetFailureCount("cred1", "gpt-4"))

	// Manual unban
	f2b.Unban("cred1", "gpt-4")

	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "gpt-4"))
}

func TestGetFailureCount(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "gpt-4"))

	f2b.RecordResponse("cred1", "gpt-4", 401)
	assert.Equal(t, 1, f2b.GetFailureCount("cred1", "gpt-4"))

	f2b.RecordResponse("cred1", "gpt-4", 500)
	assert.Equal(t, 2, f2b.GetFailureCount("cred1", "gpt-4"))
}

func TestGetBannedPairs(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// No banned pairs initially
	banned := f2b.GetBannedPairs()
	assert.Len(t, banned, 0)

	// Ban cred1|gpt-4
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	// Ban cred2|gpt-4
	f2b.RecordResponse("cred2", "gpt-4", 500)
	f2b.RecordResponse("cred2", "gpt-4", 500)
	f2b.RecordResponse("cred2", "gpt-4", 500)

	banned = f2b.GetBannedPairs()
	assert.Len(t, banned, 2)

	// Collect credentials from pairs
	credentials := make([]string, len(banned))
	for i, pair := range banned {
		credentials[i] = pair.Credential
	}
	assert.Contains(t, credentials, "cred1")
	assert.Contains(t, credentials, "cred2")

	// Verify fields are populated
	for _, pair := range banned {
		assert.Equal(t, "gpt-4", pair.Model)
		assert.NotZero(t, pair.ErrorCode)
		assert.NotEmpty(t, pair.ErrorCodeCounts)
		assert.False(t, pair.BanTime.IsZero())
	}
}

func TestRecordResponse_IgnoreIfAlreadyBanned(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Ban credential
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// Try to record more responses - should be ignored
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 200)

	// Failure count should remain at 3
	assert.Equal(t, 3, f2b.GetFailureCount("cred1", "gpt-4"))
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
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
			modelName := "model" + string(rune('0'+credID%3))
			for j := 0; j < requestsPerGoroutine; j++ {
				if j%2 == 0 {
					f2b.RecordResponse(credName, modelName, 401)
				} else {
					f2b.RecordResponse(credName, modelName, 200)
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
			modelName := "model" + string(rune('0'+credID%3))
			for j := 0; j < requestsPerGoroutine; j++ {
				_ = f2b.IsBanned(credName, modelName)
				_ = f2b.GetFailureCount(credName, modelName)
			}
		}(i)
	}

	wg.Wait()

	// Verify no race conditions occurred (test passes if no panic)
	_ = f2b.GetBannedPairs()
}

func TestMultipleCredentials(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Record errors for multiple credentials
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred2", "gpt-4", 403)
	f2b.RecordResponse("cred3", "gpt-4", 500)

	assert.Equal(t, 1, f2b.GetFailureCount("cred1", "gpt-4"))
	assert.Equal(t, 1, f2b.GetFailureCount("cred2", "gpt-4"))
	assert.Equal(t, 1, f2b.GetFailureCount("cred3", "gpt-4"))

	// Ban only cred1
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
	assert.False(t, f2b.IsBanned("cred2", "gpt-4"))
	assert.False(t, f2b.IsBanned("cred3", "gpt-4"))
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
	f2b.RecordResponse("vertex_v1", "gpt-4", 429)
	assert.True(t, f2b.IsBanned("vertex_v1", "gpt-4"))
	assert.Equal(t, 1, f2b.GetFailureCount("vertex_v1", "gpt-4"))
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
	f2b.RecordResponse("vertex_v1", "gpt-4", 429)
	assert.True(t, f2b.IsBanned("vertex_v1", "gpt-4"))

	// Wait for ban to expire
	time.Sleep(150 * time.Millisecond)

	// Should be auto-unbanned
	assert.False(t, f2b.IsBanned("vertex_v1", "gpt-4"))
	assert.Equal(t, 0, f2b.GetFailureCount("vertex_v1", "gpt-4"))
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
	f2b.RecordResponse("cred1", "gpt-4", 500)
	assert.False(t, f2b.IsBanned("cred1", "gpt-4"))

	// Record 2 more errors 500 - should ban (max_attempts=3 for 500)
	f2b.RecordResponse("cred1", "gpt-4", 500)
	f2b.RecordResponse("cred1", "gpt-4", 500)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// Different credential: 429 should ban immediately (max_attempts=1 for 429)
	f2b.RecordResponse("cred2", "gpt-4", 429)
	assert.True(t, f2b.IsBanned("cred2", "gpt-4"))

	// Third error 500 on cred1 is ignored since already banned
	f2b.RecordResponse("cred1", "gpt-4", 500)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestBackwardCompatibility(t *testing.T) {
	// Old-style Fail2Ban without error_code_rules should still work
	f2b := New(3, 5*time.Minute, []int{401, 403, 429, 500})

	// All error codes use same max_attempts
	f2b.RecordResponse("cred1", "gpt-4", 429)
	f2b.RecordResponse("cred1", "gpt-4", 429)
	f2b.RecordResponse("cred1", "gpt-4", 429)

	// 429 should ban after 3 attempts (not 1)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// New credential: 500 also uses max_attempts=3
	f2b.RecordResponse("cred2", "gpt-4", 500)
	assert.False(t, f2b.IsBanned("cred2", "gpt-4"))
	f2b.RecordResponse("cred2", "gpt-4", 500)
	f2b.RecordResponse("cred2", "gpt-4", 500)
	assert.True(t, f2b.IsBanned("cred2", "gpt-4"))
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
	f2b.RecordResponse("cred1", "gpt-4", 429)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// 503 uses rule (max_attempts=2)
	f2b.RecordResponse("cred2", "gpt-4", 503)
	assert.False(t, f2b.IsBanned("cred2", "gpt-4"))
	f2b.RecordResponse("cred2", "gpt-4", 503)
	assert.True(t, f2b.IsBanned("cred2", "gpt-4"))

	// 500 uses default (max_attempts=3)
	f2b.RecordResponse("cred3", "gpt-4", 500)
	f2b.RecordResponse("cred3", "gpt-4", 500)
	assert.False(t, f2b.IsBanned("cred3", "gpt-4"))
	f2b.RecordResponse("cred3", "gpt-4", 500)
	assert.True(t, f2b.IsBanned("cred3", "gpt-4"))
}

func TestMixedErrorCodesWithPerCodeRules(t *testing.T) {
	rules := []ErrorCodeRule{
		{Code: 429, MaxAttempts: 1, BanDuration: 10 * time.Second},
		{Code: 503, MaxAttempts: 2, BanDuration: 30 * time.Second},
	}

	f2b := NewWithRules(3, 0, []int{401, 429, 503, 500}, rules)

	// Verify that each error code has its own failure counter
	// but once a credential+model pair is banned (for ANY error code), it stays banned

	// 429 bans after 1 attempt
	f2b.RecordResponse("cred1", "gpt-4", 429)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))

	// Once cred1|gpt-4 is banned, further errors don't change anything
	f2b.RecordResponse("cred1", "gpt-4", 503)
	assert.True(t, f2b.IsBanned("cred1", "gpt-4")) // Still banned from 429

	// Different credential: 503 needs 2 attempts to ban
	f2b.RecordResponse("cred2", "gpt-4", 503)
	assert.False(t, f2b.IsBanned("cred2", "gpt-4"))
	f2b.RecordResponse("cred2", "gpt-4", 503)
	assert.True(t, f2b.IsBanned("cred2", "gpt-4"))

	// Another credential: 500 (default rule) needs 3 attempts to ban
	f2b.RecordResponse("cred3", "gpt-4", 500)
	assert.False(t, f2b.IsBanned("cred3", "gpt-4"))
	f2b.RecordResponse("cred3", "gpt-4", 500)
	assert.False(t, f2b.IsBanned("cred3", "gpt-4"))
	f2b.RecordResponse("cred3", "gpt-4", 500)
	assert.True(t, f2b.IsBanned("cred3", "gpt-4"))
}

// --- New tests for per-model banning ---

func TestIndependentModelBanning(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Ban cred1|model-a
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)

	// cred1|model-a should be banned
	assert.True(t, f2b.IsBanned("cred1", "model-a"))

	// cred1|model-b should NOT be banned
	assert.False(t, f2b.IsBanned("cred1", "model-b"))

	// Failure count for model-b should be 0
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "model-b"))
}

func TestSuccessResetsOnlySpecificPair(t *testing.T) {
	f2b := New(5, 0, []int{401, 403, 500})

	// Record errors for cred1|model-a
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)
	assert.Equal(t, 2, f2b.GetFailureCount("cred1", "model-a"))

	// Record errors for cred1|model-b
	f2b.RecordResponse("cred1", "model-b", 500)
	f2b.RecordResponse("cred1", "model-b", 500)
	assert.Equal(t, 2, f2b.GetFailureCount("cred1", "model-b"))

	// Success for model-a resets only model-a
	f2b.RecordResponse("cred1", "model-a", 200)
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "model-a"))

	// model-b count is unchanged
	assert.Equal(t, 2, f2b.GetFailureCount("cred1", "model-b"))
}

func TestHasAnyBan(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Initially no bans
	assert.False(t, f2b.HasAnyBan("cred1"))

	// Ban cred1|model-a
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)

	// HasAnyBan should be true
	assert.True(t, f2b.HasAnyBan("cred1"))

	// Unban cred1|model-a
	f2b.Unban("cred1", "model-a")

	// HasAnyBan should be false
	assert.False(t, f2b.HasAnyBan("cred1"))
}

func TestUnbanCredential(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Ban cred1|model-a
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)

	// Ban cred1|model-b
	f2b.RecordResponse("cred1", "model-b", 500)
	f2b.RecordResponse("cred1", "model-b", 500)
	f2b.RecordResponse("cred1", "model-b", 500)

	assert.True(t, f2b.IsBanned("cred1", "model-a"))
	assert.True(t, f2b.IsBanned("cred1", "model-b"))

	// UnbanCredential unbans all models for cred1
	f2b.UnbanCredential("cred1")

	assert.False(t, f2b.IsBanned("cred1", "model-a"))
	assert.False(t, f2b.IsBanned("cred1", "model-b"))
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "model-a"))
	assert.Equal(t, 0, f2b.GetFailureCount("cred1", "model-b"))
}

func TestGetBannedPairs_MultiplePairs(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Ban cred1|model-a
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)

	// Ban cred2|model-b
	f2b.RecordResponse("cred2", "model-b", 500)
	f2b.RecordResponse("cred2", "model-b", 500)
	f2b.RecordResponse("cred2", "model-b", 500)

	pairs := f2b.GetBannedPairs()
	assert.Len(t, pairs, 2)

	// Sort for deterministic assertions
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Credential < pairs[j].Credential
	})

	assert.Equal(t, "cred1", pairs[0].Credential)
	assert.Equal(t, "model-a", pairs[0].Model)
	assert.Equal(t, 401, pairs[0].ErrorCode)
	assert.Equal(t, map[int]int{401: 3}, pairs[0].ErrorCodeCounts)
	assert.False(t, pairs[0].BanTime.IsZero())
	assert.Equal(t, time.Duration(0), pairs[0].BanDuration)

	assert.Equal(t, "cred2", pairs[1].Credential)
	assert.Equal(t, "model-b", pairs[1].Model)
	assert.Equal(t, 500, pairs[1].ErrorCode)
	assert.Equal(t, map[int]int{500: 3}, pairs[1].ErrorCodeCounts)
	assert.False(t, pairs[1].BanTime.IsZero())
	assert.Equal(t, time.Duration(0), pairs[1].BanDuration)
}

func TestGetBannedModelsForCredential(t *testing.T) {
	f2b := New(3, 0, []int{401, 403, 500})

	// Ban cred1|model-a
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)
	f2b.RecordResponse("cred1", "model-a", 401)

	// Ban cred1|model-b
	f2b.RecordResponse("cred1", "model-b", 500)
	f2b.RecordResponse("cred1", "model-b", 500)
	f2b.RecordResponse("cred1", "model-b", 500)

	models := f2b.GetBannedModelsForCredential("cred1")
	assert.Len(t, models, 2)

	sort.Strings(models)
	assert.Equal(t, []string{"model-a", "model-b"}, models)

	// A credential with no bans should return empty
	models2 := f2b.GetBannedModelsForCredential("cred2")
	assert.Len(t, models2, 0)
}
