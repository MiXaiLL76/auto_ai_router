package ratelimit

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	rl := New()

	assert.NotNil(t, rl)
	assert.NotNil(t, rl.limiters)
	assert.NotNil(t, rl.modelLimiters)
}

func TestAddCredential(t *testing.T) {
	rl := New()

	rl.AddCredential("cred1", 100)
	rl.AddCredential("cred2", 200)

	// Verify limiters were created
	assert.True(t, rl.Allow("cred1"))
	assert.True(t, rl.Allow("cred2"))
}

func TestAddModel(t *testing.T) {
	rl := New()

	rl.AddModel("cred1", "gpt-4o", 50)
	rl.AddModel("cred1", "gpt-4o-mini", 100)

	// Verify model limiters were created for cred1
	assert.True(t, rl.AllowModel("cred1", "gpt-4o"))
	assert.True(t, rl.AllowModel("cred1", "gpt-4o-mini"))
}

func TestAllow_UnderLimit(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 5)

	// Make requests under limit
	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("cred1"), "Request %d should be allowed", i+1)
	}

	// 6th request should be denied
	assert.False(t, rl.Allow("cred1"))
}

func TestAllow_AtLimit(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 3)

	// Make exactly 3 requests (at limit)
	assert.True(t, rl.Allow("cred1"))
	assert.True(t, rl.Allow("cred1"))
	assert.True(t, rl.Allow("cred1"))

	// 4th request should be denied
	assert.False(t, rl.Allow("cred1"))
}

func TestAllow_OverLimit(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 2)

	// Make 2 requests (at limit)
	assert.True(t, rl.Allow("cred1"))
	assert.True(t, rl.Allow("cred1"))

	// Next requests should be denied
	assert.False(t, rl.Allow("cred1"))
	assert.False(t, rl.Allow("cred1"))
}

func TestAllow_UnlimitedRPM(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", -1) // -1 means unlimited

	// Make many requests - all should be allowed
	for i := 0; i < 1000; i++ {
		assert.True(t, rl.Allow("cred1"), "Request %d should be allowed for unlimited RPM", i+1)
	}
}

func TestAllow_NonExistentCredential(t *testing.T) {
	rl := New()

	// Should return false for non-existent credential
	assert.False(t, rl.Allow("non_existent"))
}

func TestAllowModel_UnderLimit(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 3)

	// Make requests under limit for cred1
	assert.True(t, rl.AllowModel("cred1", "gpt-4o"))
	assert.True(t, rl.AllowModel("cred1", "gpt-4o"))
	assert.True(t, rl.AllowModel("cred1", "gpt-4o"))

	// 4th request should be denied
	assert.False(t, rl.AllowModel("cred1", "gpt-4o"))
}

func TestAllowModel_UnlimitedRPM(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", -1) // Unlimited

	// Make many requests - all should be allowed
	for i := 0; i < 500; i++ {
		assert.True(t, rl.AllowModel("cred1", "gpt-4o"))
	}
}

func TestAllowModel_NonTrackedModel(t *testing.T) {
	rl := New()

	// Model not tracked for cred1 - should allow (default behavior)
	assert.True(t, rl.AllowModel("cred1", "unknown-model"))
}

func TestGetCurrentRPM(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)

	// Initial RPM should be 0
	assert.Equal(t, 0, rl.GetCurrentRPM("cred1"))

	// Make 3 requests
	rl.Allow("cred1")
	rl.Allow("cred1")
	rl.Allow("cred1")

	// Current RPM should be 3
	assert.Equal(t, 3, rl.GetCurrentRPM("cred1"))
}

func TestGetCurrentRPM_NonExistentCredential(t *testing.T) {
	rl := New()

	// Should return 0 for non-existent credential
	assert.Equal(t, 0, rl.GetCurrentRPM("non_existent"))
}

func TestGetCurrentModelRPM(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 100)

	// Initial RPM should be 0
	assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o"))

	// Make 5 requests for cred1:gpt-4o
	for i := 0; i < 5; i++ {
		rl.AllowModel("cred1", "gpt-4o")
	}

	// Current RPM should be 5
	assert.Equal(t, 5, rl.GetCurrentModelRPM("cred1", "gpt-4o"))
}

func TestGetAllModels(t *testing.T) {
	rl := New()

	// Initially empty
	models := rl.GetAllModels()
	assert.Len(t, models, 0)

	// Add models for cred1
	rl.AddModel("cred1", "gpt-4o", 50)
	rl.AddModel("cred1", "gpt-4o-mini", 100)
	rl.AddModel("cred2", "gpt-3.5-turbo", 150)

	models = rl.GetAllModels()
	assert.Len(t, models, 3)
	// Now models are returned as "credential:model" keys
	assert.Contains(t, models, "cred1:gpt-4o")
	assert.Contains(t, models, "cred1:gpt-4o-mini")
	assert.Contains(t, models, "cred2:gpt-3.5-turbo")
}

func TestSlidingWindow_Cleanup(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)

	// Make some requests
	rl.Allow("cred1")
	rl.Allow("cred1")
	rl.Allow("cred1")

	assert.Equal(t, 3, rl.GetCurrentRPM("cred1"))

	// Manually manipulate request times to simulate old requests
	rl.mu.Lock()
	limiter := rl.limiters["cred1"]
	limiter.mu.Lock()
	// Set all requests to 2 minutes ago
	oldTime := time.Now().UTC().Add(-2 * time.Minute)
	for i := range limiter.requests {
		limiter.requests[i] = oldTime
	}
	limiter.mu.Unlock()
	rl.mu.Unlock()

	// Make a new request - should clean up old ones
	rl.Allow("cred1")

	// Current RPM should be 1 (only the new request)
	assert.Equal(t, 1, rl.GetCurrentRPM("cred1"))
}

func TestConcurrency_Credential(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 1000)

	var wg sync.WaitGroup
	numGoroutines := 50
	requestsPerGoroutine := 20

	// Concurrent Allow calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				rl.Allow("cred1")
			}
		}()
	}

	// Concurrent GetCurrentRPM calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				_ = rl.GetCurrentRPM("cred1")
			}
		}()
	}

	wg.Wait()

	// Verify total requests recorded (should be exactly numGoroutines * requestsPerGoroutine)
	totalRequests := numGoroutines * requestsPerGoroutine
	currentRPM := rl.GetCurrentRPM("cred1")
	assert.Equal(t, totalRequests, currentRPM)
}

func TestConcurrency_Model(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 1000)

	var wg sync.WaitGroup
	numGoroutines := 30
	requestsPerGoroutine := 10

	// Concurrent AllowModel calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				rl.AllowModel("cred1", "gpt-4o")
			}
		}()
	}

	// Concurrent GetCurrentModelRPM calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				_ = rl.GetCurrentModelRPM("cred1", "gpt-4o")
			}
		}()
	}

	wg.Wait()

	// Verify total requests
	totalRequests := numGoroutines * requestsPerGoroutine
	currentRPM := rl.GetCurrentModelRPM("cred1", "gpt-4o")
	assert.Equal(t, totalRequests, currentRPM)
}

func TestMultipleCredentials(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 5)
	rl.AddCredential("cred2", 3)
	rl.AddCredential("cred3", 10)

	// Make requests to different credentials
	rl.Allow("cred1")
	rl.Allow("cred1")
	rl.Allow("cred2")
	rl.Allow("cred3")
	rl.Allow("cred3")
	rl.Allow("cred3")

	// Verify independent counters
	assert.Equal(t, 2, rl.GetCurrentRPM("cred1"))
	assert.Equal(t, 1, rl.GetCurrentRPM("cred2"))
	assert.Equal(t, 3, rl.GetCurrentRPM("cred3"))

	// Each credential should enforce its own limit
	for i := 0; i < 3; i++ {
		rl.Allow("cred1") // Total 5
	}
	assert.False(t, rl.Allow("cred1")) // Should be denied (over limit)
	assert.True(t, rl.Allow("cred2"))  // Should still work (under limit)
}

func TestMultipleModels(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 2)
	rl.AddModel("cred1", "gpt-4o-mini", 5)

	// Make requests to different models for cred1
	rl.AllowModel("cred1", "gpt-4o")
	rl.AllowModel("cred1", "gpt-4o")
	rl.AllowModel("cred1", "gpt-4o-mini")

	// Verify independent counters per (credential, model)
	assert.Equal(t, 2, rl.GetCurrentModelRPM("cred1", "gpt-4o"))
	assert.Equal(t, 1, rl.GetCurrentModelRPM("cred1", "gpt-4o-mini"))

	// cred1:gpt-4o should be at limit
	assert.False(t, rl.AllowModel("cred1", "gpt-4o"))
	// cred1:gpt-4o-mini should still allow
	assert.True(t, rl.AllowModel("cred1", "gpt-4o-mini"))
}

func TestAllow_RapidRequests(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 10)

	allowed := 0
	denied := 0

	// Make 20 rapid requests (twice the limit)
	for i := 0; i < 20; i++ {
		if rl.Allow("cred1") {
			allowed++
		} else {
			denied++
		}
	}

	// Exactly 10 should be allowed, 10 denied
	assert.Equal(t, 10, allowed)
	assert.Equal(t, 10, denied)
	assert.Equal(t, 10, rl.GetCurrentRPM("cred1"))
}

// TPM (Tokens Per Minute) Tests

func TestAddCredentialWithTPM(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	assert.True(t, rl.AllowTokens("cred1"))
}

func TestAddModelWithTPM(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)

	assert.True(t, rl.AllowModelTokens("cred1", "gpt-4o"))
}

func TestConsumeTokens(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Initially no tokens consumed
	assert.Equal(t, 0, rl.GetCurrentTPM("cred1"))

	// Consume some tokens
	rl.ConsumeTokens("cred1", 500)
	assert.Equal(t, 500, rl.GetCurrentTPM("cred1"))

	// Consume more tokens
	rl.ConsumeTokens("cred1", 300)
	assert.Equal(t, 800, rl.GetCurrentTPM("cred1"))
}

func TestConsumeTokens_NonExistentCredential(t *testing.T) {
	rl := New()

	// Should not panic for non-existent credential
	rl.ConsumeTokens("non_existent", 100)

	// Should return 0 TPM
	assert.Equal(t, 0, rl.GetCurrentTPM("non_existent"))
}

func TestAllowTokens_UnderLimit(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 1000)

	// Consume tokens under limit
	rl.ConsumeTokens("cred1", 500)
	assert.True(t, rl.AllowTokens("cred1"))

	rl.ConsumeTokens("cred1", 400)
	assert.True(t, rl.AllowTokens("cred1"))
}

func TestAllowTokens_AtLimit(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 1000)

	// Consume exactly at limit
	rl.ConsumeTokens("cred1", 1000)
	assert.Equal(t, 1000, rl.GetCurrentTPM("cred1"))

	// Should not allow more tokens
	assert.False(t, rl.AllowTokens("cred1"))
}

func TestAllowTokens_OverLimit(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 1000)

	// Consume over limit
	rl.ConsumeTokens("cred1", 1500)
	assert.Equal(t, 1500, rl.GetCurrentTPM("cred1"))

	// Should not allow more tokens
	assert.False(t, rl.AllowTokens("cred1"))
}

func TestAllowTokens_UnlimitedTPM(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, -1) // -1 means unlimited TPM

	// Consume many tokens
	for i := 0; i < 100; i++ {
		rl.ConsumeTokens("cred1", 1000)
	}

	// Should still allow tokens (unlimited)
	assert.True(t, rl.AllowTokens("cred1"))
}

func TestAllowTokens_NonExistentCredential(t *testing.T) {
	rl := New()

	// Should return false for non-existent credential
	assert.False(t, rl.AllowTokens("non_existent"))
}

func TestGetCurrentTPM(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Initially 0
	assert.Equal(t, 0, rl.GetCurrentTPM("cred1"))

	// Consume tokens
	rl.ConsumeTokens("cred1", 1000)
	assert.Equal(t, 1000, rl.GetCurrentTPM("cred1"))

	rl.ConsumeTokens("cred1", 2500)
	assert.Equal(t, 3500, rl.GetCurrentTPM("cred1"))
}

func TestGetCurrentTPM_NonExistentCredential(t *testing.T) {
	rl := New()

	// Should return 0 for non-existent credential
	assert.Equal(t, 0, rl.GetCurrentTPM("non_existent"))
}

func TestGetCurrentTPM_Cleanup(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Consume tokens
	rl.ConsumeTokens("cred1", 1000)
	assert.Equal(t, 1000, rl.GetCurrentTPM("cred1"))

	// Manually set old timestamps
	rl.mu.Lock()
	limiter := rl.limiters["cred1"]
	limiter.mu.Lock()
	oldTime := time.Now().UTC().Add(-2 * time.Minute)
	for i := range limiter.tokens {
		limiter.tokens[i].timestamp = oldTime
	}
	limiter.mu.Unlock()
	rl.mu.Unlock()

	// Current TPM should be 0 (old tokens cleaned up)
	assert.Equal(t, 0, rl.GetCurrentTPM("cred1"))
}

func TestConsumeModelTokens(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)

	// Initially no tokens consumed
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	// Consume tokens
	rl.ConsumeModelTokens("cred1", "gpt-4o", 1000)
	assert.Equal(t, 1000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	// Consume more
	rl.ConsumeModelTokens("cred1", "gpt-4o", 1500)
	assert.Equal(t, 2500, rl.GetCurrentModelTPM("cred1", "gpt-4o"))
}

func TestConsumeModelTokens_NonExistentModel(t *testing.T) {
	rl := New()

	// Should not panic for non-existent model
	rl.ConsumeModelTokens("cred1", "non-existent-model", 1000)

	// Should return 0 TPM
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "non-existent-model"))
}

func TestAllowModelTokens_UnderLimit(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)

	// Consume tokens under limit
	rl.ConsumeModelTokens("cred1", "gpt-4o", 2000)
	assert.True(t, rl.AllowModelTokens("cred1", "gpt-4o"))

	rl.ConsumeModelTokens("cred1", "gpt-4o", 2000)
	assert.True(t, rl.AllowModelTokens("cred1", "gpt-4o"))
}

func TestAllowModelTokens_AtLimit(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)

	// Consume exactly at limit
	rl.ConsumeModelTokens("cred1", "gpt-4o", 5000)
	assert.Equal(t, 5000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	// Should not allow more tokens
	assert.False(t, rl.AllowModelTokens("cred1", "gpt-4o"))
}

func TestAllowModelTokens_OverLimit(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)

	// Consume over limit
	rl.ConsumeModelTokens("cred1", "gpt-4o", 6000)
	assert.Equal(t, 6000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	// Should not allow more tokens
	assert.False(t, rl.AllowModelTokens("cred1", "gpt-4o"))
}

func TestAllowModelTokens_UnlimitedTPM(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, -1) // -1 means unlimited TPM

	// Consume many tokens
	for i := 0; i < 100; i++ {
		rl.ConsumeModelTokens("cred1", "gpt-4o", 10000)
	}

	// Should still allow tokens (unlimited)
	assert.True(t, rl.AllowModelTokens("cred1", "gpt-4o"))
}

func TestAllowModelTokens_NonTrackedModel(t *testing.T) {
	rl := New()

	// Model not tracked - should allow (default behavior)
	assert.True(t, rl.AllowModelTokens("cred1", "unknown-model"))
}

func TestGetCurrentModelTPM(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 10000)

	// Initially 0
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	// Consume tokens
	rl.ConsumeModelTokens("cred1", "gpt-4o", 3000)
	assert.Equal(t, 3000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	rl.ConsumeModelTokens("cred1", "gpt-4o", 2000)
	assert.Equal(t, 5000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))
}

func TestGetCurrentModelTPM_NonExistentModel(t *testing.T) {
	rl := New()

	// Should return 0 for non-existent model
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "non-existent-model"))
}

func TestGetModelLimitRPM(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)
	rl.AddModelWithTPM("cred1", "gpt-4o-mini", 100, -1)

	// Test existing models
	assert.Equal(t, 50, rl.GetModelLimitRPM("cred1", "gpt-4o"))
	assert.Equal(t, 100, rl.GetModelLimitRPM("cred1", "gpt-4o-mini"))

	// Test non-existent model (should return -1)
	assert.Equal(t, -1, rl.GetModelLimitRPM("cred1", "non-existent-model"))
}

func TestGetModelLimitTPM(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)
	rl.AddModelWithTPM("cred1", "gpt-4o-mini", 100, 10000)
	rl.AddModelWithTPM("cred2", "claude-3", 75, -1) // Unlimited TPM

	// Test existing models
	assert.Equal(t, 5000, rl.GetModelLimitTPM("cred1", "gpt-4o"))
	assert.Equal(t, 10000, rl.GetModelLimitTPM("cred1", "gpt-4o-mini"))
	assert.Equal(t, -1, rl.GetModelLimitTPM("cred2", "claude-3"))

	// Test non-existent model (should return -1)
	assert.Equal(t, -1, rl.GetModelLimitTPM("cred1", "non-existent-model"))
}

func TestGetCurrentModelRPM_EmptyLimiter(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 100)

	// Should return 0 when no requests have been made
	rpm := rl.GetCurrentModelRPM("cred1", "gpt-4o")
	assert.Equal(t, 0, rpm)

	// Make one request
	rl.AllowModel("cred1", "gpt-4o")
	assert.Equal(t, 1, rl.GetCurrentModelRPM("cred1", "gpt-4o"))
}

func TestMultipleModelsTokens(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)
	rl.AddModelWithTPM("cred1", "gpt-4o-mini", 100, 10000)

	// Consume tokens for different models
	rl.ConsumeModelTokens("cred1", "gpt-4o", 3000)
	rl.ConsumeModelTokens("cred1", "gpt-4o-mini", 6000)

	// Verify independent counters
	assert.Equal(t, 3000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))
	assert.Equal(t, 6000, rl.GetCurrentModelTPM("cred1", "gpt-4o-mini"))

	// gpt-4o should still allow tokens
	assert.True(t, rl.AllowModelTokens("cred1", "gpt-4o"))

	// Consume more tokens to reach limit for gpt-4o
	rl.ConsumeModelTokens("cred1", "gpt-4o", 2000)
	assert.False(t, rl.AllowModelTokens("cred1", "gpt-4o"))

	// gpt-4o-mini should still allow
	assert.True(t, rl.AllowModelTokens("cred1", "gpt-4o-mini"))
}
func TestCanAllow(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 2)

	// Should allow first request without recording
	assert.True(t, rl.CanAllow("cred1"))
	assert.Equal(t, 0, rl.GetCurrentRPM("cred1")) // No requests recorded

	// Record one request
	assert.True(t, rl.Allow("cred1"))
	assert.Equal(t, 1, rl.GetCurrentRPM("cred1"))

	// Should still allow second request without recording
	assert.True(t, rl.CanAllow("cred1"))
	assert.Equal(t, 1, rl.GetCurrentRPM("cred1")) // Still only 1 recorded

	// Record second request
	assert.True(t, rl.Allow("cred1"))
	assert.Equal(t, 2, rl.GetCurrentRPM("cred1"))

	// Should not allow third request (at limit)
	assert.False(t, rl.CanAllow("cred1"))
	assert.Equal(t, 2, rl.GetCurrentRPM("cred1")) // Still only 2 recorded
}

func TestCanAllowModel(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 2)

	// Should allow first request without recording
	assert.True(t, rl.CanAllowModel("cred1", "gpt-4o"))
	assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o")) // No requests recorded

	// Record one request
	assert.True(t, rl.AllowModel("cred1", "gpt-4o"))
	assert.Equal(t, 1, rl.GetCurrentModelRPM("cred1", "gpt-4o"))

	// Should still allow second request without recording
	assert.True(t, rl.CanAllowModel("cred1", "gpt-4o"))
	assert.Equal(t, 1, rl.GetCurrentModelRPM("cred1", "gpt-4o")) // Still only 1 recorded

	// Record second request
	assert.True(t, rl.AllowModel("cred1", "gpt-4o"))
	assert.Equal(t, 2, rl.GetCurrentModelRPM("cred1", "gpt-4o"))

	// Should not allow third request (at limit)
	assert.False(t, rl.CanAllowModel("cred1", "gpt-4o"))
	assert.Equal(t, 2, rl.GetCurrentModelRPM("cred1", "gpt-4o")) // Still only 2 recorded
}

func TestCanAllow_NonExistentCredential(t *testing.T) {
	rl := New()

	// Should return false for non-existent credential
	assert.False(t, rl.CanAllow("non_existent"))
}

func TestCanAllowModel_NonTrackedModel(t *testing.T) {
	rl := New()

	// Model not tracked - should allow (default behavior)
	assert.True(t, rl.CanAllowModel("cred1", "unknown-model"))
}

func TestRPMLimit_Zero(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 0)

	// With RPM limit 0, no requests should be allowed
	assert.False(t, rl.Allow("cred1"))
	assert.False(t, rl.CanAllow("cred1"))
	assert.Equal(t, 0, rl.GetCurrentRPM("cred1"))
}

func TestModelRPMLimit_Zero(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4o", 0)

	// With RPM limit 0, no requests should be allowed
	assert.False(t, rl.AllowModel("cred1", "gpt-4o"))
	assert.False(t, rl.CanAllowModel("cred1", "gpt-4o"))
	assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o"))
}

func TestTPMLimit_Zero(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 0)

	// With TPM limit 0, no tokens should be allowed
	assert.False(t, rl.AllowTokens("cred1"))
	assert.Equal(t, 0, rl.GetCurrentTPM("cred1"))
}

func TestModelTPMLimit_Zero(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 0)

	// With TPM limit 0, no tokens should be allowed
	assert.False(t, rl.AllowModelTokens("cred1", "gpt-4o"))
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "gpt-4o"))
}

func TestConsumeTokens_TokenCleanup(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Consume tokens
	rl.ConsumeTokens("cred1", 5000)
	assert.Equal(t, 5000, rl.GetCurrentTPM("cred1"))

	// Manually set old timestamps
	rl.mu.Lock()
	limiter := rl.limiters["cred1"]
	limiter.mu.Lock()
	oldTime := time.Now().UTC().Add(-2 * time.Minute)
	for i := range limiter.tokens {
		limiter.tokens[i].timestamp = oldTime
	}
	limiter.mu.Unlock()
	rl.mu.Unlock()

	// After cleanup, should be 0
	assert.Equal(t, 0, rl.GetCurrentTPM("cred1"))

	// New tokens should work
	rl.ConsumeTokens("cred1", 3000)
	assert.Equal(t, 3000, rl.GetCurrentTPM("cred1"))
}

func TestConsumeModelTokens_TokenCleanup(t *testing.T) {
	rl := New()
	rl.AddModelWithTPM("cred1", "gpt-4o", 50, 10000)

	// Consume tokens
	rl.ConsumeModelTokens("cred1", "gpt-4o", 5000)
	assert.Equal(t, 5000, rl.GetCurrentModelTPM("cred1", "gpt-4o"))

	// Manually set old timestamps
	rl.mu.Lock()
	key := "cred1:gpt-4o"
	limiter := rl.modelLimiters[key]
	limiter.mu.Lock()
	oldTime := time.Now().UTC().Add(-2 * time.Minute)
	for i := range limiter.tokens {
		limiter.tokens[i].timestamp = oldTime
	}
	limiter.mu.Unlock()
	rl.mu.Unlock()

	// After cleanup, should be 0
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "gpt-4o"))
}

func TestMultipleTokenConsumptions_Accumulate(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Multiple consumption should accumulate
	rl.ConsumeTokens("cred1", 1000)
	rl.ConsumeTokens("cred1", 2000)
	rl.ConsumeTokens("cred1", 3000)

	assert.Equal(t, 6000, rl.GetCurrentTPM("cred1"))
}

func TestAllowRPM_ExactlyAtLimit(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 5)

	// Fill up to exactly the limit
	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("cred1"))
	}

	// 6th should fail
	assert.False(t, rl.Allow("cred1"))

	// CanAllow should also return false
	assert.False(t, rl.CanAllow("cred1"))
}

func TestAllowTokens_ExactlyAtLimit(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 1000)

	// Consume exactly at limit
	rl.ConsumeTokens("cred1", 1000)
	assert.False(t, rl.AllowTokens("cred1"))

	// Consume one more
	rl.ConsumeTokens("cred1", 1)
	assert.False(t, rl.AllowTokens("cred1"))
}

func TestSetCredentialCurrentUsage_NoLimiter(t *testing.T) {
	rl := New()

	// Set usage for non-existent credential - should not panic
	rl.SetCredentialCurrentUsage("non-existent", 50, 5000)
}

func TestSetCredentialCurrentUsage_WithRPM(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)

	// Set current usage to 50 RPM
	rl.SetCredentialCurrentUsage("cred1", 50, 0)

	// GetCurrentRPM should return updated value (after cleanup of old requests)
	currentRPM := rl.GetCurrentRPM("cred1")
	assert.Greater(t, currentRPM, 0, "Should have some RPM after setting usage")
}

func TestSetCredentialCurrentUsage_WithTPM(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Set current usage to 5000 TPM
	rl.SetCredentialCurrentUsage("cred1", 0, 5000)

	// GetCurrentTPM should return updated value (after cleanup of old tokens)
	currentTPM := rl.GetCurrentTPM("cred1")
	assert.Greater(t, currentTPM, 0, "Should have some TPM after setting usage")
}

func TestSetCredentialCurrentUsage_WithBoth(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// Set current usage to 50 RPM and 5000 TPM
	rl.SetCredentialCurrentUsage("cred1", 50, 5000)

	// Both should be updated
	currentRPM := rl.GetCurrentRPM("cred1")
	currentTPM := rl.GetCurrentTPM("cred1")
	assert.Greater(t, currentRPM, 0, "Should have some RPM after setting usage")
	assert.Greater(t, currentTPM, 0, "Should have some TPM after setting usage")
}

func TestSetCredentialCurrentUsage_ZeroValues(t *testing.T) {
	rl := New()
	rl.AddCredentialWithTPM("cred1", 100, 10000)

	// First set some usage
	rl.SetCredentialCurrentUsage("cred1", 50, 5000)
	assert.Greater(t, rl.GetCurrentRPM("cred1"), 0)

	// Then reset to zero
	rl.SetCredentialCurrentUsage("cred1", 0, 0)
	assert.Equal(t, 0, rl.GetCurrentRPM("cred1"))
	assert.Equal(t, 0, rl.GetCurrentTPM("cred1"))
}

func TestSetModelCurrentUsage_NoLimiter(t *testing.T) {
	rl := New()

	// Set usage for non-existent model - should not panic
	rl.SetModelCurrentUsage("non-existent", "gpt-4", 50, 5000)
}

func TestSetModelCurrentUsage_WithRPM(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)
	rl.AddModel("cred1", "gpt-4", 50)

	// Set current usage to 25 RPM
	rl.SetModelCurrentUsage("cred1", "gpt-4", 25, 0)

	// GetCurrentModelRPM should return updated value
	currentRPM := rl.GetCurrentModelRPM("cred1", "gpt-4")
	assert.Greater(t, currentRPM, 0, "Should have some model RPM after setting usage")
}

func TestSetModelCurrentUsage_WithTPM(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)
	rl.AddModelWithTPM("cred1", "gpt-4", 50, 5000)

	// Set current usage to 2500 TPM
	rl.SetModelCurrentUsage("cred1", "gpt-4", 0, 2500)

	// GetCurrentModelTPM should return updated value
	currentTPM := rl.GetCurrentModelTPM("cred1", "gpt-4")
	assert.Greater(t, currentTPM, 0, "Should have some model TPM after setting usage")
}

func TestSetModelCurrentUsage_WithBoth(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)
	rl.AddModelWithTPM("cred1", "gpt-4", 50, 5000)

	// Set current usage to 25 RPM and 2500 TPM
	rl.SetModelCurrentUsage("cred1", "gpt-4", 25, 2500)

	// Both should be updated
	currentRPM := rl.GetCurrentModelRPM("cred1", "gpt-4")
	currentTPM := rl.GetCurrentModelTPM("cred1", "gpt-4")
	assert.Greater(t, currentRPM, 0, "Should have some model RPM after setting usage")
	assert.Greater(t, currentTPM, 0, "Should have some model TPM after setting usage")
}

func TestSetModelCurrentUsage_ZeroValues(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)
	rl.AddModelWithTPM("cred1", "gpt-4", 50, 5000)

	// First set some usage
	rl.SetModelCurrentUsage("cred1", "gpt-4", 25, 2500)
	assert.Greater(t, rl.GetCurrentModelRPM("cred1", "gpt-4"), 0)

	// Then reset to zero
	rl.SetModelCurrentUsage("cred1", "gpt-4", 0, 0)
	assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4"))
	assert.Equal(t, 0, rl.GetCurrentModelTPM("cred1", "gpt-4"))
}

func TestSetCredentialCurrentUsage_MultipleModels(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 100)
	rl.AddModel("cred1", "gpt-4", 50)
	rl.AddModel("cred1", "gpt-3.5", 60)

	// Set credential usage
	rl.SetCredentialCurrentUsage("cred1", 40, 4000)

	// Credential level should be updated
	assert.Greater(t, rl.GetCurrentRPM("cred1"), 0)

	// Model levels should not be affected
	assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4"))
	assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-3.5"))
}

// TestConcurrentSetCredentialUsageAndAllow tests concurrent SetCredentialCurrentUsage and Allow
func TestConcurrentSetCredentialUsageAndAllow(t *testing.T) {
	rl := New()
	rl.AddCredential("cred1", 1000)

	done := make(chan bool, 20)

	// 10 goroutines setting credential usage (from remote proxy updates)
	for i := 0; i < 10; i++ {
		go func() {
			rl.SetCredentialCurrentUsage("cred1", 50, 5000)
			done <- true
		}()
	}

	// 10 goroutines making requests
	for i := 0; i < 10; i++ {
		go func() {
			_ = rl.Allow("cred1")
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}

	// Should not panic and system should be consistent
	_ = rl.GetCurrentRPM("cred1")
}

// TestConcurrentSetModelUsageAndAllow tests concurrent SetModelCurrentUsage and AllowModel
func TestConcurrentSetModelUsageAndAllow(t *testing.T) {
	rl := New()
	rl.AddModel("cred1", "gpt-4", 500)

	done := make(chan bool, 20)

	// 10 goroutines setting model usage
	for i := 0; i < 10; i++ {
		go func() {
			rl.SetModelCurrentUsage("cred1", "gpt-4", 100, 10000)
			done <- true
		}()
	}

	// 10 goroutines making requests
	for i := 0; i < 10; i++ {
		go func() {
			_ = rl.AllowModel("cred1", "gpt-4")
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}

	// Should not panic and system should be consistent
	_ = rl.GetCurrentModelRPM("cred1", "gpt-4")
}

// TestConcurrentMultipleCredentials tests concurrent operations on multiple credentials
func TestConcurrentMultipleCredentials(t *testing.T) {
	rl := New()

	// Add multiple credentials
	for i := 0; i < 5; i++ {
		credName := "cred" + string(rune(i+'0'))
		rl.AddCredential(credName, 100)
	}

	done := make(chan bool, 50)

	// Concurrent operations on different credentials
	for i := 0; i < 10; i++ {
		for j := 0; j < 5; j++ {
			credName := "cred" + string(rune(j+'0'))

			// Allow request
			go func(cred string) {
				_ = rl.Allow(cred)
				done <- true
			}(credName)

			// Get current RPM
			go func(cred string) {
				_ = rl.GetCurrentRPM(cred)
				done <- true
			}(credName)
		}
	}

	// Wait for all goroutines
	for i := 0; i < 100; i++ {
		<-done
	}
}

// TestConcurrentMultipleModels tests concurrent operations on multiple models for same credential
func TestConcurrentMultipleModels(t *testing.T) {
	rl := New()

	// Add multiple models for same credential
	for i := 0; i < 5; i++ {
		modelName := "model-" + string(rune(i+'0'))
		rl.AddModel("cred1", modelName, 100)
	}

	done := make(chan bool, 50)

	// Concurrent operations on different models
	for i := 0; i < 10; i++ {
		for j := 0; j < 5; j++ {
			modelName := "model-" + string(rune(j+'0'))

			// Allow model request
			go func(model string) {
				_ = rl.AllowModel("cred1", model)
				done <- true
			}(modelName)

			// Set model usage
			go func(model string) {
				rl.SetModelCurrentUsage("cred1", model, 50, 5000)
				done <- true
			}(modelName)
		}
	}

	// Wait for all goroutines
	for i := 0; i < 100; i++ {
		<-done
	}
}

func TestTryAllowAll(t *testing.T) {
	tests := []struct {
		name           string
		setup          func() *RPMLimiter
		credentialName string
		modelName      string
		wantAllowed    bool
		// postCheck runs after TryAllowAll to verify side effects
		postCheck func(t *testing.T, rl *RPMLimiter)
	}{
		{
			name: "all limits pass — returns true, usage recorded",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 10, 10000)
				rl.AddModelWithTPM("cred1", "gpt-4o", 5, 5000)
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    true,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				assert.Equal(t, 1, rl.GetCurrentRPM("cred1"), "credential RPM should be recorded")
				assert.Equal(t, 1, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should be recorded")
			},
		},
		{
			name: "credential RPM exhausted — returns false",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 2, -1)
				rl.AddModelWithTPM("cred1", "gpt-4o", 10, -1)
				// Exhaust credential RPM
				rl.Allow("cred1")
				rl.Allow("cred1")
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    false,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				assert.Equal(t, 2, rl.GetCurrentRPM("cred1"), "credential RPM should not increase")
				assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should not be recorded")
			},
		},
		{
			name: "credential TPM exhausted — returns false",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 100, 1000)
				rl.AddModelWithTPM("cred1", "gpt-4o", 50, 5000)
				// Exhaust credential TPM
				rl.ConsumeTokens("cred1", 1000)
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    false,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				assert.Equal(t, 0, rl.GetCurrentRPM("cred1"), "credential RPM should not be recorded")
				assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should not be recorded")
			},
		},
		{
			name: "model RPM exhausted — returns false",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 100, -1)
				rl.AddModelWithTPM("cred1", "gpt-4o", 2, -1)
				// Exhaust model RPM
				rl.AllowModel("cred1", "gpt-4o")
				rl.AllowModel("cred1", "gpt-4o")
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    false,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				assert.Equal(t, 0, rl.GetCurrentRPM("cred1"), "credential RPM should not be recorded on model RPM failure")
				assert.Equal(t, 2, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should not increase")
			},
		},
		{
			name: "model TPM exhausted — returns false",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 100, -1)
				rl.AddModelWithTPM("cred1", "gpt-4o", 50, 1000)
				// Exhaust model TPM
				rl.ConsumeModelTokens("cred1", "gpt-4o", 1000)
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    false,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				assert.Equal(t, 0, rl.GetCurrentRPM("cred1"), "credential RPM should not be recorded on model TPM failure")
				assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should not be recorded")
			},
		},
		{
			name: "credential RPM passes but model RPM fails — credential RPM NOT recorded (rollback)",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 10, -1)
				rl.AddModelWithTPM("cred1", "gpt-4o", 3, -1)
				// Exhaust model RPM only
				rl.AllowModel("cred1", "gpt-4o")
				rl.AllowModel("cred1", "gpt-4o")
				rl.AllowModel("cred1", "gpt-4o")
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    false,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				// Credential RPM should be 0 — the attempt should NOT have recorded it
				assert.Equal(t, 0, rl.GetCurrentRPM("cred1"), "credential RPM must NOT be recorded when model RPM fails")
				assert.Equal(t, 3, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should remain at limit")
			},
		},
		{
			name: "unlimited RPM (-1) — always passes",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", -1, -1)
				rl.AddModelWithTPM("cred1", "gpt-4o", -1, -1)
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    true,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				// Even with unlimited, usage should be recorded
				assert.Equal(t, 1, rl.GetCurrentRPM("cred1"), "credential RPM should be recorded")
				assert.Equal(t, 1, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should be recorded")
			},
		},
		{
			name: "no model-specific limits — only credential limits checked",
			setup: func() *RPMLimiter {
				rl := New()
				rl.AddCredentialWithTPM("cred1", 10, 10000)
				// No model limiter added
				return rl
			},
			credentialName: "cred1",
			modelName:      "gpt-4o",
			wantAllowed:    true,
			postCheck: func(t *testing.T, rl *RPMLimiter) {
				assert.Equal(t, 1, rl.GetCurrentRPM("cred1"), "credential RPM should be recorded")
				assert.Equal(t, 0, rl.GetCurrentModelRPM("cred1", "gpt-4o"), "model RPM should be 0 (no model limiter)")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rl := tt.setup()
			got := rl.TryAllowAll(tt.credentialName, tt.modelName)
			assert.Equal(t, tt.wantAllowed, got)
			if tt.postCheck != nil {
				tt.postCheck(t, rl)
			}
		})
	}
}
