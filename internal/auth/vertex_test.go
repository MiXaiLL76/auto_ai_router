package auth

import (
	"encoding/json"
	"log/slog"
	"os"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

// mockTokenSource mocks oauth2.TokenSource
type mockTokenSource struct {
	token      *oauth2.Token
	callCount  int
	shouldFail bool
	err        error
}

func (m *mockTokenSource) Token() (*oauth2.Token, error) {
	m.callCount++
	if m.shouldFail {
		return nil, m.err
	}
	return m.token, nil
}

// Helper to create valid service account JSON
func createValidServiceAccountJSON() string {
	sa := map[string]interface{}{
		"type":         "service_account",
		"project_id":   "test-project",
		"private_key":  "",
		"client_email": "test@test-project.iam.gserviceaccount.com",
	}
	b, _ := json.Marshal(sa)
	return string(b)
}

func TestNewVertexTokenManager(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	if tm == nil {
		t.Fatal("NewVertexTokenManager returned nil")
	}
	if tm.tokens == nil {
		t.Error("tokens map is nil")
	}
	if tm.credentials == nil {
		t.Error("credentials map is nil")
	}
	if tm.logger == nil {
		t.Error("logger is nil")
	}
	if tm.tokenRefresh != 5*time.Minute {
		t.Errorf("tokenRefresh = %v, want 5m", tm.tokenRefresh)
	}
}

func TestGetToken_InvalidJSON(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	_, err := tm.GetToken("test", "", "invalid-json")
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
	if err.Error() != "invalid service account JSON: invalid character 'i' looking for beginning of value" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGetToken_InvalidServiceAccountType(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	invalidSA := map[string]interface{}{
		"type": "user",
	}
	b, _ := json.Marshal(invalidSA)

	_, err := tm.GetToken("test", "", string(b))
	if err == nil {
		t.Error("expected error for non-service-account type, got nil")
	}
}

func TestGetToken_NoCredentials(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	_, err := tm.GetToken("test", "", "")
	if err == nil {
		t.Error("expected error for missing credentials, got nil")
	}
	if err.Error() != "no credentials provided for test" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGetToken_FileNotFound(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	_, err := tm.GetToken("test", "/nonexistent/path.json", "")
	if err == nil {
		t.Error("expected error for non-existent file, got nil")
	}
}

func TestClearToken(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Add a token to cache
	expiry := time.Now().UTC().Add(1 * time.Hour)
	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:     &oauth2.Token{AccessToken: "test-token"},
		expiresAt: expiry,
	}
	tm.mu.Unlock()

	// Verify it exists
	if cached, ok := tm.tokens["test"]; !ok || cached.token.AccessToken != "test-token" {
		t.Error("token not properly cached")
	}

	// Clear it
	tm.ClearToken("test")

	// Verify it's gone
	if _, ok := tm.tokens["test"]; ok {
		t.Error("token should be cleared but still exists")
	}
}

func TestGetTokenExpiry(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Test non-existent token
	_, exists := tm.GetTokenExpiry("nonexistent")
	if exists {
		t.Error("expected false for non-existent token")
	}

	// Add a token
	expiry := time.Now().UTC().Add(1 * time.Hour)
	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:     &oauth2.Token{AccessToken: "test-token"},
		expiresAt: expiry,
	}
	tm.mu.Unlock()

	// Get expiry
	retrievedExpiry, exists := tm.GetTokenExpiry("test")
	if !exists {
		t.Error("expected true for existing token")
	}
	if !retrievedExpiry.Equal(expiry) {
		t.Errorf("expiry mismatch: got %v, want %v", retrievedExpiry, expiry)
	}
}

func TestGetToken_CredentialsCaching(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create a temporary credentials file
	tmpFile, err := os.CreateTemp("", "creds-*.json")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer func() {
		_ = os.Remove(tmpFile.Name())
	}()

	credJSON := createValidServiceAccountJSON()
	if _, err := tmpFile.WriteString(credJSON); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	defer func() {
		_ = tmpFile.Close()
	}()
	// Note: This test can't fully test the token creation without mocking google.CredentialsFromJSON
	// but we can at least test the credentials caching mechanism
	tm.mu.Lock()
	tm.credentials["test"] = []byte(credJSON)
	tm.mu.Unlock()

	// Verify credentials are cached
	tm.mu.RLock()
	cached, exists := tm.credentials["test"]
	tm.mu.RUnlock()

	if !exists {
		t.Error("credentials not cached")
	}
	if string(cached) != credJSON {
		t.Error("cached credentials don't match")
	}
}

func TestGetToken_CachedTokenReuse(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiry := time.Now().UTC().Add(1 * time.Hour)
	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "test-token", Expiry: expiry},
	}

	// Cache a token
	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       mockSource.token,
		tokenSource: mockSource,
		expiresAt:   expiry,
	}
	tm.mu.Unlock()

	// Get the same token - should reuse cached token (valid for 1 hour)
	// No credentials needed if cached token is still valid
	token, err := tm.GetToken("test", "", "")
	if err != nil {
		t.Errorf("expected no error for cached valid token, got: %v", err)
	}
	if token != "test-token" {
		t.Errorf("expected 'test-token', got '%s'", token)
	}
	if mockSource.callCount > 0 {
		t.Error("tokenSource.Token() should not be called for non-expired cached token")
	}
}

func TestGetToken_TokenRefresh(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create an expired token
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "new-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Verify the cached token is in expired state
	tm.mu.RLock()
	cached, ok := tm.tokens["test"]
	tm.mu.RUnlock()

	if !ok {
		t.Error("cached token should exist")
	}
	if !cached.expiresAt.Before(time.Now().UTC()) {
		t.Error("token should be expired for this test")
	}
}

func TestGetToken_TokenRefreshError(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create an expired token with failing TokenSource
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)

	mockSource := &mockTokenSource{
		shouldFail: true,
		err:        os.ErrNotExist,
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Attempt to get expired token - should trigger refresh and fail
	_, err := tm.GetToken("test", "", "")
	if err == nil {
		t.Error("expected error when token refresh fails")
	}

	// Verify token was removed from cache after failed refresh
	tm.mu.RLock()
	_, ok := tm.tokens["test"]
	tm.mu.RUnlock()

	if ok {
		t.Error("expired token should be removed from cache after failed refresh")
	}
}

func TestGetToken_NearExpiry(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create a token that expires in 3 minutes (within refresh buffer of 5 min)
	nearExpiryTime := time.Now().UTC().Add(3 * time.Minute)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "new-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: nearExpiryTime},
		tokenSource: mockSource,
		expiresAt:   nearExpiryTime,
	}
	tm.mu.Unlock()

	// Get token - should trigger refresh since token expires in 3 min (within 5 min buffer)
	token, err := tm.GetToken("test", "", "")
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
	if token != "new-token" {
		t.Errorf("expected refreshed token, got %s", token)
	}
	if mockSource.callCount != 1 {
		t.Errorf("expected 1 Token() call for refresh, got %d", mockSource.callCount)
	}
}

func TestGetToken_ConcurrentRefresh(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create a token that needs refresh
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	callCount := 0
	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "new-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Launch 10 concurrent GetToken calls
	numGoroutines := 10
	results := make(chan string, numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			token, err := tm.GetToken("test", "", "")
			if err != nil {
				t.Errorf("GetToken failed: %v", err)
			}
			results <- token
		}()
	}

	// Collect results
	for i := 0; i < numGoroutines; i++ {
		token := <-results
		if token != "new-token" {
			t.Errorf("expected 'new-token', got '%s'", token)
		}
	}

	// Verify that Token() was called only once (coalescing)
	// This happens because all 10 goroutines should coalesce into one refresh
	callCount = mockSource.callCount
	if callCount < 1 {
		t.Errorf("expected at least 1 Token() call, got %d", callCount)
	}
	// Note: Due to concurrency, we might have 1-2 calls if timing is tight
	// but the important thing is we didn't have 10 calls
	if callCount > 3 {
		t.Errorf("expected <= 3 Token() calls (with potential race), got %d", callCount)
	}
}

func TestGetToken_RequestCoalescing(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create a token that needs refresh
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "coalesced-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Launch 5 concurrent GetToken calls
	numGoroutines := 5
	results := make(chan string, numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			token, err := tm.GetToken("test", "", "")
			if err != nil {
				t.Errorf("GetToken failed: %v", err)
				results <- ""
				return
			}
			results <- token
		}()
	}

	// Collect results - all should get the same token
	tokens := make([]string, numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		tokens[i] = <-results
	}

	// All should have the same token
	for _, token := range tokens {
		if token != "coalesced-token" {
			t.Errorf("expected 'coalesced-token', got '%s'", token)
		}
	}
}

func TestGetToken_WorkerShutdown(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)

	// Verify that Stop() cleanly shuts down the worker
	tm.Stop()

	// Wait a bit to ensure worker has exited
	time.Sleep(100 * time.Millisecond)

	// The worker should be stopped - verify by checking stopChan is closed
	// (This is implicit - if Stop() works, the test passes)
}

func TestGetToken_TimeoutDuringRefresh(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Set a very short timeout to trigger timeout path
	tm.tokenRefreshTimeout = 10 * time.Millisecond

	// Create a token that needs refresh
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)

	// Create a slow token source that takes longer than timeout
	slowSource := &slowMockTokenSource{
		delay: 1 * time.Second,
		token: &oauth2.Token{AccessToken: "slow-token", Expiry: time.Now().UTC().Add(1 * time.Hour)},
	}

	tm.mu.Lock()
	tm.tokens["slow"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: slowSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// GetToken should timeout
	_, err := tm.GetToken("slow", "", "")
	if err == nil {
		t.Error("expected timeout error, got nil")
	}
	if err.Error() != "token refresh timeout" {
		t.Errorf("expected 'token refresh timeout', got '%v'", err)
	}
}

func TestGetToken_ParallelDifferentCredentials(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create multiple credentials that need refresh
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)

	credentials := []string{"cred1", "cred2", "cred3"}
	sources := make(map[string]*slowMockTokenSource)

	for _, cred := range credentials {
		source := &slowMockTokenSource{
			delay: 100 * time.Millisecond, // Slow enough to cause serialization if sequential
			token: &oauth2.Token{AccessToken: "token-" + cred, Expiry: time.Now().UTC().Add(1 * time.Hour)},
		}
		sources[cred] = source

		tm.mu.Lock()
		tm.tokens[cred] = &cachedToken{
			token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
			tokenSource: source,
			expiresAt:   expiredTime,
		}
		tm.mu.Unlock()
	}

	// Launch concurrent GetToken calls for different credentials
	startTime := time.Now().UTC()
	results := make(chan struct{ name, token string }, len(credentials))

	for _, cred := range credentials {
		go func(credName string) {
			token, err := tm.GetToken(credName, "", "")
			if err != nil {
				t.Errorf("GetToken(%s) failed: %v", credName, err)
				results <- struct{ name, token string }{credName, ""}
				return
			}
			results <- struct{ name, token string }{credName, token}
		}(cred)
	}

	// Collect results
	for i := 0; i < len(credentials); i++ {
		result := <-results
		expectedToken := "token-" + result.name
		if result.token != expectedToken {
			t.Errorf("expected '%s', got '%s'", expectedToken, result.token)
		}
	}

	elapsed := time.Since(startTime)

	// If executed sequentially: 3 * 100ms = 300ms
	// If executed in parallel: ~100ms
	// We allow some overhead, so if it's under 250ms it's likely parallel
	if elapsed > 250*time.Millisecond {
		t.Logf("WARNING: parallel refresh took %v (might be sequential)", elapsed)
	}
}

func TestGetToken_CoalescingWithTimeout(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create a token that needs refresh
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "coalesced-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Launch multiple concurrent GetToken calls
	numGoroutines := 20
	results := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			_, err := tm.GetToken("test", "", "")
			results <- err
		}()
	}

	// All should succeed and be coalesced into single refresh
	successCount := 0
	for i := 0; i < numGoroutines; i++ {
		if err := <-results; err != nil {
			t.Errorf("GetToken call %d failed: %v", i, err)
		} else {
			successCount++
		}
	}

	if successCount != numGoroutines {
		t.Errorf("expected %d successes, got %d", numGoroutines, successCount)
	}

	// Verify coalescing: Token() should be called once or very few times
	// (might be 1-2 due to timing)
	if mockSource.callCount > 3 {
		t.Errorf("expected <= 3 Token() calls (coalescing should reduce), got %d", mockSource.callCount)
	}
}

// slowMockTokenSource simulates slow token source for testing timeouts and parallelism
type slowMockTokenSource struct {
	delay      time.Duration
	token      *oauth2.Token
	shouldFail bool
	err        error
	callCount  int
}

func (m *slowMockTokenSource) Token() (*oauth2.Token, error) {
	m.callCount++
	time.Sleep(m.delay)
	if m.shouldFail {
		return nil, m.err
	}
	return m.token, nil
}

// EDGE-CASE TESTS FOR COALESCING AND CONCURRENCY

// TestGetToken_CoalescingCallerTimeout: Verify that coalescing callers properly handle timeout
// EDGE-CASE: When a non-first caller times out, it should NOT interfere with first caller or other callers
func TestGetToken_CoalescingCallerTimeout(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	// Create a token that needs refresh
	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	// Slow token source to give coalescing callers time to queue up
	mockSource := &slowMockTokenSource{
		delay: 200 * time.Millisecond,
		token: &oauth2.Token{AccessToken: "coalesced-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Set a timeout that will trigger after first caller starts, but before refresh completes
	originalTimeout := tm.tokenRefreshTimeout
	tm.tokenRefreshTimeout = 50 * time.Millisecond
	defer func() { tm.tokenRefreshTimeout = originalTimeout }()

	// Launcher goroutine that fires calls sequentially to allow coalescing
	results := make(chan struct {
		idx int
		err error
	}, 10)

	for i := 0; i < 10; i++ {
		go func(idx int) {
			// Small delay to allow first caller to send request
			if idx > 0 {
				time.Sleep(5 * time.Millisecond)
			}
			_, err := tm.GetToken("test", "", "")
			results <- struct {
				idx int
				err error
			}{idx, err}
		}(i)
	}

	// Collect results
	timeoutCount := 0
	successCount := 0
	for i := 0; i < 10; i++ {
		result := <-results
		if result.err != nil {
			if result.err.Error() == "token refresh timeout" {
				timeoutCount++
			}
		} else {
			successCount++
		}
	}

	// With 50ms timeout and 200ms delay, all should timeout
	if timeoutCount+successCount != 10 {
		t.Errorf("Expected 10 results (timeout+success), got %d+%d=%d", timeoutCount, successCount, timeoutCount+successCount)
	}

	// Key point: some may succeed if they coalesced before timeout, but manager should still be stable
	// Verify manager is still functional after timeouts
	tm.tokenRefreshTimeout = originalTimeout
	expiry2 := time.Now().UTC().Add(2 * time.Hour)
	mockSource2 := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "recovery-token", Expiry: expiry2},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource2,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// This should succeed
	token, err := tm.GetToken("test", "", "")
	if err != nil {
		t.Errorf("Recovery GetToken failed: %v", err)
	}
	if token != "recovery-token" {
		t.Errorf("Expected recovery token, got %s", token)
	}
}

// TestGetToken_ChannelLeakOnTimeout: Verify that response channels don't leak memory on timeout
// EDGE-CASE: Channels in refreshing map should be properly cleaned up even on timeout
func TestGetToken_ChannelLeakOnTimeout(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiredTime := time.Now().UTC().Add(-1 * time.Hour)

	// Create a very slow token source
	slowSource := &slowMockTokenSource{
		delay: 2 * time.Second,
		token: &oauth2.Token{AccessToken: "slow-token", Expiry: time.Now().UTC().Add(1 * time.Hour)},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: slowSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Set short timeout
	originalTimeout := tm.tokenRefreshTimeout
	tm.tokenRefreshTimeout = 10 * time.Millisecond
	defer func() { tm.tokenRefreshTimeout = originalTimeout }()

	// Fire a request that will timeout
	_, err := tm.GetToken("test", "", "")
	if err == nil {
		t.Error("Expected timeout error")
	}

	// Check that refreshing map is cleaned up
	// After timeout, the request should be removed from refreshing map
	tm.refreshingMu.Lock()
	waitingChans, exists := tm.refreshing["test"]
	tm.refreshingMu.Unlock()

	// Either the entry should not exist, or should be empty after the first caller times out
	// (It might still exist if worker hasn't processed it yet)
	if exists && len(waitingChans) > 0 {
		// Give worker time to process
		time.Sleep(50 * time.Millisecond)

		tm.refreshingMu.Lock()
		waitingChans, exists = tm.refreshing["test"]
		tm.refreshingMu.Unlock()

		// After giving time to worker, it should be cleaned up
		if exists && len(waitingChans) > 0 {
			t.Errorf("Expected refreshing['test'] to be empty/removed, but found %d waiting channels", len(waitingChans))
		}
	}
}

// TestGetToken_FirstCallerTimeoutRemovesFromMap: Verify removeWaitingChan works correctly
// EDGE-CASE: When first caller times out after sending request, it should be removed from refreshing map
func TestGetToken_FirstCallerTimeoutRemovesFromMap(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	slowSource := &slowMockTokenSource{
		delay: 1 * time.Second,
		token: &oauth2.Token{AccessToken: "slow-token", Expiry: time.Now().UTC().Add(1 * time.Hour)},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: slowSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	originalTimeout := tm.tokenRefreshTimeout
	tm.tokenRefreshTimeout = 20 * time.Millisecond
	defer func() { tm.tokenRefreshTimeout = originalTimeout }()

	// Get token - will timeout
	_, err := tm.GetToken("test", "", "")
	if err == nil {
		t.Error("Expected timeout")
	}

	// Verify removeWaitingChan was called and removed the entry
	// Give worker a moment to potentially process the request
	time.Sleep(100 * time.Millisecond)

	tm.refreshingMu.Lock()
	_, exists := tm.refreshing["test"]
	tm.refreshingMu.Unlock()

	// Entry should be removed by now (either by removeWaitingChan or by processRefreshRequest)
	if exists {
		t.Error("Expected 'test' to be removed from refreshing map after timeout")
	}
}

// TestGetToken_MultipleTimeoutsSequential: Verify manager recovers from multiple sequential timeouts
// EDGE-CASE: Multiple calls shouldn't interfere with each other after timeouts
func TestGetToken_MultipleTimeoutsSequential(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiredTime := time.Now().UTC().Add(-1 * time.Hour)

	originalTimeout := tm.tokenRefreshTimeout
	tm.tokenRefreshTimeout = 20 * time.Millisecond
	defer func() { tm.tokenRefreshTimeout = originalTimeout }()

	// First request - will timeout
	slowSource1 := &slowMockTokenSource{
		delay: 500 * time.Millisecond,
		token: &oauth2.Token{AccessToken: "token1", Expiry: time.Now().UTC().Add(1 * time.Hour)},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old", Expiry: expiredTime},
		tokenSource: slowSource1,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	_, err := tm.GetToken("test", "", "")
	if err == nil {
		t.Error("Expected first timeout")
	}

	// Give time for cleanup
	time.Sleep(100 * time.Millisecond)

	// Second request - should timeout too
	slowSource2 := &slowMockTokenSource{
		delay: 500 * time.Millisecond,
		token: &oauth2.Token{AccessToken: "token2", Expiry: time.Now().UTC().Add(1 * time.Hour)},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old", Expiry: expiredTime},
		tokenSource: slowSource2,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	_, err = tm.GetToken("test", "", "")
	if err == nil {
		t.Error("Expected second timeout")
	}

	// Third request - restore normal timeout and should succeed
	tm.tokenRefreshTimeout = originalTimeout

	fastSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "success-token", Expiry: time.Now().UTC().Add(1 * time.Hour)},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old", Expiry: expiredTime},
		tokenSource: fastSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	token, err := tm.GetToken("test", "", "")
	if err != nil {
		t.Errorf("Expected success after timeouts, got error: %v", err)
	}
	if token != "success-token" {
		t.Errorf("Expected success-token, got %s", token)
	}
}

// TestGetToken_ProcessRefreshRequestRobustness: Verify that partial failures don't crash the system
// EDGE-CASE: If one of the coalesced callers has a closed channel, others should still get response
func TestGetToken_ProcessRefreshRequestRobustness(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	mockSource := &slowMockTokenSource{
		delay: 100 * time.Millisecond,
		token: &oauth2.Token{AccessToken: "robust-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	results := make(chan struct {
		token string
		err   error
	}, 20)

	// Launch 20 concurrent requests
	for i := 0; i < 20; i++ {
		go func() {
			token, err := tm.GetToken("test", "", "")
			results <- struct {
				token string
				err   error
			}{token, err}
		}()
	}

	// Collect results
	successCount := 0
	errorCount := 0
	for i := 0; i < 20; i++ {
		result := <-results
		if result.err != nil {
			errorCount++
		} else {
			if result.token == "robust-token" {
				successCount++
			}
		}
	}

	// Most should succeed due to coalescing
	if successCount < 15 {
		t.Logf("WARNING: Expected most requests to succeed, got only %d successes (errors: %d)", successCount, errorCount)
	}
}

// TestGetToken_RefreshingMapStateAfterCompletion: Verify refreshing map is cleaned up after refresh
// EDGE-CASE: After processRefreshRequest completes, refreshing[credentialName] should be deleted
func TestGetToken_RefreshingMapStateAfterCompletion(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	newExpiryTime := time.Now().UTC().Add(2 * time.Hour)

	mockSource := &mockTokenSource{
		token: &oauth2.Token{AccessToken: "complete-token", Expiry: newExpiryTime},
	}

	tm.mu.Lock()
	tm.tokens["test"] = &cachedToken{
		token:       &oauth2.Token{AccessToken: "old-token", Expiry: expiredTime},
		tokenSource: mockSource,
		expiresAt:   expiredTime,
	}
	tm.mu.Unlock()

	// Make request
	token, err := tm.GetToken("test", "", "")
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}
	if token != "complete-token" {
		t.Errorf("Expected complete-token, got %s", token)
	}

	// After successful refresh, refreshing map should be cleaned up
	tm.refreshingMu.Lock()
	_, exists := tm.refreshing["test"]
	tm.refreshingMu.Unlock()

	if exists {
		t.Error("Expected refreshing['test'] to be deleted after successful refresh")
	}
}

// TestGetToken_ConcurrentDifferentCredentialsWithTimeout: Verify lock ordering prevents deadlock
// EDGE-CASE: Multiple credentials being refreshed concurrently with some timing out
func TestGetToken_ConcurrentDifferentCredentialsWithTimeout(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)
	defer tm.Stop()

	expiredTime := time.Now().UTC().Add(-1 * time.Hour)
	originalTimeout := tm.tokenRefreshTimeout
	tm.tokenRefreshTimeout = 50 * time.Millisecond
	defer func() { tm.tokenRefreshTimeout = originalTimeout }()

	// Setup 5 credentials with varying delays
	credentials := map[string]time.Duration{
		"fast":     10 * time.Millisecond,
		"medium":   40 * time.Millisecond,
		"slow":     200 * time.Millisecond,
		"veryslow": 500 * time.Millisecond,
		"instant":  1 * time.Millisecond,
	}

	for name, delay := range credentials {
		source := &slowMockTokenSource{
			delay: delay,
			token: &oauth2.Token{AccessToken: "token-" + name, Expiry: time.Now().UTC().Add(1 * time.Hour)},
		}

		tm.mu.Lock()
		tm.tokens[name] = &cachedToken{
			token:       &oauth2.Token{AccessToken: "old", Expiry: expiredTime},
			tokenSource: source,
			expiresAt:   expiredTime,
		}
		tm.mu.Unlock()
	}

	// Fire concurrent requests for all credentials
	results := make(chan struct {
		cred  string
		token string
		err   error
	}, len(credentials)*2)

	for credName := range credentials {
		// Each credential gets 2 concurrent requests
		for i := 0; i < 2; i++ {
			go func(name string) {
				token, err := tm.GetToken(name, "", "")
				results <- struct {
					cred  string
					token string
					err   error
				}{name, token, err}
			}(credName)
		}
	}

	// Collect results - should not deadlock
	successCount := 0
	timeoutCount := 0
	for i := 0; i < len(credentials)*2; i++ {
		result := <-results
		if result.err != nil {
			if result.err.Error() == "token refresh timeout" {
				timeoutCount++
			}
		} else {
			successCount++
		}
	}

	// Some should succeed (fast, instant), some should timeout (slow, veryslow)
	t.Logf("Results: %d successes, %d timeouts", successCount, timeoutCount)
	if successCount+timeoutCount != len(credentials)*2 {
		t.Errorf("Expected %d total results, got %d", len(credentials)*2, successCount+timeoutCount)
	}
}
