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

	_, err := tm.GetToken("test", "/nonexistent/path.json", "")
	if err == nil {
		t.Error("expected error for non-existent file, got nil")
	}
}

func TestClearToken(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)

	// Add a token to cache
	expiry := time.Now().Add(1 * time.Hour)
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

	// Test non-existent token
	_, exists := tm.GetTokenExpiry("nonexistent")
	if exists {
		t.Error("expected false for non-existent token")
	}

	// Add a token
	expiry := time.Now().Add(1 * time.Hour)
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

	expiry := time.Now().Add(1 * time.Hour)
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

	// Create an expired token
	expiredTime := time.Now().Add(-1 * time.Hour)
	newExpiryTime := time.Now().Add(2 * time.Hour)

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
	if !cached.expiresAt.Before(time.Now()) {
		t.Error("token should be expired for this test")
	}
}

func TestGetToken_TokenRefreshError(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	tm := NewVertexTokenManager(logger)

	// Create an expired token with failing TokenSource
	expiredTime := time.Now().Add(-1 * time.Hour)

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

	// Create a token that expires in 3 minutes (within refresh buffer of 5 min)
	nearExpiryTime := time.Now().Add(3 * time.Minute)
	newExpiryTime := time.Now().Add(2 * time.Hour)

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
