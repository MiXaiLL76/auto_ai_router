package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// VertexTokenManager manages OAuth2 tokens for Vertex AI credentials
type VertexTokenManager struct {
	mu           sync.RWMutex
	tokens       map[string]*cachedToken
	logger       *slog.Logger
	tokenRefresh time.Duration
}

// cachedToken represents a cached OAuth2 token with expiry
type cachedToken struct {
	token       *oauth2.Token
	tokenSource oauth2.TokenSource
	expiresAt   time.Time
}

// NewVertexTokenManager creates a new token manager
func NewVertexTokenManager(logger *slog.Logger) *VertexTokenManager {
	return &VertexTokenManager{
		tokens:       make(map[string]*cachedToken),
		logger:       logger,
		tokenRefresh: 5 * time.Minute, // Refresh 5 minutes before expiry
	}
}

// GetToken returns a valid OAuth2 token for the given credential
// It loads credentials from file or JSON string and caches the token
func (tm *VertexTokenManager) GetToken(credentialName, credentialsFile, credentialsJSON string) (string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Check if we have a valid cached token
	if cached, exists := tm.tokens[credentialName]; exists {
		// If token is still valid (not expired and has buffer time), return it
		if time.Now().Before(cached.expiresAt.Add(-tm.tokenRefresh)) {
			return cached.token.AccessToken, nil
		}

		// Token is expired or about to expire, refresh it
		tm.logger.Debug("Refreshing Vertex AI token",
			"credential", credentialName,
			"expires_at", cached.expiresAt,
		)

		newToken, err := cached.tokenSource.Token()
		if err != nil {
			tm.logger.Error("Failed to refresh Vertex AI token",
				"credential", credentialName,
				"error", err,
			)
			// Remove invalid token from cache
			delete(tm.tokens, credentialName)
			return "", fmt.Errorf("failed to refresh token: %w", err)
		}

		// Update cached token
		cached.token = newToken
		cached.expiresAt = newToken.Expiry
		tm.logger.Info("Vertex AI token refreshed",
			"credential", credentialName,
			"expires_at", newToken.Expiry,
		)
		return newToken.AccessToken, nil
	}

	// No cached token, create a new one
	tm.logger.Debug("Creating new Vertex AI token", "credential", credentialName)

	var credBytes []byte
	var err error

	// Load credentials from file or JSON string
	if credentialsFile != "" {
		credBytes, err = os.ReadFile(credentialsFile)
		if err != nil {
			return "", fmt.Errorf("failed to read credentials file %s: %w", credentialsFile, err)
		}
		tm.logger.Debug("Loaded credentials from file",
			"credential", credentialName,
			"file", credentialsFile,
		)
	} else if credentialsJSON != "" {
		credBytes = []byte(credentialsJSON)
		tm.logger.Debug("Using credentials from config", "credential", credentialName)
	} else {
		return "", fmt.Errorf("no credentials provided for %s", credentialName)
	}

	// Parse and validate service account JSON
	var serviceAccount map[string]interface{}
	if err := json.Unmarshal(credBytes, &serviceAccount); err != nil {
		return "", fmt.Errorf("invalid service account JSON: %w", err)
	}

	// Verify it's a service account
	if accountType, ok := serviceAccount["type"].(string); !ok || accountType != "service_account" {
		return "", fmt.Errorf("credentials must be for a service account, got type: %v", serviceAccount["type"])
	}

	// Create credentials with Vertex AI scope
	creds, err := google.CredentialsFromJSON(
		context.Background(),
		credBytes,
		"https://www.googleapis.com/auth/cloud-platform",
	)
	if err != nil {
		return "", fmt.Errorf("failed to create credentials: %w", err)
	}

	// Get initial token
	token, err := creds.TokenSource.Token()
	if err != nil {
		return "", fmt.Errorf("failed to get initial token: %w", err)
	}

	// Cache the token
	tm.tokens[credentialName] = &cachedToken{
		token:       token,
		tokenSource: creds.TokenSource,
		expiresAt:   token.Expiry,
	}

	tm.logger.Info("Vertex AI token created",
		"credential", credentialName,
		"expires_at", token.Expiry,
	)

	return token.AccessToken, nil
}

// ClearToken removes a token from the cache (useful for testing or manual refresh)
func (tm *VertexTokenManager) ClearToken(credentialName string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	delete(tm.tokens, credentialName)
	tm.logger.Debug("Cleared cached token", "credential", credentialName)
}

// GetTokenExpiry returns the expiry time of a cached token
func (tm *VertexTokenManager) GetTokenExpiry(credentialName string) (time.Time, bool) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	if cached, exists := tm.tokens[credentialName]; exists {
		return cached.expiresAt, true
	}
	return time.Time{}, false
}
