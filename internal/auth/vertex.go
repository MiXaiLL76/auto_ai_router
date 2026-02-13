package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// tokenRefreshRequest represents a request to refresh a token
type tokenRefreshRequest struct {
	credentialName  string
	credentialsFile string
	credentialsJSON string
	responseChan    chan tokenRefreshResponse
}

// tokenRefreshResponse represents a response to a refresh request
type tokenRefreshResponse struct {
	token string
	err   error
}

// VertexTokenManager manages OAuth2 tokens for Vertex AI credentials
type VertexTokenManager struct {
	mu                  sync.RWMutex
	tokens              map[string]*cachedToken
	credentials         map[string][]byte // Cache for credentials
	logger              *slog.Logger
	tokenRefresh        time.Duration
	tokenRefreshTimeout time.Duration // Timeout for token refresh operations
	refreshRequests     chan tokenRefreshRequest
	stopChan            chan struct{}
	stopOnce            sync.Once
	stopped             atomic.Bool
	refreshing          map[string][]chan tokenRefreshResponse // Coalescing waiting requests
	refreshingMu        sync.Mutex
	wg                  sync.WaitGroup // Track active refresh operations
}

// cachedToken represents a cached OAuth2 token with expiry
type cachedToken struct {
	token       *oauth2.Token
	tokenSource oauth2.TokenSource
	expiresAt   time.Time
}

// NewVertexTokenManager creates a new token manager
func NewVertexTokenManager(logger *slog.Logger) *VertexTokenManager {
	tm := &VertexTokenManager{
		tokens:              make(map[string]*cachedToken),
		credentials:         make(map[string][]byte),
		logger:              logger,
		tokenRefresh:        5 * time.Minute,  // Refresh 5 minutes before expiry
		tokenRefreshTimeout: 30 * time.Second, // Default timeout for refresh operations
		refreshRequests:     make(chan tokenRefreshRequest, 100),
		stopChan:            make(chan struct{}),
		refreshing:          make(map[string][]chan tokenRefreshResponse),
	}
	tm.wg.Add(1)
	go tm.refreshWorker()
	return tm
}

// GetToken returns a valid OAuth2 token for the given credential.
// It loads credentials from file or JSON string and caches the token.
//
// Token acquisition uses a two-path strategy:
//   - Fast path: Returns cached token if still valid (not expiring within tokenRefresh window)
//   - Slow path: If refresh needed, coalesces concurrent requests to avoid redundant API calls
//
// Coalescing strategy:
// Multiple concurrent callers requesting the same credential will be grouped together.
// The first caller sends the refresh request to the worker goroutine via refreshRequests channel.
// Subsequent callers within the same refresh window are added to refreshing[credentialName] slice
// and their response channels are batched together. When the worker completes the refresh,
// it sends the same response to all waiting channels in the batch.
// This avoids thundering herd scenarios where hundreds of goroutines would independently
// attempt to refresh the same token.
//
// Response channel buffering: Each response channel has a buffer size of 1, which allows
// the worker to send responses non-blocking. If a waiter has already given up due to timeout,
// the response will still be sent but go unread.
func (tm *VertexTokenManager) GetToken(credentialName, credentialsFile, credentialsJSON string) (string, error) {
	if tm.stopped.Load() {
		return "", fmt.Errorf("token manager is stopped")
	}

	// Fast path: check if we have a valid cached token (read-only)
	tm.mu.RLock()
	if cached, exists := tm.tokens[credentialName]; exists {
		if utils.NowUTC().Before(cached.expiresAt.Add(-tm.tokenRefresh)) {
			token := cached.token.AccessToken
			tm.mu.RUnlock()
			return token, nil
		}
	}
	tm.mu.RUnlock()

	// Slow path: need to refresh or create token
	// Coalesce concurrent refresh requests
	// IMPORTANT: Lock ordering is critical to avoid deadlock:
	// Always acquire mu lock BEFORE refreshingMu lock
	tm.mu.RLock()

	// Double-check cache to prevent race condition: cache may have been updated
	// after our first check and before acquiring this lock
	if cached, exists := tm.tokens[credentialName]; exists {
		if utils.NowUTC().Before(cached.expiresAt.Add(-tm.tokenRefresh)) {
			token := cached.token.AccessToken
			tm.mu.RUnlock()
			return token, nil
		}
	}
	tm.mu.RUnlock()

	// Only allocate channel if we actually need to refresh (avoids allocation pressure)
	responseChan := make(chan tokenRefreshResponse, 1)

	ctx, cancel := context.WithTimeout(context.Background(), tm.tokenRefreshTimeout)
	defer cancel()

	tm.refreshingMu.Lock()
	waitingChans, isRefreshing := tm.refreshing[credentialName]
	if !isRefreshing {
		// First request - send to worker
		tm.refreshing[credentialName] = []chan tokenRefreshResponse{responseChan}
		tm.refreshingMu.Unlock()

		req := tokenRefreshRequest{
			credentialName:  credentialName,
			credentialsFile: credentialsFile,
			credentialsJSON: credentialsJSON,
			responseChan:    responseChan,
		}
		select {
		case tm.refreshRequests <- req:
		case <-ctx.Done():
			tm.removeWaitingChan(credentialName, responseChan)
			return "", fmt.Errorf("token refresh timeout")
		}
	} else {
		// Coalesce with existing refresh
		tm.refreshing[credentialName] = append(waitingChans, responseChan)
		tm.refreshingMu.Unlock()
	}

	// Wait for response with timeout
	select {
	case resp := <-responseChan:
		return resp.token, resp.err
	case <-ctx.Done():
		// Important: ALL callers (both first and coalescing) must remove their channel
		// from refreshing map on timeout to prevent channel leak and ensure proper cleanup.
		// Coalescing callers that timeout must be removed so processRefreshRequest doesn't
		// attempt to send response to an abandoned channel.
		tm.removeWaitingChan(credentialName, responseChan)
		return "", fmt.Errorf("token refresh timeout")
	}
}

// refreshWorker is a background goroutine that handles token refresh requests
// It spawns a goroutine per request to allow parallel refresh of different credentials
// Coalescing ensures only one refresh per credential at a time
func (tm *VertexTokenManager) refreshWorker() {
	defer tm.wg.Done()
	for {
		select {
		case <-tm.stopChan:
			return
		case req, ok := <-tm.refreshRequests:
			if !ok {
				return
			}
			tm.wg.Add(1)
			// Process each request in a separate goroutine to allow parallel refresh
			// of different credentials while coalescing maintains single refresh per credential
			go func(r tokenRefreshRequest) {
				defer tm.wg.Done()
				tm.processRefreshRequest(r)
			}(req)
		}
	}
}

// processRefreshRequest handles a single refresh request and notifies all coalesced waiters
// Panic recovery ensures that even if token operations fail catastrophically,
// all waiting callers are notified so they don't hang on timeout.
func (tm *VertexTokenManager) processRefreshRequest(req tokenRefreshRequest) {
	var token string
	var err error

	// Recover from panic to ensure all waiters are notified
	defer func() {
		if r := recover(); r != nil {
			tm.logger.Error("Panic during token refresh",
				"credential", req.credentialName,
				"panic", r,
			)
			err = fmt.Errorf("token refresh panicked: %v", r)
			token = ""

			// Send error to all waiting callers even after panic
			tm.refreshingMu.Lock()
			waitingChans, exists := tm.refreshing[req.credentialName]
			delete(tm.refreshing, req.credentialName)
			tm.refreshingMu.Unlock()

			if exists {
				resp := tokenRefreshResponse{token: "", err: err}
				for _, ch := range waitingChans {
					select {
					case ch <- resp:
					default:
					}
				}
			}
		}
	}()

	// Check if token exists and needs refresh
	tm.mu.RLock()
	cached, exists := tm.tokens[req.credentialName]
	tm.mu.RUnlock()
	if exists {
		// Token exists, refresh it
		token, err = tm.refreshToken(req.credentialName, cached)
		// If refresh fails, try to create a new token immediately instead of returning error
		if err != nil {
			tm.mu.Lock()
			delete(tm.tokens, req.credentialName)
			tm.mu.Unlock()
			token, err = tm.createNewToken(req.credentialName, req.credentialsFile, req.credentialsJSON)
		}
	} else {
		// No cached token, create a new one
		token, err = tm.createNewToken(req.credentialName, req.credentialsFile, req.credentialsJSON)
	}

	// Send response to all waiting goroutines
	tm.refreshingMu.Lock()
	waitingChans, exists := tm.refreshing[req.credentialName]
	delete(tm.refreshing, req.credentialName)
	tm.refreshingMu.Unlock()

	if exists {
		resp := tokenRefreshResponse{token: token, err: err}
		failedCount := 0
		for _, ch := range waitingChans {
			select {
			case ch <- resp:
			default:
				// Channel send failed - this can happen if the waiter gave up after timeout
				// before this response was ready. With a buffered channel of size 1, the
				// only way this occurs is if the receiver has already closed or abandoned the channel.
				failedCount++
			}
		}
		if failedCount > 0 {
			tm.logger.Debug("Failed to send token refresh responses",
				"credential", req.credentialName,
				"failed_count", failedCount,
				"total_waiters", len(waitingChans),
			)
		}
	}
}

func (tm *VertexTokenManager) refreshToken(credentialName string, cached *cachedToken) (string, error) {
	tm.logger.Debug("Refreshing Vertex AI token",
		"credential", credentialName,
		"expires_at", cached.expiresAt,
	)

	newToken, err := tm.tokenFromSource(credentialName, "refresh", cached.tokenSource)
	if err != nil {
		return "", err
	}

	// Update cached token
	tm.mu.Lock()
	cached.token = newToken
	cached.expiresAt = newToken.Expiry
	tm.mu.Unlock()
	tm.logger.Info("Vertex AI token refreshed",
		"credential", credentialName,
		"expires_at", newToken.Expiry,
	)
	return newToken.AccessToken, nil
}

func (tm *VertexTokenManager) createNewToken(credentialName, credentialsFile, credentialsJSON string) (string, error) {
	tm.logger.Debug("Creating new Vertex AI token", "credential", credentialName)

	credBytes, err := tm.loadCredentials(credentialName, credentialsFile, credentialsJSON)
	if err != nil {
		return "", err
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
	token, err := tm.tokenFromSource(credentialName, "get initial", creds.TokenSource)
	if err != nil {
		return "", err
	}

	// Cache the token
	tm.mu.Lock()
	tm.tokens[credentialName] = &cachedToken{
		token:       token,
		tokenSource: creds.TokenSource,
		expiresAt:   token.Expiry,
	}
	tm.mu.Unlock()

	tm.logger.Info("Vertex AI token created",
		"credential", credentialName,
		"expires_at", token.Expiry,
	)

	return token.AccessToken, nil
}

func (tm *VertexTokenManager) tokenFromSource(credentialName, action string, source oauth2.TokenSource) (*oauth2.Token, error) {
	token, err := source.Token()
	if err != nil {
		tm.logger.Error("Failed to "+action+" Vertex AI token",
			"credential", credentialName,
			"error", err,
		)
		return nil, fmt.Errorf("failed to %s token: %w", action, err)
	}

	return token, nil
}

func (tm *VertexTokenManager) loadCredentials(credentialName, credentialsFile, credentialsJSON string) ([]byte, error) {
	// Check if credentials are cached
	tm.mu.RLock()
	if cached, exists := tm.credentials[credentialName]; exists {
		tm.mu.RUnlock()
		tm.logger.Debug("Using cached credentials", "credential", credentialName)
		return cached, nil
	}
	tm.mu.RUnlock()

	var credBytes []byte
	var err error

	// Load credentials from file or JSON string
	if credentialsFile != "" {
		credBytes, err = os.ReadFile(credentialsFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read credentials file %s: %w", credentialsFile, err)
		}
		tm.logger.Debug("Loaded credentials from file",
			"credential", credentialName,
			"file", credentialsFile,
		)
	} else if credentialsJSON != "" {
		credBytes = []byte(credentialsJSON)
		tm.logger.Debug("Using credentials from config", "credential", credentialName)
	} else {
		return nil, fmt.Errorf("no credentials provided for %s", credentialName)
	}

	// Cache the credentials
	tm.mu.Lock()
	if cached, exists := tm.credentials[credentialName]; exists {
		tm.mu.Unlock()
		return cached, nil
	}
	tm.credentials[credentialName] = credBytes
	tm.mu.Unlock()
	return credBytes, nil
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

// Stop gracefully stops the token manager and its worker goroutine
func (tm *VertexTokenManager) Stop() {
	tm.stopOnce.Do(func() {
		tm.stopped.Store(true)
		close(tm.stopChan)
		tm.failAllWaiters(fmt.Errorf("token manager is stopped"))
	})
	tm.wg.Wait() // Wait for all active refresh operations to complete
}

func (tm *VertexTokenManager) failAllWaiters(err error) {
	tm.refreshingMu.Lock()
	waiting := tm.refreshing
	tm.refreshing = make(map[string][]chan tokenRefreshResponse)
	tm.refreshingMu.Unlock()

	if len(waiting) == 0 {
		return
	}

	resp := tokenRefreshResponse{token: "", err: err}
	failedCount := 0
	totalWaiters := 0
	for _, chans := range waiting {
		for _, ch := range chans {
			totalWaiters++
			select {
			case ch <- resp:
			default:
				failedCount++
			}
		}
	}
	if failedCount > 0 {
		tm.logger.Debug("Failed to send stop responses to token waiters",
			"failed_count", failedCount,
			"total_waiters", totalWaiters,
		)
	}
}

// removeWaitingChan removes a specific response channel from the waiting list.
// This is called when a caller times out or context is cancelled.
// It prevents processRefreshRequest from sending responses to abandoned channels.
func (tm *VertexTokenManager) removeWaitingChan(credentialName string, responseChan chan tokenRefreshResponse) {
	tm.refreshingMu.Lock()
	defer tm.refreshingMu.Unlock()

	waitingChans, exists := tm.refreshing[credentialName]
	if !exists {
		return
	}

	// Find and create new slice without the channel
	newChans := make([]chan tokenRefreshResponse, 0, len(waitingChans)-1)
	for _, ch := range waitingChans {
		if ch != responseChan {
			newChans = append(newChans, ch)
		}
	}

	// Update or delete from map based on remaining channels
	if len(newChans) == 0 {
		delete(tm.refreshing, credentialName)
	} else {
		tm.refreshing[credentialName] = newChans
	}
}
