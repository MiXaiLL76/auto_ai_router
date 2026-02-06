package proxy

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/stretchr/testify/assert"
)

// TestLogMasking_MasterKeyNotExposed verifies that master key errors don't leak raw tokens
func TestLogMasking_MasterKeyNotExposed(t *testing.T) {
	// Create a logger that writes to a buffer so we can inspect output
	var logBuf bytes.Buffer

	// Create handler that writes structured logs as text
	handler := slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})
	logger := slog.New(handler)

	// Create test balancer and proxy
	bal, rl := createTestBalancer("http://test.com")
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 30*time.Second, metrics,
		"sk_test_valid_master_key_12345", // Real master key
		rl, tm, mm, "test-version", "test-commit",
	)

	// Create request with wrong token
	invalidToken := "sk_test_invalid_token_abcdefghijklmnop"
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer "+invalidToken)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()

	// Make request - this will fail auth and log the error
	prx.ProxyRequest(w, req)

	// Check response is unauthorized
	assert.Equal(t, http.StatusUnauthorized, w.Code)

	// Check that full token is NOT in logs
	logOutput := logBuf.String()
	assert.NotContains(t, logOutput, invalidToken, "Full invalid token should not appear in logs")
	assert.NotContains(t, logOutput, "invalid_token_abcdefgh", "Token suffix should not appear in logs")
	assert.NotContains(t, logOutput, "valid_master_key", "Master key should not appear in logs")

	// Check that masked token IS in logs
	assert.Contains(t, logOutput, "sk_t", "Masked token prefix should appear in logs")
}

// TestLogMasking_HeadersNotExposed verifies that auth headers are not exposed in debug logs
func TestLogMasking_HeadersNotExposed(t *testing.T) {
	var logBuf bytes.Buffer
	handler := slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})
	logger := slog.New(handler)

	// Mock upstream server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the upstream request has our API key
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer upstream-key-")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result": "ok"}`))
	}))
	defer mockServer.Close()

	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 30*time.Second, metrics,
		"master-key", rl, tm, mm, "test-version", "test-commit",
	)

	// Create request
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Custom-Header", "custom-value")

	w := httptest.NewRecorder()

	// Make request
	prx.ProxyRequest(w, req)

	// Check response is OK
	assert.Equal(t, http.StatusOK, w.Code)

	// Check logs don't expose auth headers
	logOutput := logBuf.String()
	assert.NotContains(t, logOutput, "master-key", "Master key should not appear in logs")
	assert.NotContains(t, logOutput, "upstream-key-", "Upstream API key should not appear in logs")
	// Authorization header is intentionally skipped in debug logs (see headers.go)
	assert.NotContains(t, logOutput, "Authorization", "Authorization header should not appear in debug logs")
	assert.NotContains(t, logOutput, "X-Api-Key", "X-Api-Key header should not appear in debug logs")
}

// TestMaskKey_VariousFormats tests maskKey with various credential formats
func TestMaskKey_VariousFormats(t *testing.T) {
	tests := []struct {
		name        string
		key         string
		shouldNotBe string // What should NOT appear in the masked result
	}{
		{
			name:        "OpenAI API key format",
			key:         "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890",
			shouldNotBe: "AbCdEfGhIjKlMnOpQrStUvWxYz1234567890",
		},
		{
			name:        "Bearer token format",
			key:         "sk_live_1234567890abcdefghijklmnopqrstuvwxyz",
			shouldNotBe: "1234567890abcdefghijklmnopqrstuvwxyz",
		},
		{
			name:        "Long API key",
			key:         "gsk_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz567",
			shouldNotBe: "def456ghi789jkl012mno345pqr678stu901vwx234yz567",
		},
		{
			name:        "Short key should be masked",
			key:         "short",
			shouldNotBe: "hort", // Rest of the string after first char
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			masked := security.MaskAPIKey(tt.key)
			assert.NotContains(t, masked, tt.shouldNotBe,
				"Masked key should not contain credential secret part")
		})
	}
}

// TestMaskToken_VariousFormats tests maskToken with various hashed token formats
func TestMaskToken_VariousFormats(t *testing.T) {
	tests := []struct {
		name        string
		token       string
		shouldNotBe string
	}{
		{
			name:        "SHA256 hash - full",
			token:       "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
			shouldNotBe: "d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
		},
		{
			name:        "Long hash value",
			token:       "abcdef0123456789abcdef0123456789abcdef0123456789",
			shouldNotBe: "def0123456789abcdef0123456789abcdef0123456789",
		},
		{
			name:        "Short token",
			token:       "short",
			shouldNotBe: "ort", // Rest after first 4 chars
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			masked := security.MaskToken(tt.token)
			assert.NotContains(t, masked, tt.shouldNotBe,
				"Masked token should not contain secret part")
		})
	}
}

// TestMaskingConsistency verifies that maskKey and maskToken have similar security properties
func TestMaskingConsistency(t *testing.T) {
	testCases := []string{
		"1",
		"12",
		"123",
		"1234",
		"12345",
		"123456789",
		"abcdefghijklmnopqrstuvwxyz",
		"sk_test_1234567890abcdefghijklmnopqrstuvwxyz",
	}

	for _, test := range testCases {
		maskedKey := security.MaskAPIKey(test)
		maskedToken := security.MaskToken(test)

		// Both should have similar structure: prefix + "..." or full string if short
		if len(test) <= 4 {
			// Short keys/tokens are not masked or minimally masked
			assert.True(t, len(maskedKey) <= len(test)+3,
				"Short key masking should be minimal")
			assert.True(t, len(maskedToken) <= len(test)+3,
				"Short token masking should be minimal")
		} else {
			// Both should use the same masking pattern: first 4 chars + "..."
			assert.Equal(t, test[:4]+"...", maskedKey,
				"Key should show first 4 chars + ...")
			assert.Equal(t, test[:4]+"...", maskedToken,
				"Token should show first 4 chars + ... (matching key format)")
		}
	}
}

// TestNoRawTokensInErrorMessages verifies error handling doesn't leak tokens
func TestNoRawTokensInErrorMessages(t *testing.T) {
	var logBuf bytes.Buffer
	handler := slog.NewTextHandler(&logBuf, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})
	logger := slog.New(handler)

	testToken := "sk_test_super_secret_token_12345678901234567890"

	// Create proxy with minimal setup
	bal, rl := createTestBalancer("http://test.com")
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 30*time.Second, metrics,
		"master-key", rl, tm, mm, "test-version", "test-commit",
	)

	// Simulate auth error
	w := httptest.NewRecorder()
	result := prx.validateMasterKeyOrError(w, testToken)

	// Should fail because token doesn't match master-key
	assert.False(t, result)

	// Check logs
	logOutput := logBuf.String()

	// Raw token should NOT be in logs
	assert.NotContains(t, logOutput, testToken, "Full token should not be in logs")
	assert.NotContains(t, logOutput, "secret_token_123456", "Token secret part should not be in logs")

	// But masked prefix should be
	assert.Contains(t, logOutput, "sk_t", "Masked token prefix should be in logs")
}

// BenchmarkMaskKey benchmarks the maskKey function
func BenchmarkMaskKey(b *testing.B) {
	key := "sk_live_1234567890abcdefghijklmnopqrstuvwxyz_production_key"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = security.MaskAPIKey(key)
	}
}

// BenchmarkMaskToken benchmarks the maskToken function
func BenchmarkMaskToken(b *testing.B) {
	token := "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = security.MaskToken(token)
	}
}

// TestMaskingEdgeCases tests edge cases for masking functions
func TestMaskingEdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		expectedKey   string
		expectedToken string
	}{
		{
			name:          "empty string",
			input:         "",
			expectedKey:   "",
			expectedToken: "",
		},
		{
			name:          "single character",
			input:         "a",
			expectedKey:   "***",
			expectedToken: "***",
		},
		{
			name:          "exactly 4 characters",
			input:         "abcd",
			expectedKey:   "***",
			expectedToken: "***",
		},
		{
			name:          "5 characters",
			input:         "abcde",
			expectedKey:   "abcd...",
			expectedToken: "abcd...",
		},
		{
			name:          "unicode characters",
			input:         "абвгд1234567890",
			expectedKey:   "аб...",
			expectedToken: "аб...",
		},
		{
			name:          "special characters",
			input:         "sk_!@#$%^&*()_+1234567890",
			expectedKey:   "sk_!...",
			expectedToken: "sk_!...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maskedKey := security.MaskAPIKey(tt.input)
			assert.Equal(t, tt.expectedKey, maskedKey, "maskKey mismatch")

			maskedToken := security.MaskToken(tt.input)
			assert.Equal(t, tt.expectedToken, maskedToken, "maskToken mismatch")
		})
	}
}
