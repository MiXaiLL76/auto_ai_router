package httputil

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func createTestLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
}

func TestFetchFromProxy_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/models", r.URL.Path)
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data": "test response"}`))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "test-api-key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/api/models", logger)

	assert.NoError(t, err)
	assert.Equal(t, `{"data": "test response"}`, string(body))
}

func TestFetchFromProxy_NoAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "", r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/health", logger)

	assert.NoError(t, err)
	assert.Equal(t, "ok", string(body))
}

func TestFetchFromProxy_BaseURLTrailingSlash(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/test", r.URL.Path)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL + "/",
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.NoError(t, err)
	assert.Equal(t, "ok", string(body))
}

func TestFetchFromProxy_NonOKStatus(t *testing.T) {
	tests := []struct {
		name          string
		statusCode    int
		responseBody  string
		expectedError string
	}{
		{
			name:          "401 Unauthorized",
			statusCode:    http.StatusUnauthorized,
			responseBody:  "Unauthorized",
			expectedError: "proxy returned status 401",
		},
		{
			name:          "403 Forbidden",
			statusCode:    http.StatusForbidden,
			responseBody:  "Forbidden",
			expectedError: "proxy returned status 403",
		},
		{
			name:          "404 Not Found",
			statusCode:    http.StatusNotFound,
			responseBody:  "Not Found",
			expectedError: "proxy returned status 404",
		},
		{
			name:          "500 Internal Server Error",
			statusCode:    http.StatusInternalServerError,
			responseBody:  "Internal Server Error",
			expectedError: "proxy returned status 500",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(tt.responseBody))
			}))
			defer server.Close()

			cred := &config.CredentialConfig{
				Name:    "test_cred",
				BaseURL: server.URL,
				APIKey:  "key",
			}

			logger := createTestLogger()
			ctx := context.Background()

			body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

			assert.Error(t, err)
			assert.Nil(t, body)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestFetchFromProxy_LongResponseBody(t *testing.T) {
	longBody := bytes.Repeat([]byte("x"), 300)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(longBody)
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.Error(t, err)
	assert.Nil(t, body)
}

func TestFetchFromProxy_Timeout(t *testing.T) {
	slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer slowServer.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: slowServer.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.Error(t, err)
	assert.Nil(t, body)
	// Error can be either "context deadline exceeded" (rate limiter timeout)
	// or "failed to fetch" (HTTP request timeout)
	errMsg := err.Error()
	isTimeoutError := strings.Contains(errMsg, "context deadline exceeded") ||
		strings.Contains(errMsg, "failed to fetch")
	assert.True(t, isTimeoutError, "error should contain timeout-related message, got: %s", errMsg)
}

func TestFetchFromProxy_ContextAlreadyHasDeadline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.NoError(t, err)
	assert.Equal(t, "ok", string(body))
}

func TestFetchFromProxy_InvalidURL(t *testing.T) {
	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: "http://[invalid:url",
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.Error(t, err)
	assert.Nil(t, body)
	assert.Contains(t, err.Error(), "failed to create request")
}

func TestFetchJSONFromProxy_Success(t *testing.T) {
	type TestResponse struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}

	expectedData := TestResponse{ID: "123", Name: "test"}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		encoder := json.NewEncoder(w)
		_ = encoder.Encode(expectedData)
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	var result TestResponse
	err := FetchJSONFromProxy(ctx, cred, "/api/data", logger, &result)

	assert.NoError(t, err)
	assert.Equal(t, expectedData.ID, result.ID)
	assert.Equal(t, expectedData.Name, result.Name)
}

func TestFetchJSONFromProxy_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("invalid json {"))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	var result map[string]interface{}
	err := FetchJSONFromProxy(ctx, cred, "/api/data", logger, &result)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse JSON")
}

func TestFetchJSONFromProxy_FetchError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("error"))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	var result map[string]interface{}
	err := FetchJSONFromProxy(ctx, cred, "/api/data", logger, &result)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "proxy returned status")
}

func TestFetchJSONFromProxy_ComplexJSON(t *testing.T) {
	type NestedData struct {
		Values []int `json:"values"`
	}

	type ComplexResponse struct {
		ID     string     `json:"id"`
		Nested NestedData `json:"nested"`
	}

	expectedData := ComplexResponse{
		ID:     "test-id",
		Nested: NestedData{Values: []int{1, 2, 3}},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		encoder := json.NewEncoder(w)
		_ = encoder.Encode(expectedData)
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	var result ComplexResponse
	err := FetchJSONFromProxy(ctx, cred, "/api/complex", logger, &result)

	assert.NoError(t, err)
	assert.Equal(t, expectedData.ID, result.ID)
	assert.Equal(t, expectedData.Nested.Values, result.Nested.Values)
}

func TestMin(t *testing.T) {
	tests := []struct {
		name     string
		a        int
		b        int
		expected int
	}{
		{"a is smaller", 5, 10, 5},
		{"b is smaller", 10, 5, 5},
		{"equal values", 5, 5, 5},
		{"both zero", 0, 0, 0},
		{"negative values", -5, -10, -10},
		{"mixed signs", -5, 10, -5},
		{"large numbers", 1000000, 999999, 999999},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := min(tt.a, tt.b)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestFetchFromProxy_ResponseBodyClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("test"))
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.NoError(t, err)
	assert.Equal(t, "test", string(body))
}

func TestFetchFromProxy_LargeResponse(t *testing.T) {
	largeBody := bytes.Repeat([]byte("x"), 10*1024)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(largeBody)
	}))
	defer server.Close()

	cred := &config.CredentialConfig{
		Name:    "test_cred",
		BaseURL: server.URL,
		APIKey:  "key",
	}

	logger := createTestLogger()
	ctx := context.Background()

	body, err := FetchFromProxy(ctx, cred, "/api/test", logger)

	assert.NoError(t, err)
	assert.Equal(t, len(largeBody), len(body))
	assert.Equal(t, largeBody, body)
}
