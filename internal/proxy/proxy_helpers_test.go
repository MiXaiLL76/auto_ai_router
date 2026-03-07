package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
)

// timeoutError is a mock net.Error that reports timeout.
type timeoutError struct{}

func (e *timeoutError) Error() string   { return "i/o timeout" }
func (e *timeoutError) Timeout() bool   { return true }
func (e *timeoutError) Temporary() bool { return false }

// nonTimeoutNetError is a mock net.Error that does not report timeout.
type nonTimeoutNetError struct{}

func (e *nonTimeoutNetError) Error() string   { return "connection refused" }
func (e *nonTimeoutNetError) Timeout() bool   { return false }
func (e *nonTimeoutNetError) Temporary() bool { return false }

func TestIsTimeoutError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"context_deadline_exceeded", context.DeadlineExceeded, true},
		{"net_timeout", &timeoutError{}, true},
		{"context_canceled", context.Canceled, false},
		{"generic_error", errors.New("something"), false},
		{"non_timeout_net_error", &nonTimeoutNetError{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isTimeoutError(tt.err))
		})
	}
}

func TestIsClientDisconnectError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"context_canceled", context.Canceled, true},
		{"epipe", syscall.EPIPE, true},
		{"econnreset", syscall.ECONNRESET, true},
		{"broken_pipe_msg", errors.New("write: broken pipe"), true},
		{"conn_reset_msg", errors.New("connection reset by peer"), true},
		{"generic_error", errors.New("something"), false},
		{"timeout", context.DeadlineExceeded, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isClientDisconnectError(tt.err))
		})
	}
}

func TestMapHTTPStatusToErrorClass(t *testing.T) {
	tests := []struct {
		name   string
		status int
		want   string
	}{
		{"400", http.StatusBadRequest, "BadRequestError"},
		{"401", http.StatusUnauthorized, "AuthenticationError"},
		{"403", http.StatusForbidden, "PermissionDeniedError"},
		{"404", http.StatusNotFound, "NotFoundError"},
		{"408", http.StatusRequestTimeout, "Timeout"},
		{"422", http.StatusUnprocessableEntity, "UnprocessableEntityError"},
		{"429", http.StatusTooManyRequests, "RateLimitError"},
		{"500", http.StatusInternalServerError, "InternalServerError"},
		{"503", http.StatusServiceUnavailable, "ServiceUnavailableError"},
		{"405_4xx_default", http.StatusMethodNotAllowed, "BadRequestError"},
		{"502_5xx_default", http.StatusBadGateway, "APIConnectionError"},
		{"200_other", http.StatusOK, "APIError"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, mapHTTPStatusToErrorClass(tt.status))
		})
	}
}

func TestExtractVersionSuffix(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		want    string
	}{
		{"v1_suffix", "https://api.example.com/v1", "/v1"},
		{"v4_suffix", "https://api.example.com/v4", "/v4"},
		{"no_version", "https://api.example.com", ""},
		{"not_version", "https://api.example.com/abc", ""},
		{"v_no_digits", "https://api.example.com/v", ""},
		{"v_with_chars", "https://api.example.com/vx1", ""},
		{"no_slash", "example", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractVersionSuffix(tt.baseURL))
		})
	}
}

func TestExtractVersionPrefix(t *testing.T) {
	tests := []struct {
		name    string
		urlPath string
		want    string
	}{
		{"v1_chat", "/v1/chat/completions", "/v1"},
		{"v4_models", "/v4/models", "/v4"},
		{"no_version", "/chat/completions", ""},
		{"too_short", "/v", ""},
		{"v_no_digits", "/va/chat", ""},
		{"empty", "", ""},
		{"just_v1", "/v1", "/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractVersionPrefix(tt.urlPath))
		})
	}
}

func TestGetClientIP(t *testing.T) {
	tests := []struct {
		name       string
		headers    map[string]string
		remoteAddr string
		want       string
	}{
		{
			name:       "x_forwarded_for_single",
			headers:    map[string]string{"X-Forwarded-For": "1.2.3.4"},
			remoteAddr: "5.6.7.8:1234",
			want:       "1.2.3.4",
		},
		{
			name:       "x_forwarded_for_multiple",
			headers:    map[string]string{"X-Forwarded-For": "1.2.3.4, 10.0.0.1"},
			remoteAddr: "5.6.7.8:1234",
			want:       "1.2.3.4",
		},
		{
			name:       "x_real_ip",
			headers:    map[string]string{"X-Real-IP": "9.8.7.6"},
			remoteAddr: "5.6.7.8:1234",
			want:       "9.8.7.6",
		},
		{
			name:       "remote_addr_with_port",
			headers:    map[string]string{},
			remoteAddr: "5.6.7.8:1234",
			want:       "5.6.7.8",
		},
		{
			name:       "remote_addr_no_port",
			headers:    map[string]string{},
			remoteAddr: "5.6.7.8",
			want:       "5.6.7.8",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "/", nil)
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}
			req.RemoteAddr = tt.remoteAddr
			assert.Equal(t, tt.want, getClientIP(req))
		})
	}
}

func TestBuildMetadata(t *testing.T) {
	t.Run("nil_tokenInfo", func(t *testing.T) {
		result := buildMetadata("hashed123", nil, "", 0)
		var m map[string]interface{}
		err := json.Unmarshal([]byte(result), &m)
		require.NoError(t, err)
		assert.Equal(t, "hashed123", m["user_api_key"])
		assert.Equal(t, "", m["user_api_key_user_id"])
		assert.Equal(t, "success", m["status"])
	})

	t.Run("with_tokenInfo_and_aliases", func(t *testing.T) {
		tokenInfo := &litellmdb.TokenInfo{
			UserID:         "user-1",
			TeamID:         "team-1",
			OrganizationID: "org-1",
			KeyAlias:       "my-key",
			UserAlias:      "my-user",
			TeamAlias:      "my-team",
		}
		result := buildMetadata("hashed456", tokenInfo, "", 0)
		var m map[string]interface{}
		err := json.Unmarshal([]byte(result), &m)
		require.NoError(t, err)
		assert.Equal(t, "user-1", m["user_api_key_user_id"])
		assert.Equal(t, "team-1", m["user_api_key_team_id"])
		assert.Equal(t, "org-1", m["user_api_key_org_id"])
		assert.Equal(t, "my-key", m["user_api_key_alias"])
		assert.Equal(t, "my-user", m["user_api_key_user_alias"])
		assert.Equal(t, "my-team", m["user_api_key_team_alias"])
		assert.Equal(t, "success", m["status"])
	})

	t.Run("with_error_info", func(t *testing.T) {
		result := buildMetadata("hashed789", nil, "rate limit exceeded", http.StatusTooManyRequests)
		var m map[string]interface{}
		err := json.Unmarshal([]byte(result), &m)
		require.NoError(t, err)
		assert.Equal(t, "failure", m["status"])

		errInfo, ok := m["error_information"].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "rate limit exceeded", errInfo["error_message"])
		assert.Equal(t, float64(429), errInfo["error_code"])
		assert.Equal(t, "RateLimitError", errInfo["error_class"])
	})
}

func TestExtractEndUser(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{
			name:    "header_present",
			headers: map[string]string{"X-End-User": "user@example.com"},
			want:    "user@example.com",
		},
		{
			name:    "header_absent",
			headers: map[string]string{},
			want:    "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "/", nil)
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}
			assert.Equal(t, tt.want, extractEndUser(req))
		})
	}
}

// Compile-time check that timeoutError implements net.Error
var _ net.Error = (*timeoutError)(nil)
var _ net.Error = (*nonTimeoutNetError)(nil)
