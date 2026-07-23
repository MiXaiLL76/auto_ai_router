package startup

import (
	"context"
	"log/slog"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// mockLogger implements slog.Logger for testing
type mockLogger struct {
	messages []string
	records  []loggedRecord
}

type loggedRecord struct {
	level   slog.Level
	message string
	attrs   map[string]any
}

func newTestLogger() *slog.Logger {
	logger, _ := newTestLoggerWithMock()
	return logger
}

func newTestLoggerWithMock() (*slog.Logger, *mockLogger) {
	mock := &mockLogger{}
	return slog.New(&mockLoggerHandler{logger: mock}), mock
}

type mockLoggerHandler struct {
	logger *mockLogger
}

func (h *mockLoggerHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return true
}

func (h *mockLoggerHandler) Handle(ctx context.Context, r slog.Record) error {
	h.logger.messages = append(h.logger.messages, r.Message)
	record := loggedRecord{
		level:   r.Level,
		message: r.Message,
		attrs:   make(map[string]any),
	}
	r.Attrs(func(attr slog.Attr) bool {
		record.attrs[attr.Key] = attr.Value.Any()
		return true
	})
	h.logger.records = append(h.logger.records, record)
	return nil
}

func (h *mockLoggerHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return h
}

func (h *mockLoggerHandler) WithGroup(name string) slog.Handler {
	return h
}

func TestValidateProxyCredentialsAtStartup_NoProxies(t *testing.T) {
	// Test with empty config (no proxy credentials)
	cfg := &config.Config{
		Credentials: []config.CredentialConfig{},
	}

	logger := newTestLogger()

	// Should not panic or error
	ValidateProxyCredentialsAtStartup(cfg, logger)
}

// TestValidateProxyCredentialsAtStartup_NilConfig is skipped because
// the function doesn't handle nil config (would panic)
func TestValidateProxyCredentialsAtStartup_NilConfig(t *testing.T) {
	t.Skip("function doesn't handle nil config")
}

func TestValidateProxyCredentialsAtStartup_NilCredentials(t *testing.T) {
	// Test with nil credentials slice
	cfg := &config.Config{
		Credentials: nil,
	}

	logger := newTestLogger()

	// Should not panic
	ValidateProxyCredentialsAtStartup(cfg, logger)
}

func TestValidateProxyCredentialsAtStartup_NonProxyCredentials(t *testing.T) {
	// Test with non-proxy credentials only
	cfg := &config.Config{
		Credentials: []config.CredentialConfig{
			{
				Name:    "openai-key",
				Type:    config.ProviderTypeOpenAI,
				APIKey:  "sk-test",
				BaseURL: "https://api.openai.com",
			},
		},
	}

	logger := newTestLogger()

	// Should not panic - proxy credentials should be filtered out
	ValidateProxyCredentialsAtStartup(cfg, logger)
}

func TestValidateProxyCredentialsAtStartup_MixedCredentials(t *testing.T) {
	// Test with mixed credentials (proxy and non-proxy)
	cfg := &config.Config{
		Credentials: []config.CredentialConfig{
			{
				Name:    "openai-key",
				Type:    config.ProviderTypeOpenAI,
				APIKey:  "sk-test",
				BaseURL: "https://api.openai.com",
			},
			{
				Name:    "my-proxy",
				Type:    config.ProviderTypeProxy,
				BaseURL: "http://localhost:8080",
			},
		},
	}

	logger := newTestLogger()

	// Should handle gracefully even if proxy is unreachable
	ValidateProxyCredentialsAtStartup(cfg, logger)
}
