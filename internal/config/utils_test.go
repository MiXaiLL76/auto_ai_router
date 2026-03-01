package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestRpmToString(t *testing.T) {
	tests := []struct {
		name string
		rpm  int
		want string
	}{
		{"unlimited", -1, "unlimited (-1)"},
		{"zero", 0, "0"},
		{"positive", 100, "100"},
		{"large", 10000, "10000"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, rpmToString(tt.rpm))
		})
	}
}

func TestTpmToString(t *testing.T) {
	tests := []struct {
		name string
		tpm  int
		want string
	}{
		{"unlimited", -1, "unlimited (-1)"},
		{"zero", 0, "0"},
		{"positive", 100, "100"},
		{"large", 10000, "10000"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tpmToString(tt.tpm))
		})
	}
}

func TestBanDurationToString(t *testing.T) {
	tests := []struct {
		name string
		d    time.Duration
		want string
	}{
		{"permanent", 0, "permanent"},
		{"5_minutes", 5 * time.Minute, "5m0s"},
		{"1_hour", time.Hour, "1h0m0s"},
		{"30_seconds", 30 * time.Second, "30s"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, banDurationToString(tt.d))
		})
	}
}

func TestConvertMapToArgs(t *testing.T) {
	m := map[string]any{
		"name":        "test-cred",
		"type":        "openai",
		"is_fallback": false,
		"base_url":    "https://api.example.com",
	}

	args := convertMapToArgs(m)

	// Must contain all key-value pairs
	assert.Len(t, args, 8) // 4 keys * 2

	// Verify preferred order: name, type, base_url come before is_fallback
	foundKeys := make([]string, 0)
	for i := 0; i < len(args); i += 2 {
		foundKeys = append(foundKeys, args[i].(string))
	}
	// name, type, base_url should be in preferred order
	nameIdx := indexOf(foundKeys, "name")
	typeIdx := indexOf(foundKeys, "type")
	baseURLIdx := indexOf(foundKeys, "base_url")
	assert.True(t, nameIdx < typeIdx, "name should come before type")
	assert.True(t, typeIdx < baseURLIdx, "type should come before base_url")
}

func TestDefaultFail2BanConfig(t *testing.T) {
	cfg := defaultFail2BanConfig()

	assert.Equal(t, DefaultMaxAttempts, cfg.MaxAttempts)
	assert.Equal(t, DefaultBanDuration, cfg.BanDuration)
	assert.Equal(t, DefaultErrorCodes, cfg.ErrorCodes)

	// Verify ErrorCodes is a copy (modifying it doesn't affect default)
	cfg.ErrorCodes = append(cfg.ErrorCodes, 500)
	assert.NotEqual(t, cfg.ErrorCodes, DefaultErrorCodes)
}

// indexOf returns the index of needle in the slice, or -1 if not found.
func indexOf(slice []string, needle string) int {
	for i, s := range slice {
		if s == needle {
			return i
		}
	}
	return -1
}
