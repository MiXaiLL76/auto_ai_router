package security

import (
	"net/http"
	"testing"
)

func TestMaskSecret(t *testing.T) {
	tests := []struct {
		name      string
		secret    string
		prefixLen int
		want      string
	}{
		// Empty string
		{"empty", "", 4, ""},

		// Short secrets (≤ prefixLen)
		{"exact_length", "abcd", 4, "***"},
		{"shorter", "ab", 4, "***"},
		{"single_char", "a", 4, "***"},

		// Long secrets (> prefixLen)
		{"long_secret", "abcdefghij", 4, "abcd..."},
		{"api_key", "sk_test_abc123def456", 4, "sk_t..."},
		{"hash", "f3d29bbcc0d020bb5875a9097827edea", 4, "f3d2..."},

		// Different prefix lengths
		{"prefix_1", "abcdefghij", 1, "a..."},
		{"prefix_10", "abcdefghijklmnop", 10, "abcdefghij..."},

		// Edge cases
		{"exactly_plus_one", "abcde", 4, "abcd..."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaskSecret(tt.secret, tt.prefixLen)
			if got != tt.want {
				t.Errorf("MaskSecret(%q, %d) = %q, want %q", tt.secret, tt.prefixLen, got, tt.want)
			}
		})
	}
}

func TestMaskAPIKey(t *testing.T) {
	tests := []struct {
		name string
		key  string
		want string
	}{
		{"empty", "", ""},
		{"short", "abc", "***"},
		{"exact_length", "abcd", "***"},
		{"long_key", "sk_test_abc123def456", "sk_t..."},
		{"openai_key", "sk-proj-abc123def456ghi789jkl", "sk-p..."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaskAPIKey(tt.key)
			if got != tt.want {
				t.Errorf("MaskAPIKey(%q) = %q, want %q", tt.key, got, tt.want)
			}
		})
	}
}

func TestMaskToken(t *testing.T) {
	tests := []struct {
		name  string
		token string
		want  string
	}{
		{"empty", "", ""},
		{"short", "abc", "***"},
		{"hashed_token", "f3d29bbcc0d020bb5875a9097827edea", "f3d2..."},
		{"short_hash", "abcd", "***"},
		{"long_token", "sk_test_token_123456789", "sk_t..."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaskToken(tt.token)
			if got != tt.want {
				t.Errorf("MaskToken(%q) = %q, want %q", tt.token, got, tt.want)
			}
		})
	}
}

func TestMaskDatabaseURL(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		want    string
		wantErr bool
	}{
		{
			name: "postgres_with_password",
			url:  "postgresql://admin:secret123@localhost:5432/mydb",
			want: "postgresql://admin:***@localhost:5432/mydb",
		},
		{
			name: "postgres_without_password",
			url:  "postgresql://admin@localhost:5432/mydb",
			want: "postgresql://admin@localhost:5432/mydb",
		},
		{
			name: "postgres_no_user_info",
			url:  "postgresql://localhost:5432/mydb",
			want: "postgresql://localhost:5432/mydb",
		},
		{
			name: "postgres_with_special_chars_in_password",
			url:  "postgresql://user:p!@ssw0rd@host:5432/db",
			want: "postgresql://user:***@ssw0rd@host:5432/db",
		},
		{
			name: "no_scheme",
			url:  "not a url at all",
			want: "not a url at all",
		},
		{
			name: "mysql_with_password",
			url:  "mysql://root:mypassword@localhost:3306/database",
			want: "mysql://root:***@localhost:3306/database",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaskDatabaseURL(tt.url)
			if got != tt.want {
				t.Errorf("MaskDatabaseURL(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}

func TestMaskSensitiveHeaders(t *testing.T) {
	tests := []struct {
		name    string
		input   http.Header
		checks  map[string]string // header -> expected masked value
		verifyV bool              // verify that value starts with expected prefix
	}{
		{
			name: "bearer_token",
			input: http.Header{
				"Authorization": {"Bearer sk_test_abc123def456"},
			},
			checks: map[string]string{
				"Authorization": "Bearer sk_t...",
			},
		},
		{
			name: "api_key_header",
			input: http.Header{
				"X-API-Key": {"sk_test_abc123def456"},
			},
			checks: map[string]string{
				"X-API-Key": "sk_t...",
			},
		},
		{
			name: "non_bearer_auth",
			input: http.Header{
				"Authorization": {"Basic dXNlcjpwYXNz"},
			},
			checks: map[string]string{
				"Authorization": "Basi...",
			},
		},
		{
			name: "cookie_header",
			input: http.Header{
				"Cookie": {"session=abc123; user=john"},
			},
			checks: map[string]string{
				"Cookie": "***cookie***",
			},
		},
		{
			name: "mixed_headers",
			input: http.Header{
				"Authorization": {"Bearer token123456789"},
				"Content-Type":  {"application/json"},
				"X-API-Key":     {"key123456789"},
				"User-Agent":    {"test-client"},
			},
			checks: map[string]string{
				"Authorization": "Bearer toke...",
				"Content-Type":  "application/json", // Not masked
				"X-API-Key":     "key1...",
				"User-Agent":    "test-client", // Not masked
			},
		},
		{
			name:   "empty_headers",
			input:  http.Header{},
			checks: map[string]string{},
		},
		{
			name: "proxy_authorization",
			input: http.Header{
				"Proxy-Authorization": {"Basic cHJveHl1c2VyOnByb3h5cGFzcw=="},
			},
			checks: map[string]string{
				"Proxy-Authorization": "Basi...",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaskSensitiveHeaders(tt.input)

			// Check all expected headers
			for header, expectedValue := range tt.checks {
				actual := got.Get(header)
				if actual != expectedValue {
					t.Errorf("MaskSensitiveHeaders: %s = %q, want %q", header, actual, expectedValue)
				}
			}

			// Check that no unexpected headers are present
			for header := range got {
				if _, ok := tt.checks[header]; !ok {
					if tt.input.Get(header) != "" {
						t.Errorf("MaskSensitiveHeaders: unexpected header %s = %q", header, got.Get(header))
					}
				}
			}
		})
	}
}
