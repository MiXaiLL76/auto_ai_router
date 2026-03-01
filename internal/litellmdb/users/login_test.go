package users

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCheckPassword(t *testing.T) {
	tests := []struct {
		name     string
		password string
		stored   string
		want     bool
	}{
		{
			name:     "plaintext_match",
			password: "my-secret",
			stored:   "my-secret",
			want:     true,
		},
		{
			name:     "sha256_hash_match",
			password: "my-secret",
			stored:   sha256Hex("my-secret"),
			want:     true,
		},
		{
			name:     "wrong_password",
			password: "wrong",
			stored:   "correct",
			want:     false,
		},
		{
			name:     "wrong_hash",
			password: "wrong",
			stored:   sha256Hex("correct"),
			want:     false,
		},
		{
			name:     "empty_password",
			password: "",
			stored:   "",
			want:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, checkPassword(tt.password, tt.stored))
		})
	}
}

func TestConstantTimeEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b string
		want bool
	}{
		{"equal", "hello", "hello", true},
		{"unequal", "hello", "world", false},
		{"empty_both", "", "", true},
		{"empty_one", "hello", "", false},
		{"different_length", "short", "longer", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, constantTimeEqual(tt.a, tt.b))
		})
	}
}

func TestAuthenticateUser_AdminPath(t *testing.T) {
	// Set env vars for admin auth
	t.Setenv("UI_USERNAME", "testadmin")
	t.Setenv("UI_PASSWORD", "testpass123")

	masterKey := "sk-master-key"

	t.Run("admin_login_success", func(t *testing.T) {
		req := LoginRequest{
			Username: "testadmin",
			Password: "testpass123",
		}
		result, err := AuthenticateUser(context.TODO(), req, masterKey, nil)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "testadmin", result.UserID)
		assert.Equal(t, masterKey, result.Key)
		assert.Equal(t, "proxy_admin", result.UserRole)
	})

	t.Run("admin_login_wrong_password", func(t *testing.T) {
		req := LoginRequest{
			Username: "testadmin",
			Password: "wrongpass",
		}
		// Without a DB pool, it should fail (no DB path, wrong admin creds)
		result, err := AuthenticateUser(context.TODO(), req, masterKey, nil)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("empty_credentials", func(t *testing.T) {
		req := LoginRequest{
			Username: "",
			Password: "",
		}
		result, err := AuthenticateUser(context.TODO(), req, masterKey, nil)
		assert.ErrorIs(t, err, ErrInvalidCredentials)
		assert.Nil(t, result)
	})
}

// sha256Hex computes SHA256 hex hash of a string.
func sha256Hex(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}
