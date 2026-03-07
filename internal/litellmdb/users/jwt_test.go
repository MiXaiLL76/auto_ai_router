package users

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateAndValidateSessionJWT(t *testing.T) {
	masterKey := "test-secret-key-123"
	now := time.Now()

	claims := &SessionClaims{
		UserID:    "user-123",
		UserRole:  "proxy_admin",
		UserEmail: "admin@example.com",
		Key:       "sk-test",
		Exp:       now.Add(time.Hour).Unix(),
		Iat:       now.Unix(),
	}

	// Generate
	token, err := GenerateSessionJWT(claims, masterKey)
	require.NoError(t, err)
	assert.NotEmpty(t, token)

	// Validate
	decoded, err := ValidateSessionJWT(token, masterKey)
	require.NoError(t, err)
	require.NotNil(t, decoded)

	// Claims should match
	assert.Equal(t, claims.UserID, decoded.UserID)
	assert.Equal(t, claims.UserRole, decoded.UserRole)
	assert.Equal(t, claims.UserEmail, decoded.UserEmail)
	assert.Equal(t, claims.Key, decoded.Key)
	assert.Equal(t, claims.Exp, decoded.Exp)
	assert.Equal(t, claims.Iat, decoded.Iat)
}

func TestValidateSessionJWT_InvalidSignature(t *testing.T) {
	masterKey := "correct-key"
	claims := &SessionClaims{
		UserID:   "user-1",
		UserRole: "admin",
		Exp:      time.Now().Add(time.Hour).Unix(),
		Iat:      time.Now().Unix(),
	}

	token, err := GenerateSessionJWT(claims, masterKey)
	require.NoError(t, err)

	// Validate with wrong key
	_, err = ValidateSessionJWT(token, "wrong-key")
	assert.ErrorIs(t, err, ErrInvalidToken)
}

func TestValidateSessionJWT_Expired(t *testing.T) {
	masterKey := "test-key"
	claims := &SessionClaims{
		UserID:   "user-1",
		UserRole: "admin",
		Exp:      time.Now().Add(-time.Hour).Unix(), // expired 1 hour ago
		Iat:      time.Now().Add(-2 * time.Hour).Unix(),
	}

	token, err := GenerateSessionJWT(claims, masterKey)
	require.NoError(t, err)

	_, err = ValidateSessionJWT(token, masterKey)
	assert.ErrorIs(t, err, ErrTokenExpired)
}

func TestSplitToken(t *testing.T) {
	tests := []struct {
		name  string
		token string
		valid bool
	}{
		{"valid_3_parts", "header.payload.signature", true},
		{"too_few_dots", "header.payload", false},
		{"no_dots", "headeronly", false},
		{"too_many_dots", "a.b.c.d", false},
		{"empty_part", "header..signature", false},
		{"empty_string", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := splitToken(tt.token)
			if tt.valid {
				require.NotNil(t, result)
				assert.Len(t, result, 3)
			} else {
				assert.Nil(t, result)
			}
		})
	}
}
