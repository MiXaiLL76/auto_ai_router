package users

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

var (
	ErrInvalidToken   = errors.New("invalid token")
	ErrTokenExpired   = errors.New("token expired")
	ErrInvalidSegment = errors.New("invalid token segment")
)

// SessionClaims holds the JWT payload for session tokens.
type SessionClaims struct {
	UserID    string `json:"user_id"`
	UserRole  string `json:"user_role"`
	UserEmail string `json:"user_email,omitempty"`
	Key       string `json:"key"`
	Exp       int64  `json:"exp"`
	Iat       int64  `json:"iat"`
}

// jwtHeader is the fixed JWT header for HMAC-SHA256.
var jwtHeaderB64 = base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))

// GenerateSessionJWT creates a JWT signed with HMAC-SHA256 using the master key.
func GenerateSessionJWT(claims *SessionClaims, masterKey string) (string, error) {
	payloadJSON, err := json.Marshal(claims)
	if err != nil {
		return "", fmt.Errorf("marshal claims: %w", err)
	}

	payloadB64 := base64.RawURLEncoding.EncodeToString(payloadJSON)
	signingInput := jwtHeaderB64 + "." + payloadB64

	sig := hmacSHA256([]byte(signingInput), []byte(masterKey))
	sigB64 := base64.RawURLEncoding.EncodeToString(sig)

	return signingInput + "." + sigB64, nil
}

// ValidateSessionJWT verifies a JWT signature and checks expiry.
// Returns the claims if valid, or an error.
func ValidateSessionJWT(token, masterKey string) (*SessionClaims, error) {
	// Split into 3 parts: header.payload.signature
	parts := splitToken(token)
	if parts == nil {
		return nil, ErrInvalidToken
	}

	// Verify signature
	signingInput := parts[0] + "." + parts[1]
	expectedSig := hmacSHA256([]byte(signingInput), []byte(masterKey))

	actualSig, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, ErrInvalidSegment
	}

	if !hmac.Equal(expectedSig, actualSig) {
		return nil, ErrInvalidToken
	}

	// Decode payload
	payloadJSON, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, ErrInvalidSegment
	}

	var claims SessionClaims
	if err := json.Unmarshal(payloadJSON, &claims); err != nil {
		return nil, fmt.Errorf("unmarshal claims: %w", err)
	}

	// Check expiry
	if claims.Exp > 0 && time.Now().Unix() > claims.Exp {
		return nil, ErrTokenExpired
	}

	return &claims, nil
}

func hmacSHA256(data, key []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

// splitToken splits a JWT into exactly 3 parts by '.'.
// Returns nil if the token doesn't have exactly 3 parts.
func splitToken(token string) []string {
	var parts [3]string
	idx := 0
	start := 0
	for i := 0; i < len(token); i++ {
		if token[i] == '.' {
			if idx >= 2 {
				return nil // too many dots
			}
			parts[idx] = token[start:i]
			idx++
			start = i + 1
		}
	}
	if idx != 2 {
		return nil // not enough dots
	}
	parts[2] = token[start:]
	if parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return nil
	}
	return parts[:]
}
