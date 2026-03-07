package users

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

var (
	ErrInvalidCredentials = errors.New("invalid credentials")
)

// LoginRequest represents the JSON body of a login request.
type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// LoginResult holds the result of a successful authentication.
type LoginResult struct {
	UserID    string
	Key       string // master key for admin, JWT for DB users
	UserEmail string
	UserRole  string
}

// SessionJWTDuration is the default session JWT expiry.
const SessionJWTDuration = 24 * time.Hour

// AuthenticateUser validates credentials against admin config and DB users.
// Admin path: compares with UI_USERNAME/UI_PASSWORD env vars.
// DB user path: looks up user by email in LiteLLM_UserTable.
func AuthenticateUser(ctx context.Context, req LoginRequest, masterKey string, pool *pgxpool.Pool) (*LoginResult, error) {
	if req.Username == "" || req.Password == "" {
		return nil, ErrInvalidCredentials
	}

	// Admin path
	uiUsername := os.Getenv("UI_USERNAME")
	if uiUsername == "" {
		uiUsername = "admin"
	}
	uiPassword := os.Getenv("UI_PASSWORD")
	if uiPassword == "" {
		uiPassword = masterKey
	}

	if constantTimeEqual(req.Username, uiUsername) && constantTimeEqual(req.Password, uiPassword) {
		return &LoginResult{
			UserID:    uiUsername,
			Key:       masterKey,
			UserEmail: "",
			UserRole:  "proxy_admin",
		}, nil
	}

	// DB user path
	if pool == nil {
		return nil, ErrInvalidCredentials
	}

	user, err := FindUserByEmail(ctx, pool, req.Username)
	if err != nil {
		return nil, fmt.Errorf("find user: %w", err)
	}
	if user == nil {
		return nil, ErrInvalidCredentials
	}

	if !checkPassword(req.Password, user.Password) {
		return nil, ErrInvalidCredentials
	}

	// Generate JWT for DB user
	now := time.Now()
	claims := &SessionClaims{
		UserID:    user.UserID,
		UserRole:  user.UserRole,
		UserEmail: user.UserEmail,
		Exp:       now.Add(SessionJWTDuration).Unix(),
		Iat:       now.Unix(),
	}

	jwt, err := GenerateSessionJWT(claims, masterKey)
	if err != nil {
		return nil, fmt.Errorf("generate jwt: %w", err)
	}

	// Set key in claims for the session cookie
	claims.Key = jwt

	return &LoginResult{
		UserID:    user.UserID,
		Key:       jwt,
		UserEmail: user.UserEmail,
		UserRole:  user.UserRole,
	}, nil
}

// checkPassword compares the provided password against the stored password.
// Supports plain text and SHA256 hex hash comparison.
func checkPassword(password, stored string) bool {
	// Direct comparison
	if constantTimeEqual(password, stored) {
		return true
	}

	// SHA256 hash comparison
	hash := sha256.Sum256([]byte(password))
	hashHex := hex.EncodeToString(hash[:])
	return constantTimeEqual(hashHex, stored)
}

// constantTimeEqual performs a constant-time string comparison.
func constantTimeEqual(a, b string) bool {
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}
