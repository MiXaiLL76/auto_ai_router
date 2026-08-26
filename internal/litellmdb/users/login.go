package users

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/crypto/scrypt"
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

	if user.UserID == nil || user.Password == nil {
		return nil, ErrInvalidCredentials
	}

	userEmail := ""
	if user.UserEmail != nil {
		userEmail = *user.UserEmail
	}
	userRole := ""
	if user.UserRole != nil {
		userRole = *user.UserRole
	}

	if !checkPassword(req.Password, *user.Password) {
		return nil, ErrInvalidCredentials
	}

	// Generate JWT for DB user
	now := time.Now()
	claims := &SessionClaims{
		UserID:    *user.UserID,
		UserRole:  userRole,
		UserEmail: userEmail,
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
		UserID:    *user.UserID,
		Key:       jwt,
		UserEmail: userEmail,
		UserRole:  userRole,
	}, nil
}

// scryptPrefix marks a LiteLLM_UserTable.password value produced by litellm's
// hash_password() (litellm/proxy/utils.py): "scrypt:" + base64(salt(16) || dk(32)),
// derived with N=16384, r=8, p=1. Litellm rehashes legacy SHA256 passwords to
// this format on next successful login (see _rehash_password_if_needed in
// litellm/proxy/auth/login_utils.py), so both formats must stay supported here.
const scryptPrefix = "scrypt:"

const (
	scryptN       = 16384
	scryptR       = 8
	scryptP       = 1
	scryptSaltLen = 16
	scryptKeyLen  = 32
)

// checkPassword compares the provided password against the stored password.
// Supports plain text, SHA256 hex hash, and litellm's scrypt format.
func checkPassword(password, stored string) bool {
	if encoded, ok := strings.CutPrefix(stored, scryptPrefix); ok {
		return checkScryptPassword(password, encoded)
	}

	// Direct comparison
	if constantTimeEqual(password, stored) {
		return true
	}

	// SHA256 hash comparison
	hash := sha256.Sum256([]byte(password))
	hashHex := hex.EncodeToString(hash[:])
	return constantTimeEqual(hashHex, stored)
}

// checkScryptPassword verifies password against a litellm-format scrypt hash
// (the part after the "scrypt:" prefix).
func checkScryptPassword(password, encoded string) bool {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || len(raw) != scryptSaltLen+scryptKeyLen {
		return false
	}
	salt, want := raw[:scryptSaltLen], raw[scryptSaltLen:]

	got, err := scrypt.Key([]byte(password), salt, scryptN, scryptR, scryptP, scryptKeyLen)
	if err != nil {
		return false
	}
	return subtle.ConstantTimeCompare(got, want) == 1
}

// constantTimeEqual performs a constant-time string comparison.
func constantTimeEqual(a, b string) bool {
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}
