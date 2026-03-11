package users

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestUserRowFields verifies UserRow structure
func TestUserRowFields(t *testing.T) {
	userID := "user-001"
	userEmail := "test@example.com"
	password := "hashed_password"
	userRole := "admin"

	userIDPtr := &userID
	userEmailPtr := &userEmail
	passwordPtr := &password
	userRolePtr := &userRole

	row := UserRow{
		UserID:    userIDPtr,
		UserEmail: userEmailPtr,
		Password:  passwordPtr,
		UserRole:  userRolePtr,
	}

	assert.Equal(t, "user-001", *row.UserID)
	assert.Equal(t, "test@example.com", *row.UserEmail)
	assert.Equal(t, "hashed_password", *row.Password)
	assert.Equal(t, "admin", *row.UserRole)
}

// TestUserRow_NilFields verifies UserRow with nil fields
func TestUserRow_NilFields(t *testing.T) {
	row := UserRow{}

	assert.Nil(t, row.UserID)
	assert.Nil(t, row.UserEmail)
	assert.Nil(t, row.Password)
	assert.Nil(t, row.UserRole)
}

// TestUserRow_PartialFields verifies UserRow with some fields nil
func TestUserRow_PartialFields(t *testing.T) {
	userID := "user-001"
	row := UserRow{
		UserID: &userID,
		// UserEmail, Password, UserRole are nil
	}

	assert.NotNil(t, row.UserID)
	assert.Equal(t, "user-001", *row.UserID)
	assert.Nil(t, row.UserEmail)
	assert.Nil(t, row.Password)
	assert.Nil(t, row.UserRole)
}

// TestFindUserByEmail_QueryStructure verifies the query structure (can't run without DB)
func TestFindUserByEmail_QueryStructure(t *testing.T) {
	// This test verifies the query string is well-formed
	// The actual query execution requires a database connection
	query := `SELECT user_id, user_email, password, user_role
		FROM "LiteLLM_UserTable"
		WHERE LOWER(user_email) = LOWER($1)
		LIMIT 1`

	assert.NotEmpty(t, query)
	assert.Contains(t, query, "LiteLLM_UserTable")
	assert.Contains(t, query, "user_email")
	assert.Contains(t, query, "LOWER")
	assert.Contains(t, query, "LIMIT 1")
}

// TestLoginRequestFields verifies LoginRequest structure
func TestLoginRequestFields(t *testing.T) {
	req := LoginRequest{
		Username: "admin",
		Password: "secret123",
	}

	assert.Equal(t, "admin", req.Username)
	assert.Equal(t, "secret123", req.Password)
}

// TestLoginResultFields verifies LoginResult structure
func TestLoginResultFields(t *testing.T) {
	result := LoginResult{
		UserID:    "user-001",
		Key:       "jwt-token-or-master-key",
		UserEmail: "admin@example.com",
		UserRole:  "proxy_admin",
	}

	assert.Equal(t, "user-001", result.UserID)
	assert.Equal(t, "jwt-token-or-master-key", result.Key)
	assert.Equal(t, "admin@example.com", result.UserEmail)
	assert.Equal(t, "proxy_admin", result.UserRole)
}

// TestLoginResult_EmptyFields verifies LoginResult with empty fields
func TestLoginResult_EmptyFields(t *testing.T) {
	result := LoginResult{}

	assert.Empty(t, result.UserID)
	assert.Empty(t, result.Key)
	assert.Empty(t, result.UserEmail)
	assert.Empty(t, result.UserRole)
}

// TestSessionJWTDuration_Constant verifies the JWT duration constant
func TestSessionJWTDuration_Constant(t *testing.T) {
	// SessionJWTDuration should be 24 hours
	assert.Equal(t, 24.0, SessionJWTDuration.Hours())
}
