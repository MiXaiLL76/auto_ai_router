// Package security provides security utilities for the application
package security

import (
	"strings"
)

// MaskSecret masks sensitive strings for logging.
// Shows first N characters followed by "..." to minimize secret exposure.
// Returns "***" for very short secrets (≤ prefixLen).
//
// Examples:
//
//	MaskSecret("sk_test_abc123", 4) -> "sk_t..."
//	MaskSecret("short", 4) -> "***"
//	MaskSecret("", 4) -> ""
func MaskSecret(secret string, prefixLen int) string {
	if secret == "" {
		return ""
	}
	if len(secret) <= prefixLen {
		return "***"
	}
	return secret[:prefixLen] + "..."
}

// MaskAPIKey masks API keys (shows first 4 characters).
// Convenience wrapper for MaskSecret with prefixLen=4.
//
// Example:
//
//	MaskAPIKey("sk_test_abc123") -> "sk_t..."
func MaskAPIKey(key string) string {
	return MaskSecret(key, 4)
}

// MaskToken masks tokens (shows first 4 characters).
// Alias for MaskAPIKey for semantic clarity.
// Operates on already-hashed tokens (SHA256), not raw tokens.
//
// Example:
//
//	MaskToken("f3d29bbcc0d020bb5875a9097827edea") -> "f3d2..."
func MaskToken(token string) string {
	return MaskAPIKey(token)
}

// MaskDatabaseURL masks password in PostgreSQL connection strings.
// Format: postgresql://user:password@host:port/db
// Returns: postgresql://user:***@host:port/db
//
// Example:
//
//	MaskDatabaseURL("postgresql://admin:secret123@localhost:5432/mydb") ->
//	"postgresql://admin:***@localhost:5432/mydb"
func MaskDatabaseURL(dbURL string) string {
	// Find the @ sign to locate where password ends
	atIdx := strings.Index(dbURL, "@")
	if atIdx == -1 {
		return dbURL // No @ sign, no password to mask
	}

	// Find the scheme end (://)
	schemeEnd := strings.Index(dbURL, "://")
	if schemeEnd == -1 {
		return dbURL // Invalid URL format
	}

	// Extract user:password part
	userPass := dbURL[schemeEnd+3 : atIdx]
	colonIdx := strings.Index(userPass, ":")
	if colonIdx == -1 {
		return dbURL // No password (no colon in user:pass part)
	}

	// Extract username
	user := userPass[:colonIdx]
	// Reconstruct with masked password
	return dbURL[:schemeEnd+3] + user + ":***" + dbURL[atIdx:]
}
