package litellmdb

// This file exports hash functions for external use

import "github.com/mixaill76/auto_ai_router/internal/litellmdb/auth"

// HashToken is exported for backwards compatibility
func HashToken(token string) string {
	return auth.HashToken(token)
}
