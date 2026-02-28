package users

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// UserRow represents a user record from LiteLLM_UserTable.
type UserRow struct {
	UserID    string
	UserEmail string
	Password  string
	UserRole  string
}

// FindUserByEmail looks up a user by email (case-insensitive).
// Returns nil, nil if no user found.
func FindUserByEmail(ctx context.Context, pool *pgxpool.Pool, email string) (*UserRow, error) {
	const query = `SELECT user_id, user_email, password, user_role
		FROM "LiteLLM_UserTable"
		WHERE LOWER(user_email) = LOWER($1)
		LIMIT 1`

	var row UserRow
	err := pool.QueryRow(ctx, query, email).Scan(
		&row.UserID,
		&row.UserEmail,
		&row.Password,
		&row.UserRole,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("query user by email: %w", err)
	}

	return &row, nil
}
