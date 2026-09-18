package video

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMigrationConfigurationErrors(t *testing.T) {
	err := MigrateDatabase(t.Context(), "postgres://user:private-password@host:invalid/db")
	require.Error(t, err)
	require.NotContains(t, err.Error(), "private-password")
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	require.ErrorIs(t, MigrateDatabase(ctx, "postgres://localhost/db"), context.Canceled)
}
