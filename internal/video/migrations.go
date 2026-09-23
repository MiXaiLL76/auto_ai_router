package video

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"time"

	"github.com/golang-migrate/migrate/v4"
	migratepgx "github.com/golang-migrate/migrate/v4/database/pgx/v5"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/stdlib"
)

//go:embed migrations/*.sql
var migrationFiles embed.FS

const schemaVersion = 1

// MigrateDatabase applies the video schema before the server deployment.
func MigrateDatabase(ctx context.Context, databaseURL string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	config, err := pgx.ParseConfig(databaseURL)
	if err != nil {
		return errors.New("invalid migration database configuration")
	}
	config.ConnectTimeout = 10 * time.Second
	config.RuntimeParams["statement_timeout"] = "60000"
	config.RuntimeParams["lock_timeout"] = "30000"
	db := stdlib.OpenDB(*config)
	defer func() { _ = db.Close() }()
	if err = db.PingContext(ctx); err != nil {
		return fmt.Errorf("connect migration database: %w", err)
	}
	driver, err := migratepgx.WithInstance(db, &migratepgx.Config{
		MigrationsTable:  "air_schema_migrations",
		StatementTimeout: time.Minute,
	})
	if err != nil {
		return fmt.Errorf("initialize migration database: %w", err)
	}
	source, err := iofs.New(migrationFiles, "migrations")
	if err != nil {
		_ = driver.Close()
		return err
	}
	runner, err := migrate.NewWithInstance("iofs", source, "pgx5", driver)
	if err != nil {
		_ = source.Close()
		_ = driver.Close()
		return err
	}
	defer func() { _, _ = runner.Close() }()
	stop := context.AfterFunc(ctx, func() { runner.GracefulStop <- true })
	defer stop()
	runner.LockTimeout = 30 * time.Second
	if err = runner.Up(); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return fmt.Errorf("apply video migrations: %w", err)
	}
	return ctx.Err()
}

// CheckSchema checks the migration version without modifying the database.
func (s *PostgresStore) CheckSchema(ctx context.Context) error {
	if s == nil {
		return ErrInvalid
	}
	var version int
	var dirty bool
	if err := s.db.QueryRow(ctx, `SELECT version, dirty FROM air_schema_migrations`).Scan(&version, &dirty); err != nil {
		return fmt.Errorf("read video schema version: %w", err)
	}
	if dirty || version != schemaVersion {
		return fmt.Errorf("video schema requires version %d, found version %d with dirty=%t", schemaVersion, version, dirty)
	}
	return nil
}
