package spendlog

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/security"
)

// schemaHasKeyBudgetWindows is cleared on the first undefined table or column
// of the budget window queries (a schema from before LiteLLM added
// LiteLLM_BudgetWindowSpend). The transaction is aborted by then; its retry
// runs without the window projection, as the 42703 fallbacks above do.
var schemaHasKeyBudgetWindows atomic.Bool

func init() {
	schemaHasKeyBudgetWindows.Store(true)
}

// isUndefinedTable returns true if err is PostgreSQL SQLSTATE 42P01 (undefined_table).
func isUndefinedTable(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42P01"
}

// keyBudgetWindow names one LiteLLM_BudgetWindowSpend row of a key.
type keyBudgetWindow struct {
	key      string
	duration string
}

// updateKeyBudgetWindows moves the batch's spend into the window rows of the
// keys it was spent by. It runs after executeSpendUpdates, so every key of
// entries is already row-locked by updateTokens: writers of one key run it
// one at a time, in the lock order the transaction already has.
//
// Each entry counts towards the window its StartTime falls in:
//   - a row on that window gets the entry's spend added;
//   - a missing row, or one on an earlier window, is set to the sum of the
//     key's accounted spend logs in the window, which already holds this
//     transaction's own rows;
//   - a row already on a later window is left alone (a late entry of an
//     earlier window; its spend stays in LiteLLM_SpendLogs).
//
// The windows are the key's budget_limits plus any row the key already has,
// so a limit taken off and put back within a window finds a row that kept
// counting.
func (sl *Logger) updateKeyBudgetWindows(ctx context.Context, tx pgx.Tx, entries []*models.SpendLogEntry) error {
	if !schemaHasKeyBudgetWindows.Load() {
		return nil
	}
	entriesByKey := make(map[string][]*models.SpendLogEntry)
	for _, entry := range entries {
		if entry != nil && entry.APIKey != "" {
			entriesByKey[entry.APIKey] = append(entriesByKey[entry.APIKey], entry)
		}
	}
	if len(entriesByKey) == 0 {
		return nil
	}
	keys := make([]string, 0, len(entriesByKey))
	for key := range entriesByKey {
		keys = append(keys, key)
	}
	slices.Sort(keys)

	durations, stored, err := sl.readKeyBudgetWindows(ctx, tx, keys)
	if err != nil {
		if isUndefinedTable(err) || isColumnNotExist(err) {
			schemaHasKeyBudgetWindows.Store(false)
		}
		return err
	}

	calendar := sl.config.BudgetCalendar()
	for _, key := range keys {
		for _, duration := range durations[key] {
			window := keyBudgetWindow{key: key, duration: duration}
			grid, _ := models.ParseBudgetDuration(duration)
			type windowSpend struct {
				start, end time.Time
				spend      float64
			}
			var spends []windowSpend
			for _, entry := range entriesByKey[key] {
				start, end := grid.Bounds(entry.StartTime, calendar)
				i := slices.IndexFunc(spends, func(s windowSpend) bool { return s.start.Equal(start) })
				if i < 0 {
					spends = append(spends, windowSpend{start: start, end: end})
					i = len(spends) - 1
				}
				spends[i].spend += entry.Spend
			}
			slices.SortFunc(spends, func(a, b windowSpend) int { return a.start.Compare(b.start) })

			current, hasRow := stored[window]
			for _, s := range spends {
				switch {
				case hasRow && current.Equal(s.start):
					if s.spend == 0 {
						continue
					}
					if _, err := tx.Exec(ctx, queries.QueryAddKeyBudgetWindowSpend, key, duration, s.start, s.spend); err != nil {
						return err
					}
				case hasRow && current.After(s.start):
					continue
				default:
					var total float64
					if err := tx.QueryRow(ctx, queries.QuerySumKeyWindowSpendLogs, key, s.start, s.end).Scan(&total); err != nil {
						return err
					}
					if _, err := tx.Exec(ctx, queries.QuerySetKeyBudgetWindowSpend, key, duration, s.start, total); err != nil {
						return err
					}
					current, hasRow = s.start, true
				}
			}
		}
	}
	return nil
}

// readKeyBudgetWindows returns, per key, the durations of its budget_limits
// and of its existing window rows that ParseBudgetDuration accepts, and the
// window each row is on. A budget_limits AIR cannot read keeps only the existing rows; auth
// refuses such a key anyway.
func (sl *Logger) readKeyBudgetWindows(
	ctx context.Context,
	tx pgx.Tx,
	keys []string,
) (map[string][]string, map[keyBudgetWindow]time.Time, error) {
	rows, err := tx.Query(ctx, queries.QuerySelectKeysBudgetWindows, keys)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()

	durations := make(map[string][]string)
	stored := make(map[keyBudgetWindow]time.Time)
	limitsRead := make(map[string]bool)
	addDuration := func(key, duration string) {
		if _, ok := models.ParseBudgetDuration(duration); ok && !slices.Contains(durations[key], duration) {
			durations[key] = append(durations[key], duration)
		}
	}
	for rows.Next() {
		var key string
		var rawLimits []byte
		var duration *string
		var start *time.Time
		if err := rows.Scan(&key, &rawLimits, &duration, &start); err != nil {
			return nil, nil, err
		}
		if !limitsRead[key] {
			limitsRead[key] = true
			limits, err := models.ParseBudgetLimits(rawLimits)
			if err != nil {
				sl.logger.Warn("budget_limits has unexpected structure, keeping only its existing window rows",
					"token_prefix", security.MaskToken(key),
					"error", err,
				)
			}
			for _, limit := range limits {
				addDuration(key, limit.Duration)
			}
		}
		if duration != nil && start != nil {
			addDuration(key, *duration)
			stored[keyBudgetWindow{key: key, duration: *duration}] = *start
		}
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	for key := range durations {
		slices.Sort(durations[key])
	}
	return durations, stored, nil
}
