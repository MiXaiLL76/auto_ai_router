package spendlog

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"

	"github.com/jackc/pgx/v5"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

// requestIDGroup contains all logical effects that prefer the same
// provider-controlled response ID. The first effect gets the preferred ID when
// it is free; every other distinct event falls back to its AIR event ID.
type requestIDGroup struct {
	preferredID    string
	representative *models.SpendLogEntry
	entries        []*models.SpendLogEntry
}

// insertSpendRowsCollisionSafe inserts each logical effect exactly once while
// preserving LiteLLM's normal request_id shape. Ownership of a conflicting
// provider-controlled ID is resolved by a targeted claim UPDATE in a separate
// statement after the INSERT: INSERT ... ON CONFLICT may wait for a concurrent
// transaction that is invisible to the INSERT statement snapshot, while the
// next READ COMMITTED statement sees the committed winner. No SpendLogs row is
// ever read back.
//
// AirEventID must be a value that is always distinct per logical HTTP
// request, independent of RequestID (which, when otel trusts an inbound
// traceparent, can be shared across distinct requests) — see
// RequestLogContext.EventID. Without it, a conflict on a row owned by another
// transaction cannot be classified as replay vs genuine collision; the entry
// is skipped but the drop is logged and counted in
// auto_ai_router_spend_collision_unresolved_total instead of staying silent.
func insertSpendRowsCollisionSafe(ctx context.Context, tx pgx.Tx, batch []*models.SpendLogEntry, logger *slog.Logger) ([]string, error) {
	groups := groupEntriesByPreferredRequestID(batch)
	if len(groups) == 0 {
		return nil, nil
	}

	representatives := make([]*models.SpendLogEntry, 0, len(groups))
	for _, group := range groups {
		representatives = append(representatives, cloneEntryWithRequestID(group.representative, group.preferredID))
	}
	preferredInserted, err := insertSpendRowsReturningIDs(ctx, tx, representatives)
	if err != nil {
		return nil, fmt.Errorf("batch insert preferred request IDs: %w", err)
	}
	preferredInsertedSet := stringSet(preferredInserted)

	fallbacks := make([]*models.SpendLogEntry, 0)
	fallbackSeen := make(map[string]struct{})
	for _, group := range groups {
		if _, inserted := preferredInsertedSet[group.preferredID]; inserted {
			// This transaction owns the preferred row, so the owner is known
			// in memory: only the other distinct events need a fallback row.
			ownerEventID := group.representative.AirEventID
			for _, entry := range group.entries {
				eventID := entry.AirEventID
				if eventID == "" || eventID == ownerEventID {
					continue
				}
				if _, duplicateEvent := fallbackSeen[eventID]; duplicateEvent {
					continue
				}
				fallbackSeen[eventID] = struct{}{}
				fallbacks = append(fallbacks, cloneEntryWithRequestID(entry, eventID))
			}
			continue
		}

		// The preferred row belongs to another transaction. A targeted claim
		// per event distinguishes a replay of our own event (claim succeeds,
		// nothing more to write) from a genuine collision (fall back to the
		// AIR event ID row).
		for _, entry := range group.entries {
			eventID := entry.AirEventID
			if eventID == "" {
				// Without an AIR event ID there is no way to tell a benign
				// replay of our own earlier write from a genuine collision
				// with a different event. The row stays with the committed
				// winner, but the drop must be observable, not silent.
				logger.Warn("[DB] SpendLog row dropped: request_id owned by another transaction and no AIR event ID to resolve ownership",
					"request_id", group.preferredID,
				)
				monitoring.RecordSpendCollisionUnresolved(1)
				continue
			}
			if _, duplicateEvent := fallbackSeen[eventID]; duplicateEvent {
				continue
			}
			fallbackSeen[eventID] = struct{}{}
			claimed, err := claimSpendRowEventOwner(ctx, tx, group.preferredID, eventID)
			if err != nil {
				return nil, fmt.Errorf("claim conflicting request ID owner: %w", err)
			}
			if claimed {
				continue
			}
			fallbacks = append(fallbacks, cloneEntryWithRequestID(entry, eventID))
		}
	}
	sort.Slice(fallbacks, func(i, j int) bool {
		return fallbacks[i].RequestID < fallbacks[j].RequestID
	})

	fallbackInserted, err := insertSpendRowsReturningIDs(ctx, tx, fallbacks)
	if err != nil {
		return nil, fmt.Errorf("batch insert AIR event ID fallbacks: %w", err)
	}
	return append(preferredInserted, fallbackInserted...), nil
}

func groupEntriesByPreferredRequestID(batch []*models.SpendLogEntry) []*requestIDGroup {
	byID := make(map[string]*requestIDGroup, len(batch))
	groups := make([]*requestIDGroup, 0, len(batch))
	for _, entry := range batch {
		if entry == nil {
			continue
		}
		preferredID := entry.RequestID
		if preferredID == "" {
			preferredID = entry.AirEventID
		}
		group := byID[preferredID]
		if group == nil {
			group = &requestIDGroup{preferredID: preferredID, representative: entry}
			byID[preferredID] = group
			groups = append(groups, group)
		}
		group.entries = append(group.entries, entry)
	}
	// Every transaction acquires unique-index locks in the same request_id
	// order. Without this, concurrent batches [P,Q] and [Q,P] can deadlock even
	// though each individual INSERT uses ON CONFLICT DO NOTHING.
	sort.Slice(groups, func(i, j int) bool {
		return groups[i].preferredID < groups[j].preferredID
	})
	return groups
}

func cloneEntryWithRequestID(entry *models.SpendLogEntry, requestID string) *models.SpendLogEntry {
	clone := *entry
	clone.RequestID = requestID
	return &clone
}

func insertSpendRowsReturningIDs(ctx context.Context, tx pgx.Tx, entries []*models.SpendLogEntry) ([]string, error) {
	if len(entries) == 0 {
		return nil, nil
	}
	rows, err := tx.Query(ctx, queries.BuildBatchInsertQuery(len(entries)), GetBatchParams(entries)...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	insertedIDs := make([]string, 0, len(entries))
	for rows.Next() {
		var requestID string
		if err := rows.Scan(&requestID); err != nil {
			return nil, fmt.Errorf("scan returning request_id: %w", err)
		}
		insertedIDs = append(insertedIDs, requestID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate returning rows: %w", err)
	}
	return insertedIDs, nil
}

// claimSpendRowEventOwner reports whether the spend row with the given primary
// key already belongs to the AIR event. The targeted claim UPDATE only matches
// when the row's metadata attributes the row to this event, so a replay
// detects its own earlier write, while a genuine provider request_id collision
// (different owner event) misses and the caller adds the AIR event ID
// fallback row.
func claimSpendRowEventOwner(ctx context.Context, tx pgx.Tx, requestID, eventID string) (bool, error) {
	var claimedID string
	err := tx.QueryRow(ctx, queries.QueryClaimSpendLogEventOwner, requestID, eventID).Scan(&claimedID)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

func stringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}
