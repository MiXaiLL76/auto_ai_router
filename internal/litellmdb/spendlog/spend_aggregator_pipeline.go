package spendlog

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// spendLogQuerier is implemented by pgx.Tx and keeps the raw-row lookup on
// the same transaction that inserted the rows.
type spendLogQuerier interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

// dailySpendExecer is implemented by pgx.Tx. Daily aggregators deliberately
// accept this narrow interface so they cannot escape to a pool connection and
// accidentally commit independently from the raw row and entity counters.
type dailySpendExecer interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
}

type spendLogRecord struct {
	UserID                   string
	Date                     string
	APIKey                   string
	Model                    string
	ModelGroup               string
	CustomLLMProvider        string
	MCPNamespacedTool        string
	Endpoint                 string
	PromptTokens             int
	CompletionTokens         int
	CacheReadInputTokens     int64
	CacheCreationInputTokens int64
	Spend                    float64
	Status                   string
	RequestID                string
	TeamID                   string
	OrganizationID           string
	EndUser                  string
	RequestTags              string
	SkipDaily                bool
}

var dailyEndpointByCallType = map[string]string{
	"acompletion":       "/chat/completions",
	"atext_completion":  "/completions",
	"aembedding":        "/embeddings",
	"aresponses":        "/responses",
	"aimage_generation": "/image/generations",
	"aimage_edit":       "/images/edits",
	"atranscription":    "/audio/transcriptions",
	"aspeech":           "/audio/speech",
	"amoderation":       "/moderations",
	"arerank":           "/rerank",
}

// dailyEndpoint mirrors LiteLLM ROUTE_ENDPOINT_MAPPING for AIR-supported
// spend routes. Unknown call types intentionally yield an empty/NULL-like
// endpoint rather than leaking api_base into the daily dimension.
func dailyEndpoint(callType string) string {
	return dailyEndpointByCallType[callType]
}

func loadUnprocessedSpendLogRecords(
	ctx context.Context,
	conn spendLogQuerier,
	logger *slog.Logger,
	scope string,
	requestIDs []string,
) ([]spendLogRecord, error) {
	rows, err := conn.Query(ctx, queries.QuerySelectUnprocessedSpendLogs, requestIDs)
	if err != nil {
		logger.Error("[DB] "+scope+" aggregation: failed to fetch spend logs", "error", err)
		return nil, err
	}
	defer rows.Close()

	records := make([]spendLogRecord, 0, len(requestIDs))
	for rows.Next() {
		var record spendLogRecord
		var userID *string
		var model, modelGroup, customLLMProvider, mcpNamespacedToolName, callType, aggregationCallType *string
		var status *string
		var teamID, organizationID, endUser, requestTags *string
		// The SELECT still returns agent_id (column list is reworked separately);
		// the value is consumed by Scan but no longer used for aggregation.
		var agentID *string

		err := rows.Scan(
			&userID,
			&record.Date,
			&record.APIKey,
			&model,
			&modelGroup,
			&customLLMProvider,
			&mcpNamespacedToolName,
			&callType,
			&aggregationCallType,
			&record.PromptTokens,
			&record.CompletionTokens,
			&record.CacheReadInputTokens,
			&record.CacheCreationInputTokens,
			&record.Spend,
			&status,
			&record.RequestID,
			&teamID,
			&organizationID,
			&endUser,
			&requestTags,
			&agentID,
		)
		if err != nil {
			logger.Error("[DB] "+scope+" aggregation: failed to scan row", "error", err)
			return nil, err
		}

		record.UserID = derefString(userID)
		record.Model = derefString(model)
		record.ModelGroup = derefString(modelGroup)
		record.CustomLLMProvider = derefString(customLLMProvider)
		record.MCPNamespacedTool = derefString(mcpNamespacedToolName)
		record.Status = derefString(status)
		rawCallType := derefString(callType)
		effectiveCallType := derefString(aggregationCallType)
		if effectiveCallType == "" {
			record.SkipDaily = true
		} else if _, knownEndpoint := dailyEndpointByCallType[effectiveCallType]; !knownEndpoint {
			// A call_type outside the endpoint map is a permanent property of
			// the row, not a transient fault: failing here would poison the
			// whole batch (retries → DLQ) and lose valid rows alongside it.
			// The raw spend row is already inserted in this transaction, so
			// only its daily aggregation is skipped.
			logger.Warn("[DB] "+scope+" aggregation: unknown call_type, skipping daily aggregation",
				"call_type", effectiveCallType, "request_id", record.RequestID)
			record.SkipDaily = true
		} else {
			if rawCallType == "" {
				if record.Status != "failure" {
					return nil, fmt.Errorf(
						"empty raw LiteLLM call_type with status %q for request %q",
						record.Status, record.RequestID,
					)
				}
				// No nonzero-spend rejection here: LiteLLM legitimately writes
				// failure rows with partial cost and usage (interrupted streams,
				// recovered_response_cost in proxy_track_cost_callback).
				// LiteLLM failure rows use the public model as their daily model and
				// leave provider and endpoint empty. Keep the richer raw
				// AIR dimensions intact while projecting the compatible daily key.
				publicModel := record.ModelGroup
				if publicModel == "" {
					publicModel = record.Model
				}
				record.Model = publicModel
				record.CustomLLMProvider = ""
				record.Endpoint = ""
				if effectiveCallType == "aresponses" {
					record.ModelGroup = ""
				}
			} else {
				record.Endpoint = dailyEndpoint(rawCallType)
			}
		}
		record.TeamID = derefString(teamID)
		record.OrganizationID = derefString(organizationID)
		record.EndUser = derefString(endUser)
		record.RequestTags = derefString(requestTags)

		records = append(records, record)
	}

	if err := rows.Err(); err != nil {
		logger.Error("[DB] "+scope+" aggregation: failed to iterate rows", "error", err)
		return nil, err
	}

	return records, nil
}

func derefString(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

// dailyLockOrdered is implemented by every daily aggregation key. lockOrder
// serializes the key fields so keys sort identically across AIR replicas.
type dailyLockOrdered interface {
	comparable
	lockOrder() string
}

// sortedDailyKeys returns daily upsert keys in a stable order so concurrent
// transactions acquire row locks consistently and queue instead of
// deadlocking — Go map iteration order is randomized. Same rationale as
// sortedSpendKeys in spend_updater.go.
func sortedDailyKeys[K dailyLockOrdered, V any](aggregations map[K]V) []K {
	keys := make([]K, 0, len(aggregations))
	for key := range aggregations {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b K) int {
		return strings.Compare(a.lockOrder(), b.lockOrder())
	})
	return keys
}

// runAggregators runs all five daily aggregators sequentially on the transaction
// that inserted the source rows. The first error aborts the pipeline because a
// PostgreSQL transaction is unusable after a statement error and must roll back.
func (sl *Logger) runAggregators(aggCtx context.Context, tx dailySpendExecer, scope string, records []spendLogRecord) error {
	dailyRecords := make([]spendLogRecord, 0, len(records))
	for _, record := range records {
		if !record.SkipDaily {
			dailyRecords = append(dailyRecords, record)
		}
	}
	aggregators := []struct {
		name string
		fn   func() error
	}{
		{"User", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyUserSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"Team", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyTeamSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"Organization", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyOrganizationSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"EndUser", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyEndUserSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"Tag", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyTagSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
	}

	for _, agg := range aggregators {
		if err := agg.fn(); err != nil {
			sl.logger.Error("[DB] "+scope+": aggregator failed", "aggregator", agg.name, "error", err)
			return fmt.Errorf("%s daily aggregation: %w", agg.name, err)
		}
	}
	return nil
}
