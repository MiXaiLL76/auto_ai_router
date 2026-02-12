package spendlog

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregateEndUserKey represents unique end user spend log grouping dimension
type aggregateEndUserKey struct {
	endUserID             string
	date                  string
	apiKey                string
	model                 string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

// aggregateDailyEndUserSpendLogs aggregates spend logs into DailyEndUserSpend
//
// This function:
// 1. Fetches spend logs from SpendLogs table filtered by requestIDs
// 2. Groups them by (end_user_id, date, api_key, model, provider, mcp_tool, endpoint)
// 3. Sums tokens, spend, and request counts per group
// 4. UPSERTs aggregated data into DailyEndUserSpend table
//
// Returns nil if successful (including "no logs to aggregate" case).
// Returns error on any database operation failure.
func aggregateDailyEndUserSpendLogs(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	requestIDs []string,
) error {
	records, err := loadUnprocessedSpendLogRecords(ctx, conn, logger, "EndUser", requestIDs)
	if err != nil {
		return err
	}

	// Map to aggregate by unique key
	aggregations := make(map[aggregateEndUserKey]*aggregationValue)
	totalRows := 0
	skippedRows := 0

	for _, record := range records {
		totalRows++

		// Skip if no end_user
		if record.EndUser == "" {
			skippedRows++
			continue
		}

		key := aggregateEndUserKey{
			endUserID:             record.EndUser,
			date:                  record.Date,
			apiKey:                record.APIKey,
			model:                 record.Model,
			customLLMProvider:     record.CustomLLMProvider,
			mcpNamespacedToolName: record.MCPNamespacedTool,
			endpoint:              record.Endpoint,
		}

		if aggregations[key] == nil {
			aggregations[key] = &aggregationValue{}
		}

		agg := aggregations[key]
		agg.promptTokens += int64(record.PromptTokens)
		agg.completionTokens += int64(record.CompletionTokens)
		agg.spend += record.Spend
		agg.apiRequests++

		if record.Status == "success" {
			agg.successfulRequests++
		} else {
			agg.failedRequests++
		}
	}

	logger.Debug("[DB] EndUser aggregation: scan complete",
		"total_rows", totalRows,
		"skipped_rows", skippedRows,
		"aggregation_groups", len(aggregations),
	)

	if len(aggregations) == 0 {
		return nil
	}

	// Insert aggregated data into DailyEndUserSpend
	for key, value := range aggregations {
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyEndUserSpend,
			key.endUserID, key.date, key.apiKey, key.model,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens, value.spend,
			value.apiRequests, value.successfulRequests, value.failedRequests,
		)

		if err != nil {
			logger.Error("[DB] EndUser aggregation: failed to upsert daily spend", "error", err, "key", key)
			return err
		}
	}

	logger.Debug("[DB] EndUser aggregation completed",
		"aggregations", len(aggregations),
	)

	return nil
}
