package spendlog

import (
	"context"
	"log/slog"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregateEndUserKey represents unique end user spend log grouping dimension
type aggregateEndUserKey struct {
	endUserID             string
	date                  string
	apiKey                string
	model                 string
	modelGroup            string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

func (k aggregateEndUserKey) lockOrder() string {
	return strings.Join([]string{k.endUserID, k.date, k.apiKey, k.model, k.modelGroup,
		k.customLLMProvider, k.mcpNamespacedToolName, k.endpoint}, "\x00")
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
	conn dailySpendExecer,
	logger *slog.Logger,
	records []spendLogRecord,
) error {
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
			modelGroup:            record.ModelGroup,
			customLLMProvider:     record.CustomLLMProvider,
			mcpNamespacedToolName: record.MCPNamespacedTool,
			endpoint:              record.Endpoint,
		}

		if aggregations[key] == nil {
			aggregations[key] = &aggregationValue{}
		}

		aggregations[key].addRecord(record)
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
	for _, key := range sortedDailyKeys(aggregations) {
		value := aggregations[key]
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyEndUserSpend,
			key.endUserID, key.date, key.apiKey, key.model, key.modelGroup,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens,
			value.cacheReadInputTokens, value.cacheCreationInputTokens,
			value.spend,
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
