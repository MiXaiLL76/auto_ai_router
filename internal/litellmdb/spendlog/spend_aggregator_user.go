package spendlog

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregationKey represents unique spend log grouping dimension
type aggregationKey struct {
	userID                string
	date                  string
	apiKey                string
	model                 string
	modelGroup            string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

// aggregationValue holds aggregated metrics for a single dimension
type aggregationValue struct {
	promptTokens       int64
	completionTokens   int64
	spend              float64
	apiRequests        int64
	successfulRequests int64
	failedRequests     int64
}

// aggregateDailyUserSpendLogs aggregates spend logs into DailyUserSpend.
//
// This function:
// 1. Fetches spend logs from SpendLogs table filtered by requestIDs
// 2. Groups them by (user_id, date, api_key, model, provider, mcp_tool, endpoint)
// 3. Sums tokens, spend, and request counts per group
// 4. UPSERTs aggregated data into DailyUserSpend table
//
// Returns nil if successful (including "no logs to aggregate" case).
// Returns error on any database operation failure.
func aggregateDailyUserSpendLogs(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	requestIDs []string,
) error {
	records, err := loadUnprocessedSpendLogRecords(ctx, conn, logger, "User", requestIDs)
	if err != nil {
		return err
	}

	// Map to aggregate by unique key (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
	aggregations := make(map[aggregationKey]*aggregationValue)

	for _, record := range records {
		logger.Debug("[DB] User aggregation: log scanned",
			"request_id", record.RequestID,
			"user_id", record.UserID,
			"date", record.Date,
			"api_key_prefix", safeAPIKeyPrefix(record.APIKey),
			"spend", record.Spend,
		)

		// Skip if no user_id
		if record.UserID == "" {
			continue
		}

		key := aggregationKey{
			userID:                record.UserID,
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

	if len(aggregations) == 0 {
		// No unprocessed logs
		return nil
	}

	// Insert aggregated data into DailyUserSpend
	upsertCount := 0
	for key, value := range aggregations {
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyUserSpend,
			key.userID, key.date, key.apiKey, key.model, key.modelGroup,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens, value.spend,
			value.apiRequests, value.successfulRequests, value.failedRequests,
		)

		if err != nil {
			logger.Error("[DB] Aggregation: failed to upsert daily spend", "error", err, "key", key)
			return err
		}
		upsertCount++

		logger.Debug("[DB] User aggregation: upsert executed",
			"user_id", key.userID,
			"date", key.date,
			"api_key", key.apiKey,
			"model", key.model,
			"api_requests", value.apiRequests,
			"spend", value.spend,
		)
	}

	logger.Debug("[DB] User aggregation: all upserts completed",
		"total_upserts", upsertCount,
	)

	return nil
}

func safeAPIKeyPrefix(apiKey string) string {
	if len(apiKey) <= 8 {
		return apiKey
	}
	return apiKey[:8]
}
