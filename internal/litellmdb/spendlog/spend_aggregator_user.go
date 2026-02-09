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
	// Fetch spend logs for the given request_ids
	rows, err := conn.Query(ctx, queries.QuerySelectUnprocessedSpendLogs, requestIDs)
	if err != nil {
		logger.Error("[DB] Aggregation: failed to fetch spend logs", "error", err)
		return err
	}
	defer rows.Close()

	// Map to aggregate by unique key (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
	aggregations := make(map[aggregationKey]*aggregationValue)

	// Aggregate rows
	for rows.Next() {
		var userID, date, apiKey string
		var model, customLLMProvider, mcpNamespacedToolName, apiBase *string
		var promptTokens, completionTokens int
		var spend float64
		var status *string
		var requestID string
		var teamID, organizationID, endUser, agentID, requestTags *string

		err := rows.Scan(&userID, &date, &apiKey, &model, &customLLMProvider, &mcpNamespacedToolName, &apiBase,
			&promptTokens, &completionTokens, &spend, &status, &requestID,
			&teamID, &organizationID, &endUser, &agentID, &requestTags)
		if err != nil {
			logger.Error("[DB] Aggregation: failed to scan row", "error", err)
			continue
		}

		logger.Debug("[DB] User aggregation: log scanned",
			"request_id", requestID,
			"user_id", userID,
			"date", date,
			"api_key_prefix", apiKey[:8],
			"spend", spend,
		)

		// Skip if no user_id
		if userID == "" {
			continue
		}

		// Handle nullable fields
		modelStr := ""
		if model != nil {
			modelStr = *model
		}
		customProviderStr := ""
		if customLLMProvider != nil {
			customProviderStr = *customLLMProvider
		}
		mcpToolStr := ""
		if mcpNamespacedToolName != nil {
			mcpToolStr = *mcpNamespacedToolName
		}
		apiBaseStr := ""
		if apiBase != nil {
			apiBaseStr = *apiBase
		}
		statusStr := ""
		if status != nil {
			statusStr = *status
		}

		key := aggregationKey{
			userID:                userID,
			date:                  date,
			apiKey:                apiKey,
			model:                 modelStr,
			customLLMProvider:     customProviderStr,
			mcpNamespacedToolName: mcpToolStr,
			endpoint:              apiBaseStr,
		}

		if aggregations[key] == nil {
			aggregations[key] = &aggregationValue{}
		}

		agg := aggregations[key]
		agg.promptTokens += int64(promptTokens)
		agg.completionTokens += int64(completionTokens)
		agg.spend += spend
		agg.apiRequests++

		if statusStr == "success" {
			agg.successfulRequests++
		} else {
			agg.failedRequests++
		}
	}

	if rows.Err() != nil {
		logger.Error("[DB] Aggregation: failed to iterate rows", "error", rows.Err())
		return rows.Err()
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
			key.userID, key.date, key.apiKey, key.model,
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
