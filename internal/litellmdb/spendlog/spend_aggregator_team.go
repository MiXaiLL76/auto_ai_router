package spendlog

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregateTeamKey represents unique team spend log grouping dimension
type aggregateTeamKey struct {
	teamID                string
	date                  string
	apiKey                string
	model                 string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

// aggregateDailyTeamSpendLogs aggregates spend logs into DailyTeamSpend
//
// This function:
// 1. Fetches spend logs from SpendLogs table filtered by requestIDs
// 2. Groups them by (team_id, date, api_key, model, provider, mcp_tool, endpoint)
// 3. Sums tokens, spend, and request counts per group
// 4. UPSERTs aggregated data into DailyTeamSpend table
//
// Returns nil if successful (including "no logs to aggregate" case).
// Returns error on any database operation failure.
func aggregateDailyTeamSpendLogs(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	requestIDs []string,
) error {
	// Fetch spend logs for the given request_ids
	rows, err := conn.Query(ctx, queries.QuerySelectUnprocessedSpendLogs, requestIDs)
	if err != nil {
		logger.Error("[DB] Team aggregation: failed to fetch spend logs", "error", err)
		return err
	}
	defer rows.Close()

	// Map to aggregate by unique key
	aggregations := make(map[aggregateTeamKey]*aggregationValue)
	totalRows := 0
	skippedRows := 0

	// Aggregate rows
	for rows.Next() {
		totalRows++
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
			logger.Error("[DB] Team aggregation: failed to scan row", "error", err)
			continue
		}

		// Skip if no team_id
		if teamID == nil || *teamID == "" {
			skippedRows++
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

		key := aggregateTeamKey{
			teamID:                *teamID,
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
		logger.Error("[DB] Team aggregation: failed to iterate rows", "error", rows.Err())
		return rows.Err()
	}

	logger.Debug("[DB] Team aggregation: scan complete",
		"total_rows", totalRows,
		"skipped_rows", skippedRows,
		"aggregation_groups", len(aggregations),
	)

	if len(aggregations) == 0 {
		// No logs to aggregate for teams
		return nil
	}

	// Insert aggregated data into DailyTeamSpend
	for key, value := range aggregations {
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyTeamSpend,
			key.teamID, key.date, key.apiKey, key.model,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens, value.spend,
			value.apiRequests, value.successfulRequests, value.failedRequests,
		)

		if err != nil {
			logger.Error("[DB] Team aggregation: failed to upsert daily spend", "error", err, "key", key)
			return err
		}
	}

	logger.Debug("[DB] Team aggregation completed",
		"aggregations", len(aggregations),
	)

	return nil
}
