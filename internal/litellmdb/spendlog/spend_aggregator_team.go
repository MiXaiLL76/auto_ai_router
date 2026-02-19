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
	modelGroup            string
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
	records []spendLogRecord,
) error {
	// Map to aggregate by unique key
	aggregations := make(map[aggregateTeamKey]*aggregationValue)
	totalRows := 0
	skippedRows := 0

	for _, record := range records {
		totalRows++

		// Skip if no team_id
		if record.TeamID == "" {
			skippedRows++
			continue
		}

		key := aggregateTeamKey{
			teamID:                record.TeamID,
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
			key.teamID, key.date, key.apiKey, key.model, key.modelGroup,
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
