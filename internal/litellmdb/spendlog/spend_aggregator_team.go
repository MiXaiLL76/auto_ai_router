package spendlog

import (
	"context"
	"log/slog"
	"strings"

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

func (k aggregateTeamKey) lockOrder() string {
	return strings.Join([]string{k.teamID, k.date, k.apiKey, k.model, k.modelGroup,
		k.customLLMProvider, k.mcpNamespacedToolName, k.endpoint}, "\x00")
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
	conn dailySpendExecer,
	logger *slog.Logger,
	records []spendLogRecord,
) error {
	// Map to aggregate by unique key
	aggregations := make(map[aggregateTeamKey]*aggregationValue)
	totalRows := 0

	for _, record := range records {
		totalRows++

		// No skip for empty team_id: LiteLLM records daily team rows with
		// team_id="" for keys without a team, and the spend comparison
		// counts them.
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

		aggregations[key].addRecord(record)
	}

	logger.Debug("[DB] Team aggregation: scan complete",
		"total_rows", totalRows,
		"aggregation_groups", len(aggregations),
	)

	if len(aggregations) == 0 {
		// No logs to aggregate for teams
		return nil
	}

	// Insert aggregated data into DailyTeamSpend
	for _, key := range sortedDailyKeys(aggregations) {
		value := aggregations[key]
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyTeamSpend,
			key.teamID, key.date, key.apiKey, key.model, key.modelGroup,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens,
			value.cacheReadInputTokens, value.cacheCreationInputTokens,
			value.spend,
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
