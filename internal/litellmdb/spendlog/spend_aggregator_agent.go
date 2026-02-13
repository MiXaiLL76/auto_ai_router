package spendlog

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregateAgentKey represents unique agent spend log grouping dimension
type aggregateAgentKey struct {
	agentID               string
	date                  string
	apiKey                string
	model                 string
	modelGroup            string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

// aggregateDailyAgentSpendLogs aggregates spend logs into DailyAgentSpend
//
// This function:
// 1. Fetches spend logs from SpendLogs table filtered by requestIDs
// 2. Groups them by (agent_id, date, api_key, model, provider, mcp_tool, endpoint)
// 3. Sums tokens, spend, and request counts per group
// 4. UPSERTs aggregated data into DailyAgentSpend table
//
// Returns nil if successful (including "no logs to aggregate" case).
// Returns error on any database operation failure.
func aggregateDailyAgentSpendLogs(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	requestIDs []string,
) error {
	records, err := loadUnprocessedSpendLogRecords(ctx, conn, logger, "Agent", requestIDs)
	if err != nil {
		return err
	}

	// Map to aggregate by unique key
	aggregations := make(map[aggregateAgentKey]*aggregationValue)
	totalRows := 0
	skippedRows := 0

	for _, record := range records {
		totalRows++

		// Skip if no agent_id
		if record.AgentID == "" {
			skippedRows++
			continue
		}

		key := aggregateAgentKey{
			agentID:               record.AgentID,
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

	logger.Debug("[DB] Agent aggregation: scan complete",
		"total_rows", totalRows,
		"skipped_rows", skippedRows,
		"aggregation_groups", len(aggregations),
	)

	if len(aggregations) == 0 {
		return nil
	}

	// Insert aggregated data into DailyAgentSpend
	for key, value := range aggregations {
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyAgentSpend,
			key.agentID, key.date, key.apiKey, key.model, key.modelGroup,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens, value.spend,
			value.apiRequests, value.successfulRequests, value.failedRequests,
		)

		if err != nil {
			logger.Error("[DB] Agent aggregation: failed to upsert daily spend", "error", err, "key", key)
			return err
		}
	}

	logger.Debug("[DB] Agent aggregation completed",
		"aggregations", len(aggregations),
	)

	return nil
}
