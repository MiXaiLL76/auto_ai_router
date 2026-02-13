package spendlog

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregateOrgKey represents unique organization spend log grouping dimension
type aggregateOrgKey struct {
	organizationID        string
	date                  string
	apiKey                string
	model                 string
	modelGroup            string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

// aggregateDailyOrganizationSpendLogs aggregates spend logs into DailyOrganizationSpend
//
// This function:
// 1. Fetches spend logs from SpendLogs table filtered by requestIDs
// 2. Groups them by (organization_id, date, api_key, model, provider, mcp_tool, endpoint)
// 3. Sums tokens, spend, and request counts per group
// 4. UPSERTs aggregated data into DailyOrganizationSpend table
//
// Returns nil if successful (including "no logs to aggregate" case).
// Returns error on any database operation failure.
func aggregateDailyOrganizationSpendLogs(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	requestIDs []string,
) error {
	records, err := loadUnprocessedSpendLogRecords(ctx, conn, logger, "Organization", requestIDs)
	if err != nil {
		return err
	}

	// Map to aggregate by unique key
	aggregations := make(map[aggregateOrgKey]*aggregationValue)
	totalRows := 0
	skippedRows := 0

	for _, record := range records {
		totalRows++

		// Skip if no organization_id
		if record.OrganizationID == "" {
			skippedRows++
			continue
		}

		key := aggregateOrgKey{
			organizationID:        record.OrganizationID,
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

		logger.Debug("[DB] Organization aggregation: log accumulated",
			"org_id", record.OrganizationID,
			"date", record.Date,
			"api_key", record.APIKey,
			"model", record.Model,
			"api_requests_total", agg.apiRequests,
			"spend_total", agg.spend,
		)
	}

	logger.Info("[DB] Organization aggregation: scan complete",
		"total_rows", totalRows,
		"skipped_rows", skippedRows,
		"aggregation_groups", len(aggregations),
	)

	if len(aggregations) == 0 {
		logger.Info("[DB] Organization aggregation: no aggregations to insert")
		return nil
	}

	// Insert aggregated data into DailyOrganizationSpend
	upsertCount := 0
	for key, value := range aggregations {
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyOrganizationSpend,
			key.organizationID, key.date, key.apiKey, key.model, key.modelGroup,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens, value.spend,
			value.apiRequests, value.successfulRequests, value.failedRequests,
		)

		if err != nil {
			logger.Error("[DB] Organization aggregation: failed to upsert daily spend", "error", err, "key", key)
			return err
		}
		upsertCount++

		logger.Info("[DB] Organization aggregation: upsert executed",
			"org_id", key.organizationID,
			"date", key.date,
			"api_requests", value.apiRequests,
			"spend", value.spend,
		)
	}

	logger.Info("[DB] Organization aggregation: all upserts completed",
		"total_upserts", upsertCount,
	)

	logger.Debug("Organization aggregation completed",
		"aggregations", len(aggregations),
	)

	return nil
}
