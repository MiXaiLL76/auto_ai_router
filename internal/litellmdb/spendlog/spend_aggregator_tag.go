package spendlog

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// aggregateTagKey represents unique tag spend log grouping dimension
type aggregateTagKey struct {
	tag                   string
	date                  string
	apiKey                string
	model                 string
	modelGroup            string
	customLLMProvider     string
	mcpNamespacedToolName string
	endpoint              string
}

// aggregateTagValue holds aggregated metrics for a tag with request_id
type aggregateTagValue struct {
	promptTokens       int64
	completionTokens   int64
	spend              float64
	apiRequests        int64
	successfulRequests int64
	failedRequests     int64
	requestID          string // Store the first request_id for the tag (required by schema)
}

// aggregateDailyTagSpendLogs aggregates spend logs into DailyTagSpend
//
// This function:
// 1. Fetches spend logs from SpendLogs table filtered by requestIDs
// 2. For each log, parses request_tags JSON array
// 3. Groups by (tag, date, api_key, model, provider, mcp_tool, endpoint)
// 4. Sums tokens, spend, and request counts per group
// 5. UPSERTs aggregated data into DailyTagSpend table
//
// Returns nil if successful (including "no logs to aggregate" case).
// Returns error on any database operation failure.
func aggregateDailyTagSpendLogs(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	records []spendLogRecord,
) error {
	// Map to aggregate by unique key
	aggregations := make(map[aggregateTagKey]*aggregateTagValue)
	totalRows := 0
	skippedRows := 0

	for _, record := range records {
		totalRows++

		// Skip if no tags
		if record.RequestTags == "" || record.RequestTags == "[]" {
			skippedRows++
			continue
		}

		// Parse tags from JSON array
		var tags []string
		err := json.Unmarshal([]byte(record.RequestTags), &tags)
		if err != nil {
			logger.Warn("[DB] Tag aggregation: failed to unmarshal request_tags JSON",
				"request_id", record.RequestID,
				"request_tags", record.RequestTags,
				"error", err,
			)
			continue
		}

		// For each tag in the array, aggregate
		for _, tag := range tags {
			if tag == "" {
				continue
			}

			key := aggregateTagKey{
				tag:                   tag,
				date:                  record.Date,
				apiKey:                record.APIKey,
				model:                 record.Model,
				modelGroup:            record.ModelGroup,
				customLLMProvider:     record.CustomLLMProvider,
				mcpNamespacedToolName: record.MCPNamespacedTool,
				endpoint:              record.Endpoint,
			}

			if aggregations[key] == nil {
				aggregations[key] = &aggregateTagValue{
					requestID: record.RequestID, // Store first request_id
				}
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
	}

	logger.Debug("[DB] Tag aggregation: scan complete",
		"total_rows", totalRows,
		"skipped_rows", skippedRows,
		"aggregation_groups", len(aggregations),
	)

	if len(aggregations) == 0 {
		return nil
	}

	// Insert aggregated data into DailyTagSpend
	for key, value := range aggregations {
		_, err := conn.Exec(ctx,
			queries.QueryUpsertDailyTagSpend,
			key.tag, value.requestID, key.date, key.apiKey, key.model, key.modelGroup,
			key.customLLMProvider, key.mcpNamespacedToolName, key.endpoint,
			value.promptTokens, value.completionTokens, value.spend,
			value.apiRequests, value.successfulRequests, value.failedRequests,
		)

		if err != nil {
			logger.Error("[DB] Tag aggregation: failed to upsert daily spend", "error", err, "key", key)
			return err
		}
	}

	logger.Debug("[DB] Tag aggregation completed",
		"aggregations", len(aggregations),
	)

	return nil
}
