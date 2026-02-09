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
	requestIDs []string,
) error {
	// Fetch spend logs for the given request_ids
	rows, err := conn.Query(ctx, queries.QuerySelectUnprocessedSpendLogs, requestIDs)
	if err != nil {
		logger.Error("[DB] Tag aggregation: failed to fetch spend logs", "error", err)
		return err
	}
	defer rows.Close()

	// Map to aggregate by unique key
	aggregations := make(map[aggregateTagKey]*aggregateTagValue)
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
			logger.Error("[DB] Tag aggregation: failed to scan row", "error", err)
			continue
		}

		// Skip if no tags
		if requestTags == nil || *requestTags == "" || *requestTags == "[]" {
			skippedRows++
			continue
		}

		// Parse tags from JSON array
		var tags []string
		err = json.Unmarshal([]byte(*requestTags), &tags)
		if err != nil {
			logger.Warn("[DB] Tag aggregation: failed to unmarshal request_tags JSON",
				"request_id", requestID,
				"request_tags", *requestTags,
				"error", err,
			)
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

		// For each tag in the array, aggregate
		for _, tag := range tags {
			if tag == "" {
				continue
			}

			key := aggregateTagKey{
				tag:                   tag,
				date:                  date,
				apiKey:                apiKey,
				model:                 modelStr,
				customLLMProvider:     customProviderStr,
				mcpNamespacedToolName: mcpToolStr,
				endpoint:              apiBaseStr,
			}

			if aggregations[key] == nil {
				aggregations[key] = &aggregateTagValue{
					requestID: requestID, // Store first request_id
				}
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
	}

	if rows.Err() != nil {
		logger.Error("[DB] Tag aggregation: failed to iterate rows", "error", rows.Err())
		return rows.Err()
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
			key.tag, value.requestID, key.date, key.apiKey, key.model,
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
