package spendlog

import (
	"context"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

type spendLogRecord struct {
	UserID            string
	Date              string
	APIKey            string
	Model             string
	ModelGroup        string
	CustomLLMProvider string
	MCPNamespacedTool string
	Endpoint          string
	PromptTokens      int
	CompletionTokens  int
	Spend             float64
	Status            string
	RequestID         string
	TeamID            string
	OrganizationID    string
	EndUser           string
	AgentID           string
	RequestTags       string
}

func loadUnprocessedSpendLogRecords(
	ctx context.Context,
	conn *pgxpool.Conn,
	logger *slog.Logger,
	scope string,
	requestIDs []string,
) ([]spendLogRecord, error) {
	rows, err := conn.Query(ctx, queries.QuerySelectUnprocessedSpendLogs, requestIDs)
	if err != nil {
		logger.Error("[DB] "+scope+" aggregation: failed to fetch spend logs", "error", err)
		return nil, err
	}
	defer rows.Close()

	records := make([]spendLogRecord, 0, len(requestIDs))
	for rows.Next() {
		var record spendLogRecord
		var model, modelGroup, customLLMProvider, mcpNamespacedToolName, apiBase *string
		var status *string
		var teamID, organizationID, endUser, agentID, requestTags *string

		err := rows.Scan(
			&record.UserID,
			&record.Date,
			&record.APIKey,
			&model,
			&modelGroup,
			&customLLMProvider,
			&mcpNamespacedToolName,
			&apiBase,
			&record.PromptTokens,
			&record.CompletionTokens,
			&record.Spend,
			&status,
			&record.RequestID,
			&teamID,
			&organizationID,
			&endUser,
			&agentID,
			&requestTags,
		)
		if err != nil {
			logger.Error("[DB] "+scope+" aggregation: failed to scan row", "error", err)
			return nil, err
		}

		record.Model = derefString(model)
		record.ModelGroup = derefString(modelGroup)
		record.CustomLLMProvider = derefString(customLLMProvider)
		record.MCPNamespacedTool = derefString(mcpNamespacedToolName)
		record.Endpoint = derefString(apiBase)
		record.Status = derefString(status)
		record.TeamID = derefString(teamID)
		record.OrganizationID = derefString(organizationID)
		record.EndUser = derefString(endUser)
		record.AgentID = derefString(agentID)
		record.RequestTags = derefString(requestTags)

		records = append(records, record)
	}

	if err := rows.Err(); err != nil {
		logger.Error("[DB] "+scope+" aggregation: failed to iterate rows", "error", err)
		return nil, err
	}

	return records, nil
}

func derefString(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
