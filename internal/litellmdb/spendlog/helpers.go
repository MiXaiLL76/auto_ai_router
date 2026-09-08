package spendlog

import (
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// GetSpendLogParams returns parameters for a single SpendLogEntry
func GetSpendLogParams(entry *models.SpendLogEntry) []interface{} {
	// metadata: ensure valid JSON
	metadata := entry.Metadata
	if metadata == "" {
		metadata = "{}" // default to empty JSON object
	}

	return []interface{}{
		entry.RequestID,           // $1
		entry.CallType,            // $2
		entry.APIKey,              // $3
		entry.Spend,               // $4
		entry.TotalTokens,         // $5
		entry.PromptTokens,        // $6
		entry.CompletionTokens,    // $7
		entry.StartTime,           // $8
		entry.EndTime,             // $9
		entry.RequestDurationMS,   // $10
		entry.CompletionStartTime, // $11
		entry.Model,               // $12
		entry.ModelID,             // $13
		entry.ModelGroup,          // $14
		entry.CustomLLMProvider,   // $15
		entry.APIBase,             // $16
		entry.UserID,              // $17 ("user" column)
		metadata,                  // $18 ("metadata" column) - JSON object
		entry.CacheHit,            // $19
		entry.CacheKey,            // $20
		entry.TeamID,              // $21
		entry.OrganizationID,      // $22
		entry.EndUser,             // $23
		entry.RequesterIP,         // $24
		entry.SessionID,           // $25
		entry.Status,              // $26
	}
}

// GetBatchParams returns all parameters for batch insert
func GetBatchParams(entries []*models.SpendLogEntry) []interface{} {
	params := make([]interface{}, 0, len(entries)*queries.SpendLogParamCount)
	for _, entry := range entries {
		params = append(params, GetSpendLogParams(entry)...)
	}
	return params
}
