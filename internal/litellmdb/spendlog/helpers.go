package spendlog

import "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"

// GetSpendLogParams returns parameters for a single SpendLogEntry
func GetSpendLogParams(entry *models.SpendLogEntry) []interface{} {
	return []interface{}{
		entry.RequestID,         // $1
		entry.CallType,          // $2
		entry.APIKey,            // $3
		entry.Spend,             // $4
		entry.TotalTokens,       // $5
		entry.PromptTokens,      // $6
		entry.CompletionTokens,  // $7
		entry.StartTime,         // $8
		entry.EndTime,           // $9
		entry.Model,             // $10
		entry.ModelID,           // $11
		entry.ModelGroup,        // $12
		entry.CustomLLMProvider, // $13
		entry.APIBase,           // $14
		entry.UserID,            // $15 ("user" column)
		entry.Metadata,          // $16 ("metadata" column)
		entry.TeamID,            // $17
		entry.OrganizationID,    // $18
		entry.EndUser,           // $19
		entry.RequesterIP,       // $20
		entry.Status,            // $21
		entry.SessionID,         // $22
	}
}

// GetBatchParams returns all parameters for batch insert
func GetBatchParams(entries []*models.SpendLogEntry) []interface{} {
	params := make([]interface{}, 0, len(entries)*22) // 22 params per entry
	for _, entry := range entries {
		params = append(params, GetSpendLogParams(entry)...)
	}
	return params
}
