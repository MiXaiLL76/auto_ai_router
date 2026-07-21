package spendlog

import (
	"encoding/json"

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

	requestTags := normalizeRequestTags(entry.RequestTags)

	return []interface{}{
		entry.RequestID,             // $1
		entry.CallType,              // $2
		entry.APIKey,                // $3
		entry.Spend,                 // $4
		entry.TotalTokens,           // $5
		entry.PromptTokens,          // $6
		entry.CompletionTokens,      // $7
		entry.StartTime,             // $8
		entry.EndTime,               // $9
		entry.RequestDurationMS,     // $10
		entry.CompletionStartTime,   // $11
		entry.Model,                 // $12
		entry.ModelID,               // $13
		entry.ModelGroup,            // $14
		entry.CustomLLMProvider,     // $15
		entry.APIBase,               // $16
		entry.UserID,                // $17 ("user" column)
		metadata,                    // $18 ("metadata" column) - JSON object
		entry.CacheHit,              // $19
		entry.CacheKey,              // $20
		requestTags,                 // $21 (JSON array as string)
		entry.TeamID,                // $22
		entry.OrganizationID,        // $23
		entry.EndUser,               // $24
		entry.RequesterIP,           // $25
		entry.SessionID,             // $26
		entry.Status,                // $27
	}
}

// normalizeRequestTags keeps the database column array-shaped even if a caller
// accidentally supplies JSON null, a scalar, or malformed JSON. Production
// views call jsonb_array_elements_text and fail on scalar JSON values.
func normalizeRequestTags(raw string) string {
	var tags []string
	if err := json.Unmarshal([]byte(raw), &tags); err != nil || tags == nil {
		return "[]"
	}
	encoded, err := json.Marshal(tags)
	if err != nil {
		return "[]"
	}
	return string(encoded)
}

// GetBatchParams returns all parameters for batch insert
func GetBatchParams(entries []*models.SpendLogEntry) []interface{} {
	params := make([]interface{}, 0, len(entries)*queries.SpendLogParamCount)
	for _, entry := range entries {
		params = append(params, GetSpendLogParams(entry)...)
	}
	return params
}
