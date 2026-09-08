package spendlog

import (
	"encoding/json"
	"fmt"

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
func GetBatchParams(entries []*models.SpendLogEntry, logCredentialName bool) ([]interface{}, error) {
	params := make([]interface{}, 0, len(entries)*queries.SpendLogParamCount)
	for _, entry := range entries {
		if logCredentialName && entry.CredentialName != "" {
			clone := *entry
			metadata, err := addCredentialNameMetadata(entry.Metadata, entry.CredentialName)
			if err != nil {
				return nil, fmt.Errorf("credential metadata for request %q: %w", entry.RequestID, err)
			}
			clone.Metadata = metadata
			entry = &clone
		}
		params = append(params, GetSpendLogParams(entry)...)
	}
	return params, nil
}

func addCredentialNameMetadata(metadata, credentialName string) (string, error) {
	var fields map[string]json.RawMessage
	if metadata != "" {
		if err := json.Unmarshal([]byte(metadata), &fields); err != nil {
			return "", err
		}
	}
	if fields == nil {
		fields = make(map[string]json.RawMessage)
	}
	encodedName, err := json.Marshal(credentialName)
	if err != nil {
		return "", err
	}
	fields["credential_name"] = encodedName
	encoded, err := json.Marshal(fields)
	return string(encoded), err
}
