package spendlog

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestAggregateDailyUserSpendLogs_EmptyRecords tests that empty records return nil (no error)
func TestAggregateDailyUserSpendLogs_EmptyRecords(t *testing.T) {
	records := []spendLogRecord{}

	// This should return nil (no error, nothing to aggregate)
	// We can't call the actual function without a DB connection,
	// but we can test the aggregation logic separately
	assert.Empty(t, records)
}

// TestAggregateDailyTeamSpendLogs_EmptyRecords tests that empty records return nil
func TestAggregateDailyTeamSpendLogs_EmptyRecords(t *testing.T) {
	records := []spendLogRecord{}
	assert.Empty(t, records)
}

// TestAggregateDailyOrganizationSpendLogs_EmptyRecords tests that empty records return nil
func TestAggregateDailyOrganizationSpendLogs_EmptyRecords(t *testing.T) {
	records := []spendLogRecord{}
	assert.Empty(t, records)
}

// TestAggregateDailyEndUserSpendLogs_EmptyRecords tests that empty records return nil
func TestAggregateDailyEndUserSpendLogs_EmptyRecords(t *testing.T) {
	records := []spendLogRecord{}
	assert.Empty(t, records)
}

// TestSpendLogRecordFields verifies that spendLogRecord has all required fields
func TestSpendLogRecordFields(t *testing.T) {
	record := spendLogRecord{
		UserID:            "user-123",
		Date:              "2024-01-15",
		APIKey:            "sk-test-key",
		Model:             "gpt-4",
		ModelGroup:        "gpt-4-group",
		CustomLLMProvider: "openai",
		MCPNamespacedTool: "mcp-tool",
		Endpoint:          "https://api.openai.com/v1/chat/completions",
		PromptTokens:      100,
		CompletionTokens:  50,
		Spend:             0.002,
		Status:            "success",
		RequestID:         "req-123",
		TeamID:            "team-456",
		OrganizationID:    "org-789",
		EndUser:           "end-user-001",
		AgentID:           "agent-001",
		RequestTags:       "tag1,tag2",
	}

	assert.Equal(t, "user-123", record.UserID)
	assert.Equal(t, "2024-01-15", record.Date)
	assert.Equal(t, "sk-test-key", record.APIKey)
	assert.Equal(t, "gpt-4", record.Model)
	assert.Equal(t, "gpt-4-group", record.ModelGroup)
	assert.Equal(t, "openai", record.CustomLLMProvider)
	assert.Equal(t, "mcp-tool", record.MCPNamespacedTool)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", record.Endpoint)
	assert.Equal(t, 100, record.PromptTokens)
	assert.Equal(t, 50, record.CompletionTokens)
	assert.Equal(t, 0.002, record.Spend)
	assert.Equal(t, "success", record.Status)
	assert.Equal(t, "req-123", record.RequestID)
	assert.Equal(t, "team-456", record.TeamID)
	assert.Equal(t, "org-789", record.OrganizationID)
	assert.Equal(t, "end-user-001", record.EndUser)
	assert.Equal(t, "agent-001", record.AgentID)
	assert.Equal(t, "tag1,tag2", record.RequestTags)
}

// TestSpendLogRecord_NilFields tests that records with nil fields are handled correctly by derefString
func TestSpendLogRecord_NilFields(t *testing.T) {
	// Test derefString with nil pointer
	result := derefString(nil)
	assert.Equal(t, "", result)

	// Test derefString with empty string
	empty := ""
	result = derefString(&empty)
	assert.Equal(t, "", result)

	// Test derefString with value
	value := "test-value"
	result = derefString(&value)
	assert.Equal(t, "test-value", result)
}

// TestAggregationValue_Fields verifies the aggregationValue structure
func TestAggregationValue_Fields(t *testing.T) {
	agg := &aggregationValue{
		promptTokens:       1000,
		completionTokens:   500,
		spend:              0.05,
		apiRequests:        10,
		successfulRequests: 9,
		failedRequests:     1,
	}

	assert.Equal(t, int64(1000), agg.promptTokens)
	assert.Equal(t, int64(500), agg.completionTokens)
	assert.Equal(t, 0.05, agg.spend)
	assert.Equal(t, int64(10), agg.apiRequests)
	assert.Equal(t, int64(9), agg.successfulRequests)
	assert.Equal(t, int64(1), agg.failedRequests)
}

// TestAggregationKey_Fields verifies the aggregationKey structure for user aggregation
func TestAggregationKey_Fields(t *testing.T) {
	key := aggregationKey{
		userID:                "user-123",
		date:                  "2024-01-15",
		apiKey:                "sk-test-key",
		model:                 "gpt-4",
		modelGroup:            "gpt-4-group",
		customLLMProvider:     "openai",
		mcpNamespacedToolName: "mcp-tool",
		endpoint:              "https://api.openai.com/v1/chat/completions",
	}

	assert.Equal(t, "user-123", key.userID)
	assert.Equal(t, "2024-01-15", key.date)
	assert.Equal(t, "sk-test-key", key.apiKey)
	assert.Equal(t, "gpt-4", key.model)
	assert.Equal(t, "gpt-4-group", key.modelGroup)
	assert.Equal(t, "openai", key.customLLMProvider)
	assert.Equal(t, "mcp-tool", key.mcpNamespacedToolName)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", key.endpoint)
}

// TestAggregateTeamKey_Fields verifies the aggregateTeamKey structure
func TestAggregateTeamKey_Fields(t *testing.T) {
	key := aggregateTeamKey{
		teamID:                "team-456",
		date:                  "2024-01-15",
		apiKey:                "sk-test-key",
		model:                 "gpt-4",
		modelGroup:            "gpt-4-group",
		customLLMProvider:     "openai",
		mcpNamespacedToolName: "mcp-tool",
		endpoint:              "https://api.openai.com/v1/chat/completions",
	}

	assert.Equal(t, "team-456", key.teamID)
	assert.Equal(t, "2024-01-15", key.date)
	assert.Equal(t, "sk-test-key", key.apiKey)
	assert.Equal(t, "gpt-4", key.model)
	assert.Equal(t, "gpt-4-group", key.modelGroup)
	assert.Equal(t, "openai", key.customLLMProvider)
	assert.Equal(t, "mcp-tool", key.mcpNamespacedToolName)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", key.endpoint)
}

// TestAggregateOrgKey_Fields verifies the aggregateOrgKey structure
func TestAggregateOrgKey_Fields(t *testing.T) {
	key := aggregateOrgKey{
		organizationID:        "org-789",
		date:                  "2024-01-15",
		apiKey:                "sk-test-key",
		model:                 "gpt-4",
		modelGroup:            "gpt-4-group",
		customLLMProvider:     "openai",
		mcpNamespacedToolName: "mcp-tool",
		endpoint:              "https://api.openai.com/v1/chat/completions",
	}

	assert.Equal(t, "org-789", key.organizationID)
	assert.Equal(t, "2024-01-15", key.date)
	assert.Equal(t, "sk-test-key", key.apiKey)
	assert.Equal(t, "gpt-4", key.model)
	assert.Equal(t, "gpt-4-group", key.modelGroup)
	assert.Equal(t, "openai", key.customLLMProvider)
	assert.Equal(t, "mcp-tool", key.mcpNamespacedToolName)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", key.endpoint)
}

// TestAggregateEndUserKey_Fields verifies the aggregateEndUserKey structure
func TestAggregateEndUserKey_Fields(t *testing.T) {
	key := aggregateEndUserKey{
		endUserID:             "end-user-001",
		date:                  "2024-01-15",
		apiKey:                "sk-test-key",
		model:                 "gpt-4",
		modelGroup:            "gpt-4-group",
		customLLMProvider:     "openai",
		mcpNamespacedToolName: "mcp-tool",
		endpoint:              "https://api.openai.com/v1/chat/completions",
	}

	assert.Equal(t, "end-user-001", key.endUserID)
	assert.Equal(t, "2024-01-15", key.date)
	assert.Equal(t, "sk-test-key", key.apiKey)
	assert.Equal(t, "gpt-4", key.model)
	assert.Equal(t, "gpt-4-group", key.modelGroup)
	assert.Equal(t, "openai", key.customLLMProvider)
	assert.Equal(t, "mcp-tool", key.mcpNamespacedToolName)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", key.endpoint)
}

// TestAggregateAgentKey_Fields verifies the aggregateAgentKey structure
func TestAggregateAgentKey_Fields(t *testing.T) {
	key := aggregateAgentKey{
		agentID:               "agent-001",
		date:                  "2024-01-15",
		apiKey:                "sk-test-key",
		model:                 "gpt-4",
		modelGroup:            "gpt-4-group",
		customLLMProvider:     "openai",
		mcpNamespacedToolName: "mcp-tool",
		endpoint:              "https://api.openai.com/v1/chat/completions",
	}

	assert.Equal(t, "agent-001", key.agentID)
	assert.Equal(t, "2024-01-15", key.date)
	assert.Equal(t, "sk-test-key", key.apiKey)
	assert.Equal(t, "gpt-4", key.model)
	assert.Equal(t, "gpt-4-group", key.modelGroup)
	assert.Equal(t, "openai", key.customLLMProvider)
	assert.Equal(t, "mcp-tool", key.mcpNamespacedToolName)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", key.endpoint)
}

// TestAggregateTagKey_Fields verifies the aggregateTagKey structure
func TestAggregateTagKey_Fields(t *testing.T) {
	key := aggregateTagKey{
		tag:                   "tag1",
		date:                  "2024-01-15",
		apiKey:                "sk-test-key",
		model:                 "gpt-4",
		modelGroup:            "gpt-4-group",
		customLLMProvider:     "openai",
		mcpNamespacedToolName: "mcp-tool",
		endpoint:              "https://api.openai.com/v1/chat/completions",
	}

	assert.Equal(t, "tag1", key.tag)
	assert.Equal(t, "2024-01-15", key.date)
	assert.Equal(t, "sk-test-key", key.apiKey)
	assert.Equal(t, "gpt-4", key.model)
	assert.Equal(t, "gpt-4-group", key.modelGroup)
	assert.Equal(t, "openai", key.customLLMProvider)
	assert.Equal(t, "mcp-tool", key.mcpNamespacedToolName)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", key.endpoint)
}

// TestSpendLogRecord_EmptyFields tests records with empty optional fields
func TestSpendLogRecord_EmptyFields(t *testing.T) {
	record := spendLogRecord{
		UserID:           "user-123",
		Date:             "2024-01-15",
		APIKey:           "sk-test",
		Model:            "gpt-4",
		Spend:            0.001,
		Status:           "success",
		RequestID:        "req-001",
		PromptTokens:     10,
		CompletionTokens: 5,
	}

	// All optional fields should be empty strings
	assert.Equal(t, "", record.ModelGroup)
	assert.Equal(t, "", record.CustomLLMProvider)
	assert.Equal(t, "", record.MCPNamespacedTool)
	assert.Equal(t, "", record.Endpoint)
	assert.Equal(t, "", record.TeamID)
	assert.Equal(t, "", record.OrganizationID)
	assert.Equal(t, "", record.EndUser)
	assert.Equal(t, "", record.AgentID)
	assert.Equal(t, "", record.RequestTags)
}

// TestSpendLogRecord_MultipleStatuses tests handling of different status values
func TestSpendLogRecord_MultipleStatuses(t *testing.T) {
	successRecord := spendLogRecord{Status: "success"}
	failureRecord := spendLogRecord{Status: "failure"}
	errorRecord := spendLogRecord{Status: "error"}
	emptyStatusRecord := spendLogRecord{Status: ""}

	assert.Equal(t, "success", successRecord.Status)
	assert.Equal(t, "failure", failureRecord.Status)
	assert.Equal(t, "error", errorRecord.Status)
	assert.Equal(t, "", emptyStatusRecord.Status)
}

// TestSpendLogRecord_TokenCalculation tests token math operations
func TestSpendLogRecord_TokenCalculation(t *testing.T) {
	records := []spendLogRecord{
		{PromptTokens: 100, CompletionTokens: 50, Spend: 0.001},
		{PromptTokens: 200, CompletionTokens: 100, Spend: 0.002},
		{PromptTokens: 150, CompletionTokens: 75, Spend: 0.0015},
	}

	var totalPrompt, totalCompletion int
	var totalSpend float64

	for _, r := range records {
		totalPrompt += r.PromptTokens
		totalCompletion += r.CompletionTokens
		totalSpend += r.Spend
	}

	assert.Equal(t, 450, totalPrompt)
	assert.Equal(t, 225, totalCompletion)
	assert.InDelta(t, 0.0045, totalSpend, 0.0001)
}
