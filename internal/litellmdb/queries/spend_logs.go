package queries

import (
	"fmt"
	"strings"
)

// SQL queries for LiteLLM_SpendLogs table

const (
	// QueryInsertSpendLog inserts a single spend log entry
	QueryInsertSpendLog = `
		INSERT INTO "LiteLLM_SpendLogs" (
			request_id,
			call_type,
			api_key,
			spend,
			total_tokens,
			prompt_tokens,
			completion_tokens,
			"startTime",
			"endTime",
			model,
			model_id,
			model_group,
			custom_llm_provider,
			api_base,
			"user",
			"metadata",
			team_id,
			organization_id,
			end_user,
			requester_ip_address,
			status,
			session_id,
			messages,
			response
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
			$11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
			$21, $22, '{}', '{}'
		)
		ON CONFLICT (request_id) DO NOTHING
	`

	// QuerySelectUnprocessedSpendLogs retrieves spend logs not yet aggregated
	QuerySelectUnprocessedSpendLogs = `
		SELECT
			"user",
			TO_CHAR(DATE("startTime"), 'YYYY-MM-DD') as date,
			api_key,
			model,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			api_base,
			prompt_tokens,
			completion_tokens,
			spend,
			status,
			request_id
		FROM "LiteLLM_SpendLogs"
		WHERE cache_hit IS NULL OR cache_hit = ''
		ORDER BY "startTime" DESC
	`

	// QueryUpsertDailyUserSpend upserts into LiteLLM_DailyUserSpend
	QueryUpsertDailyUserSpend = `
		INSERT INTO "LiteLLM_DailyUserSpend" (
			id,
			user_id,
			date,
			api_key,
			model,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			endpoint,
			prompt_tokens,
			completion_tokens,
			spend,
			api_requests,
			successful_requests,
			failed_requests,
			created_at,
			updated_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, now(), now())
		ON CONFLICT (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
		DO UPDATE SET
			prompt_tokens = "LiteLLM_DailyUserSpend".prompt_tokens + EXCLUDED.prompt_tokens,
			completion_tokens = "LiteLLM_DailyUserSpend".completion_tokens + EXCLUDED.completion_tokens,
			spend = "LiteLLM_DailyUserSpend".spend + EXCLUDED.spend,
			api_requests = "LiteLLM_DailyUserSpend".api_requests + EXCLUDED.api_requests,
			successful_requests = "LiteLLM_DailyUserSpend".successful_requests + EXCLUDED.successful_requests,
			failed_requests = "LiteLLM_DailyUserSpend".failed_requests + EXCLUDED.failed_requests,
			updated_at = now()
	`

	// QueryMarkSpendLogsAsProcessed marks spend logs as aggregated
	QueryMarkSpendLogsAsProcessed = `
		UPDATE "LiteLLM_SpendLogs"
		SET cache_hit = 'true'
		WHERE request_id = ANY($1)
	`
)

// Number of parameters per SpendLogEntry in batch insert
const spendLogParamCount = 22

// BuildBatchInsertQuery builds a query for batch INSERT
func BuildBatchInsertQuery(count int) string {
	if count <= 0 {
		return ""
	}

	var b strings.Builder
	b.Grow(500 + count*100) // Pre-allocate

	b.WriteString(`
		INSERT INTO "LiteLLM_SpendLogs" (
			request_id, call_type, api_key, spend, total_tokens,
			prompt_tokens, completion_tokens, "startTime", "endTime",
			model, model_id, model_group, custom_llm_provider, api_base,
			"user", "metadata", team_id, organization_id, end_user,
			requester_ip_address, status, session_id, messages, response
		) VALUES `)

	paramIdx := 1
	for i := 0; i < count; i++ {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString("(")
		for j := 0; j < spendLogParamCount; j++ {
			if j > 0 {
				b.WriteString(", ")
			}
			b.WriteString(fmt.Sprintf("$%d", paramIdx))
			paramIdx++
		}
		b.WriteString(", '{}', '{}')") // messages, response = empty JSON
	}

	b.WriteString(" ON CONFLICT (request_id) DO NOTHING")
	return b.String()
}
