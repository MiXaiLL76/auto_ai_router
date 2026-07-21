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
			request_duration_ms,
			"completionStartTime",
			model,
			model_id,
			model_group,
			custom_llm_provider,
			api_base,
			"user",
			"metadata",
			cache_hit,
			cache_key,
			request_tags,
			team_id,
			organization_id,
			end_user,
			requester_ip_address,
			session_id,
			status,
			messages,
			response,
			proxy_server_request
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
			$11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
			$21, $22, $23, $24, $25, $26, $27,
			'{}'::jsonb, '{}'::jsonb, '{}'::jsonb
		)
		ON CONFLICT (request_id) DO NOTHING
	`

	// QueryClaimSpendLogEventOwner reports whether the spend row with the given
	// primary key already belongs to the AIR event, without reading SpendLogs
	// rows back. The targeted UPDATE doubles as an idempotent claim: it matches
	// only when the row's metadata already attributes the row to this event, so
	// a replay detects its own earlier write, while a genuine provider
	// request_id collision (different owner event) misses and the caller falls
	// back to the AIR event ID row.
	QueryClaimSpendLogEventOwner = `
		UPDATE "LiteLLM_SpendLogs"
		SET metadata = jsonb_set(
			COALESCE(metadata, '{}'::jsonb),
			'{spend_logs_metadata,air_event_id}',
			to_jsonb($2::text),
			true
		)
		WHERE request_id = $1
			AND COALESCE(metadata #>> '{spend_logs_metadata,air_event_id}', '') = $2
		RETURNING request_id
	`

	// QueryUpsertDailyUserSpend upserts into LiteLLM_DailyUserSpend
	QueryUpsertDailyUserSpend = `
		INSERT INTO "LiteLLM_DailyUserSpend" (
			id,
			user_id,
			date,
			api_key,
			model,
			model_group,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			endpoint,
			prompt_tokens,
			completion_tokens,
			cache_read_input_tokens,
			cache_creation_input_tokens,
			spend,
			api_requests,
			successful_requests,
			failed_requests,
			created_at,
			updated_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, now(), now())
		ON CONFLICT (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
		DO UPDATE SET
			model_group = EXCLUDED.model_group,
			prompt_tokens = "LiteLLM_DailyUserSpend".prompt_tokens + EXCLUDED.prompt_tokens,
			completion_tokens = "LiteLLM_DailyUserSpend".completion_tokens + EXCLUDED.completion_tokens,
			cache_read_input_tokens = "LiteLLM_DailyUserSpend".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
			cache_creation_input_tokens = "LiteLLM_DailyUserSpend".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
			spend = "LiteLLM_DailyUserSpend".spend + EXCLUDED.spend,
			api_requests = "LiteLLM_DailyUserSpend".api_requests + EXCLUDED.api_requests,
			successful_requests = "LiteLLM_DailyUserSpend".successful_requests + EXCLUDED.successful_requests,
			failed_requests = "LiteLLM_DailyUserSpend".failed_requests + EXCLUDED.failed_requests,
			updated_at = now()
	`

	// QueryUpsertDailyTeamSpend upserts into LiteLLM_DailyTeamSpend
	QueryUpsertDailyTeamSpend = `
		INSERT INTO "LiteLLM_DailyTeamSpend" (
			id,
			team_id,
			date,
			api_key,
			model,
			model_group,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			endpoint,
			prompt_tokens,
			completion_tokens,
			cache_read_input_tokens,
			cache_creation_input_tokens,
			spend,
			api_requests,
			successful_requests,
			failed_requests,
			created_at,
			updated_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, now(), now())
		ON CONFLICT (team_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
		DO UPDATE SET
			model_group = EXCLUDED.model_group,
			prompt_tokens = "LiteLLM_DailyTeamSpend".prompt_tokens + EXCLUDED.prompt_tokens,
			completion_tokens = "LiteLLM_DailyTeamSpend".completion_tokens + EXCLUDED.completion_tokens,
			cache_read_input_tokens = "LiteLLM_DailyTeamSpend".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
			cache_creation_input_tokens = "LiteLLM_DailyTeamSpend".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
			spend = "LiteLLM_DailyTeamSpend".spend + EXCLUDED.spend,
			api_requests = "LiteLLM_DailyTeamSpend".api_requests + EXCLUDED.api_requests,
			successful_requests = "LiteLLM_DailyTeamSpend".successful_requests + EXCLUDED.successful_requests,
			failed_requests = "LiteLLM_DailyTeamSpend".failed_requests + EXCLUDED.failed_requests,
			updated_at = now()
	`

	// QueryUpsertDailyOrganizationSpend upserts into LiteLLM_DailyOrganizationSpend
	QueryUpsertDailyOrganizationSpend = `
		INSERT INTO "LiteLLM_DailyOrganizationSpend" (
			id,
			organization_id,
			date,
			api_key,
			model,
			model_group,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			endpoint,
			prompt_tokens,
			completion_tokens,
			cache_read_input_tokens,
			cache_creation_input_tokens,
			spend,
			api_requests,
			successful_requests,
			failed_requests,
			created_at,
			updated_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, now(), now())
		ON CONFLICT (organization_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
		DO UPDATE SET
			model_group = EXCLUDED.model_group,
			prompt_tokens = "LiteLLM_DailyOrganizationSpend".prompt_tokens + EXCLUDED.prompt_tokens,
			completion_tokens = "LiteLLM_DailyOrganizationSpend".completion_tokens + EXCLUDED.completion_tokens,
			cache_read_input_tokens = "LiteLLM_DailyOrganizationSpend".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
			cache_creation_input_tokens = "LiteLLM_DailyOrganizationSpend".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
			spend = "LiteLLM_DailyOrganizationSpend".spend + EXCLUDED.spend,
			api_requests = "LiteLLM_DailyOrganizationSpend".api_requests + EXCLUDED.api_requests,
			successful_requests = "LiteLLM_DailyOrganizationSpend".successful_requests + EXCLUDED.successful_requests,
			failed_requests = "LiteLLM_DailyOrganizationSpend".failed_requests + EXCLUDED.failed_requests,
			updated_at = now()
	`

	// QueryUpsertDailyEndUserSpend upserts into LiteLLM_DailyEndUserSpend
	QueryUpsertDailyEndUserSpend = `
		INSERT INTO "LiteLLM_DailyEndUserSpend" (
			id,
			end_user_id,
			date,
			api_key,
			model,
			model_group,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			endpoint,
			prompt_tokens,
			completion_tokens,
			cache_read_input_tokens,
			cache_creation_input_tokens,
			spend,
			api_requests,
			successful_requests,
			failed_requests,
			created_at,
			updated_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, now(), now())
		ON CONFLICT (end_user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
		DO UPDATE SET
			model_group = EXCLUDED.model_group,
			prompt_tokens = "LiteLLM_DailyEndUserSpend".prompt_tokens + EXCLUDED.prompt_tokens,
			completion_tokens = "LiteLLM_DailyEndUserSpend".completion_tokens + EXCLUDED.completion_tokens,
			cache_read_input_tokens = "LiteLLM_DailyEndUserSpend".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
			cache_creation_input_tokens = "LiteLLM_DailyEndUserSpend".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
			spend = "LiteLLM_DailyEndUserSpend".spend + EXCLUDED.spend,
			api_requests = "LiteLLM_DailyEndUserSpend".api_requests + EXCLUDED.api_requests,
			successful_requests = "LiteLLM_DailyEndUserSpend".successful_requests + EXCLUDED.successful_requests,
			failed_requests = "LiteLLM_DailyEndUserSpend".failed_requests + EXCLUDED.failed_requests,
			updated_at = now()
	`

	// QueryUpsertDailyTagSpend upserts into LiteLLM_DailyTagSpend
	QueryUpsertDailyTagSpend = `
		INSERT INTO "LiteLLM_DailyTagSpend" (
			id,
			tag,
			request_id,
			date,
			api_key,
			model,
			model_group,
			custom_llm_provider,
			mcp_namespaced_tool_name,
			endpoint,
			prompt_tokens,
			completion_tokens,
			cache_read_input_tokens,
			cache_creation_input_tokens,
			spend,
			api_requests,
			successful_requests,
			failed_requests,
			created_at,
			updated_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, now(), now())
		ON CONFLICT (tag, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
		DO UPDATE SET
			model_group = EXCLUDED.model_group,
			prompt_tokens = "LiteLLM_DailyTagSpend".prompt_tokens + EXCLUDED.prompt_tokens,
			completion_tokens = "LiteLLM_DailyTagSpend".completion_tokens + EXCLUDED.completion_tokens,
			cache_read_input_tokens = "LiteLLM_DailyTagSpend".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
			cache_creation_input_tokens = "LiteLLM_DailyTagSpend".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
			spend = "LiteLLM_DailyTagSpend".spend + EXCLUDED.spend,
			api_requests = "LiteLLM_DailyTagSpend".api_requests + EXCLUDED.api_requests,
			successful_requests = "LiteLLM_DailyTagSpend".successful_requests + EXCLUDED.successful_requests,
			failed_requests = "LiteLLM_DailyTagSpend".failed_requests + EXCLUDED.failed_requests,
			updated_at = now()
	`
)

// Number of parameters per SpendLogEntry in batch insert
const SpendLogParamCount = 27
const (
	spendLogParamCount = SpendLogParamCount
)

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
			request_duration_ms, "completionStartTime",
			model, model_id, model_group, custom_llm_provider, api_base,
			"user", "metadata", cache_hit, cache_key, request_tags,
			team_id, organization_id, end_user, requester_ip_address,
			session_id, status,
			messages, response, proxy_server_request
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
			fmt.Fprintf(&b, "$%d", paramIdx)
			paramIdx++
		}
		// LiteLLM's payload-disabled writer persists empty JSON objects for these
		// privacy fields. AIR matches that shape without retaining request or
		// response content.
		b.WriteString(", '{}'::jsonb, '{}'::jsonb, '{}'::jsonb)")
	}

	b.WriteString(" ON CONFLICT (request_id) DO NOTHING RETURNING request_id")
	return b.String()
}
