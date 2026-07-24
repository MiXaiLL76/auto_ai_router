package queries

// SQL statements for the atomic spend counter updates executed in the same
// transaction as the raw LiteLLM_SpendLogs insert (see
// internal/litellmdb/spendlog/spend_updater.go).

const (
	// QueryUpdateTokenSpendWithLastActive increments the scalar key spend and
	// bumps last_active (schemas that already have the last_active column).
	QueryUpdateTokenSpendWithLastActive = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = spend + $1, updated_at = NOW(), last_active = NOW()
		WHERE token = $2 AND spend IS NOT NULL`

	// QueryUpdateTokenModelSpendWithLastActive increments the scalar key spend
	// and the per-model JSON counter, and bumps last_active.
	QueryUpdateTokenModelSpendWithLastActive = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = spend + $1,
		    model_spend = jsonb_set(
		        COALESCE(model_spend, '{}'::jsonb),
		        ARRAY[$2]::text[],
		        to_jsonb(COALESCE((COALESCE(model_spend, '{}'::jsonb) ->> $2)::double precision, 0) + $1),
		        true
		    ),
		    updated_at = NOW(),
		    last_active = NOW()
		WHERE token = $3 AND spend IS NOT NULL`

	// QueryUpdateTokenSpend is the fallback for schemas without last_active.
	QueryUpdateTokenSpend = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = spend + $1, updated_at = NOW()
		WHERE token = $2 AND spend IS NOT NULL`

	// QueryUpdateTokenModelSpend is the model-counter fallback for schemas
	// without last_active.
	QueryUpdateTokenModelSpend = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = spend + $1,
		    model_spend = jsonb_set(
		        COALESCE(model_spend, '{}'::jsonb),
		        ARRAY[$2]::text[],
		        to_jsonb(COALESCE((COALESCE(model_spend, '{}'::jsonb) ->> $2)::double precision, 0) + $1),
		        true
		    ),
		    updated_at = NOW()
		WHERE token = $3 AND spend IS NOT NULL`

	// QueryUpdateEntitySpendTemplate builds an entity counter update without a
	// model dimension. Table and ID column are compile-time constants supplied
	// only by the spendlog wrappers.
	QueryUpdateEntitySpendTemplate = `
		UPDATE %s
		SET spend = spend + $1, updated_at = NOW()
		WHERE %s = $2 AND spend IS NOT NULL`

	// QueryUpdateEntityModelSpendTemplate builds an entity counter update where
	// scalar spend and the per-model JSON number change in one statement.
	QueryUpdateEntityModelSpendTemplate = `
		UPDATE %s
		SET spend = spend + $1,
		    model_spend = jsonb_set(
		        COALESCE(model_spend, '{}'::jsonb),
		        ARRAY[$2]::text[],
		        to_jsonb(COALESCE((COALESCE(model_spend, '{}'::jsonb) ->> $2)::double precision, 0) + $1),
		        true
		    ),
		    updated_at = NOW()
		WHERE %s = $3 AND spend IS NOT NULL`

	// QueryUpdateTeamMemberSpendWithTotal increments both spend and total_spend
	// (schemas that already have the total_spend column).
	QueryUpdateTeamMemberSpendWithTotal = `UPDATE "LiteLLM_TeamMembership" SET spend = spend + $1, total_spend = total_spend + $1 WHERE team_id = $2 AND user_id = $3 AND spend IS NOT NULL`

	// QueryUpdateTeamMemberSpend is the fallback for schemas without
	// total_spend.
	QueryUpdateTeamMemberSpend = `UPDATE "LiteLLM_TeamMembership" SET spend = spend + $1 WHERE team_id = $2 AND user_id = $3 AND spend IS NOT NULL`

	// QueryUpdateOrganizationMemberSpend increments the member's spend within
	// the organization.
	QueryUpdateOrganizationMemberSpend = `
		UPDATE "LiteLLM_OrganizationMembership"
		SET spend = spend + $1, updated_at = NOW()
		WHERE organization_id = $2 AND user_id = $3 AND spend IS NOT NULL`

	// QueryUpsertEndUserSpend inserts or increments the end user's spend.
	QueryUpsertEndUserSpend = `
		INSERT INTO "LiteLLM_EndUserTable" (user_id, spend)
		VALUES ($1, $2)
		ON CONFLICT (user_id) DO UPDATE
		SET spend = COALESCE("LiteLLM_EndUserTable".spend, 0) + EXCLUDED.spend`
)
