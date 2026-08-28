package queries

// SQL statements for the atomic spend counter updates executed in the same
// transaction as the raw LiteLLM_SpendLogs insert (see
// internal/litellmdb/spendlog/spend_updater.go).

const (
	// QueryUpdateTokenSpendWithLastActive increments the scalar key spend and
	// bumps last_active (schemas that already have the last_active column).
	//nolint:gosec // G101: false positive — SQL statement text, not a credential
	QueryUpdateTokenSpendWithLastActive = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = COALESCE(spend, 0) + $1, updated_at = NOW(), last_active = NOW()
		WHERE token = $2`

	// QueryUpdateTokenModelSpendWithLastActive increments the scalar key spend
	// and the per-model JSON counter, and bumps last_active.
	//nolint:gosec // G101: false positive — SQL statement text, not a credential
	QueryUpdateTokenModelSpendWithLastActive = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = COALESCE(spend, 0) + $1,
		    model_spend = jsonb_set(
		        COALESCE(model_spend, '{}'::jsonb),
		        ARRAY[$2]::text[],
		        to_jsonb(COALESCE((COALESCE(model_spend, '{}'::jsonb) ->> $2)::double precision, 0) + $1),
		        true
		    ),
		    updated_at = NOW(),
		    last_active = NOW()
		WHERE token = $3`

	// QueryUpdateTokenSpend is the fallback for schemas without last_active.
	//nolint:gosec // G101: false positive — SQL statement text, not a credential
	QueryUpdateTokenSpend = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = COALESCE(spend, 0) + $1, updated_at = NOW()
		WHERE token = $2`

	// QueryUpdateTokenModelSpend is the model-counter fallback for schemas
	// without last_active.
	//nolint:gosec // G101: false positive — SQL statement text, not a credential
	QueryUpdateTokenModelSpend = `
		UPDATE "LiteLLM_VerificationToken"
		SET spend = COALESCE(spend, 0) + $1,
		    model_spend = jsonb_set(
		        COALESCE(model_spend, '{}'::jsonb),
		        ARRAY[$2]::text[],
		        to_jsonb(COALESCE((COALESCE(model_spend, '{}'::jsonb) ->> $2)::double precision, 0) + $1),
		        true
		    ),
		    updated_at = NOW()
		WHERE token = $3`

	// QueryUpdateEntitySpendTemplate builds an entity counter update without a
	// model dimension. Table and ID column are compile-time constants supplied
	// only by the spendlog wrappers.
	QueryUpdateEntitySpendTemplate = `
		UPDATE %s
		SET spend = COALESCE(spend, 0) + $1, updated_at = NOW()
		WHERE %s = $2`

	// QueryUpdateEntityModelSpendTemplate builds an entity counter update where
	// scalar spend and the per-model JSON number change in one statement.
	QueryUpdateEntityModelSpendTemplate = `
		UPDATE %s
		SET spend = COALESCE(spend, 0) + $1,
		    model_spend = jsonb_set(
		        COALESCE(model_spend, '{}'::jsonb),
		        ARRAY[$2]::text[],
		        to_jsonb(COALESCE((COALESCE(model_spend, '{}'::jsonb) ->> $2)::double precision, 0) + $1),
		        true
		    ),
		    updated_at = NOW()
		WHERE %s = $3`

	// QueryUpdateTeamMemberSpendWithTotal increments both spend and total_spend
	// (schemas that already have the total_spend column).
	QueryUpdateTeamMemberSpendWithTotal = `UPDATE "LiteLLM_TeamMembership" SET spend = COALESCE(spend, 0) + $1, total_spend = COALESCE(total_spend, 0) + $1 WHERE team_id = $2 AND user_id = $3`

	// QueryUpdateTeamMemberSpend is the fallback for schemas without
	// total_spend.
	QueryUpdateTeamMemberSpend = `UPDATE "LiteLLM_TeamMembership" SET spend = COALESCE(spend, 0) + $1 WHERE team_id = $2 AND user_id = $3`

	// QueryUpdateOrganizationMemberSpend increments the member's spend within
	// the organization.
	QueryUpdateOrganizationMemberSpend = `
		UPDATE "LiteLLM_OrganizationMembership"
		SET spend = COALESCE(spend, 0) + $1, updated_at = NOW()
		WHERE organization_id = $2 AND user_id = $3`

	// QueryUpsertEndUserSpend inserts or increments the end user's spend.
	QueryUpsertEndUserSpend = `
		INSERT INTO "LiteLLM_EndUserTable" (user_id, spend)
		VALUES ($1, $2)
		ON CONFLICT (user_id) DO UPDATE
		SET spend = COALESCE("LiteLLM_EndUserTable".spend, 0) + EXCLUDED.spend`
)
