package queries

// SQL query for comprehensive token validation with budget hierarchy
// Loads all related data in ONE query instead of 5-7 separate queries
// Uses PostgreSQL JOINs and COALESCE for organization_id resolution

const QueryValidateTokenWithHierarchy = `
-- CTE to resolve organization_id (from token or team)
WITH org_id_resolved AS (
  SELECT
    t.token,
    COALESCE(t.organization_id, tm.organization_id) as resolved_org_id
  FROM "LiteLLM_VerificationToken" t
  LEFT JOIN "LiteLLM_TeamTable" tm ON t.team_id = tm.team_id
  WHERE t.token = $1
)

-- Main query with all JOINs
SELECT
  -- ============ Token ============
  t.token,
  t.key_name,
  t.key_alias,
  t.user_id,
  t.team_id,
  t.organization_id,
  t.spend as token_spend,
  t.max_budget as token_max_budget,
  t.tpm_limit as token_tpm_limit,
  t.rpm_limit as token_rpm_limit,
  t.expires,
  t.blocked as token_blocked,

  -- ============ User ============
  u.user_id as user_id_check,
  u.user_alias,
  u.user_email,
  u.max_budget as user_max_budget,
  u.spend as user_spend,

  -- ============ Team ============
  tm.team_id as team_id_check,
  tm.team_alias,
  tm.organization_id as team_organization_id,
  tm.max_budget as team_max_budget,
  tm.spend as team_spend,
  tm.blocked as team_blocked,
  tm.tpm_limit as team_tpm_limit,
  tm.rpm_limit as team_rpm_limit,

  -- ============ Organization ============
  o.organization_id as org_id_check,
  o.spend as org_spend,
  b_org.max_budget as org_max_budget,
  b_org.tpm_limit as org_tpm_limit,
  b_org.rpm_limit as org_rpm_limit,

  -- ============ TeamMembership ============
  tmem.spend as team_member_spend,
  b_tmem.max_budget as team_member_max_budget,
  b_tmem.tpm_limit as team_member_tpm_limit,
  b_tmem.rpm_limit as team_member_rpm_limit,

  -- ============ OrganizationMembership ============
  omem.spend as org_member_spend,
  b_omem.max_budget as org_member_max_budget,
  b_omem.tpm_limit as org_member_tpm_limit,
  b_omem.rpm_limit as org_member_rpm_limit

FROM "LiteLLM_VerificationToken" t

-- Join User (optional - if user_id exists)
LEFT JOIN "LiteLLM_UserTable" u ON t.user_id = u.user_id

-- Join Team (optional - if team_id exists)
LEFT JOIN "LiteLLM_TeamTable" tm ON t.team_id = tm.team_id

-- Join Organization
-- Organization_id resolved from: token.organization_id OR team.organization_id
LEFT JOIN "LiteLLM_OrganizationTable" o ON
  COALESCE(t.organization_id, tm.organization_id) = o.organization_id

-- Join Organization's Budget (external budget)
LEFT JOIN "LiteLLM_BudgetTable" b_org ON o.budget_id = b_org.budget_id

-- Join TeamMembership (user within team)
-- Only if both user_id AND team_id exist
LEFT JOIN "LiteLLM_TeamMembership" tmem ON
  t.user_id IS NOT NULL
  AND t.team_id IS NOT NULL
  AND t.user_id = tmem.user_id
  AND t.team_id = tmem.team_id

-- Join TeamMembership's Budget (external budget)
LEFT JOIN "LiteLLM_BudgetTable" b_tmem ON tmem.budget_id = b_tmem.budget_id

-- Join OrganizationMembership (user within organization)
-- Only if both user_id AND organization_id exist (resolved)
LEFT JOIN "LiteLLM_OrganizationMembership" omem ON
  t.user_id IS NOT NULL
  AND COALESCE(t.organization_id, tm.organization_id) IS NOT NULL
  AND t.user_id = omem.user_id
  AND COALESCE(t.organization_id, tm.organization_id) = omem.organization_id

-- Join OrganizationMembership's Budget (external budget)
LEFT JOIN "LiteLLM_BudgetTable" b_omem ON omem.budget_id = b_omem.budget_id

WHERE t.token = $1
`
