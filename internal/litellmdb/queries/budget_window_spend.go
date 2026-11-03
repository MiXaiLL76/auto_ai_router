package queries

// LiteLLM_BudgetWindowSpend holds one row per key budget window
// (budget_limits entry): the spend of the window that starts at window_start.
// Rows are written in the spend log transaction; a missing row, or one left
// on an earlier window, is read as the sum of the key's accounted spend logs
// within the window instead.

// QuerySelectKeyBudgetWindowSpend reads a key's window rows.
const QuerySelectKeyBudgetWindowSpend = `
SELECT window_duration, window_start, spend
FROM "LiteLLM_BudgetWindowSpend"
WHERE entity_type = 'key' AND entity_id = $1
`

// QuerySumKeyWindowSpendLogs sums a key's spend logs in [$2, $3). Rows the
// writer keeps out of the spend projections (accounting_eligible false) are
// left out here too; a row without the field counts.
const QuerySumKeyWindowSpendLogs = `
SELECT COALESCE(SUM(spend), 0)
FROM "LiteLLM_SpendLogs"
WHERE api_key = $1
  AND "startTime" >= $2
  AND "startTime" < $3
  AND (metadata #> '{spend_logs_metadata,accounting_eligible}') IS DISTINCT FROM 'false'::jsonb
`

// QuerySelectKeysBudgetWindows reads, for each key of a spend batch, its
// budget_limits and the windows its rows are on. budget_limits is read
// through the row's JSON like QueryValidateTokenWithHierarchy does.
const QuerySelectKeysBudgetWindows = `
SELECT t.token, to_jsonb(t) -> 'budget_limits', w.window_duration, w.window_start
FROM "LiteLLM_VerificationToken" t
LEFT JOIN "LiteLLM_BudgetWindowSpend" w
  ON w.entity_type = 'key' AND w.entity_id = t.token
WHERE t.token = ANY($1)
`

// QueryAddKeyBudgetWindowSpend adds spend to a key's window row that is on the
// window starting at $3.
const QueryAddKeyBudgetWindowSpend = `
UPDATE "LiteLLM_BudgetWindowSpend"
SET spend = spend + $4, updated_at = NOW()
WHERE entity_type = 'key' AND entity_id = $1 AND window_duration = $2 AND window_start = $3
`

// QuerySetKeyBudgetWindowSpend moves a key's window row onto the window
// starting at $3 with spend $4, creating the row if it does not exist.
const QuerySetKeyBudgetWindowSpend = `
INSERT INTO "LiteLLM_BudgetWindowSpend"
  (entity_type, entity_id, window_duration, window_start, spend, updated_at)
VALUES ('key', $1, $2, $3, $4, NOW())
ON CONFLICT (entity_type, entity_id, window_duration) DO UPDATE
SET window_start = EXCLUDED.window_start, spend = EXCLUDED.spend, updated_at = NOW()
`
