ALTER TABLE "LiteLLM_SpendLogs"
    ADD COLUMN IF NOT EXISTS credential_name TEXT;
