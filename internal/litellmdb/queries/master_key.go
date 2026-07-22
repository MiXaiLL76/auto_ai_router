package queries

// ->> (not ->) so the key arrives as text: -> returns a jsonb value that pgx
// scans into a string with the JSON quotes kept ("sk-..."), which would break
// the sk- prefix check in HashToken and silently disable the DB master key.
const QueryMasterKey = `SELECT param_value->>'master_key' as master_key FROM public."LiteLLM_Config" WHERE "param_name"='general_settings';`
