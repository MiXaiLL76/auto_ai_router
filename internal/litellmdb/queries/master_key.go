package queries

const QueryMasterKey = `SELECT param_value->'master_key' as master_key FROM public."LiteLLM_Config" WHERE "param_name"='general_settings';`
