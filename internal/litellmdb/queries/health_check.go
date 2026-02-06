package queries

// SQL queries for health check and table operations

const (
	// QueryGetConfig retrieves a config parameter
	QueryGetConfig = `
		SELECT param_name, param_value
		FROM "LiteLLM_Config"
		WHERE param_name = $1
	`

	// QueryGetGeneralSettings retrieves general_settings
	QueryGetGeneralSettings = `
		SELECT param_value
		FROM "LiteLLM_Config"
		WHERE param_name = 'general_settings'
	`

	// QueryHealthCheck is a simple connection check
	QueryHealthCheck = `SELECT 1`

	// QueryTableExists checks if a table exists
	QueryTableExists = `
		SELECT EXISTS (
			SELECT FROM information_schema.tables
			WHERE table_name = $1
		)
	`
)
