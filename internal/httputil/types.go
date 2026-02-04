package httputil

// ProxyHealthResponse represents the JSON response from /health endpoint
type ProxyHealthResponse struct {
	Status               string                           `json:"status"`
	CredentialsAvailable int                              `json:"credentials_available"`
	CredentialsBanned    int                              `json:"credentials_banned"`
	TotalCredentials     int                              `json:"total_credentials"`
	Credentials          map[string]CredentialHealthStats `json:"credentials"`
	Models               map[string]ModelHealthStats      `json:"models"`
}

// CredentialHealthStats represents health stats for a single credential
type CredentialHealthStats struct {
	Type       string `json:"type"`
	IsFallback bool   `json:"is_fallback"`
	IsBanned   bool   `json:"is_banned"`
	CurrentRPM int    `json:"current_rpm"`
	CurrentTPM int    `json:"current_tpm"`
	LimitRPM   int    `json:"limit_rpm"`
	LimitTPM   int    `json:"limit_tpm"`
}

// ModelHealthStats represents health stats for a single model
type ModelHealthStats struct {
	Credential string `json:"credential"`
	Model      string `json:"model"`
	CurrentRPM int    `json:"current_rpm"`
	CurrentTPM int    `json:"current_tpm"`
	LimitRPM   int    `json:"limit_rpm"`
	LimitTPM   int    `json:"limit_tpm"`
}
