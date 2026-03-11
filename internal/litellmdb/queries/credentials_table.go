package queries

const QueryCredentialsTable = `SELECT credential_id, credential_name, credential_values, credential_info FROM public."LiteLLM_CredentialsTable"`

// CredentialLiteLLMParams содержит параметры аутентификации для различных провайдеров
type CredentialLiteLLMParams struct {
	APIKey  *string `json:"api_key,omitempty"`
	APIBase *string `json:"api_base,omitempty"`

	// VERTEX AI
	VertexProject     *string `json:"vertex_project,omitempty"`
	VertexLocation    *string `json:"vertex_location,omitempty"`
	VertexCredentials *string `json:"vertex_credentials,omitempty"`

	// Custom
	Model                 *string `json:"model,omitempty"`
	CustomLLMProviderName *string `json:"custom_llm_provider,omitempty"`
}

type CredentialLiteLLMInfo struct {
	CustomLLMProvider *string `json:"custom_llm_provider,omitempty"`
}

type CredentialTable struct {
	CredentialId     *string                  `json:"credential_id,omitempty"`
	CredentialName   *string                  `json:"credential_name,omitempty"`
	CredentialParams *CredentialLiteLLMParams `json:"credential_values,omitempty"`
	CredentialInfo   *CredentialLiteLLMInfo   `json:"credential_info,omitempty"`
}
