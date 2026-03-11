package queries

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestCredentialLiteLLMParamsFields verifies CredentialLiteLLMParams has all required fields
func TestCredentialLiteLLMParamsFields(t *testing.T) {
	apiKey := "sk-test-key"
	apiBase := "https://api.openai.com/v1"
	vertexProject := "my-project"
	vertexLocation := "us-central1"
	vertexCreds := `{"type":"service_account"}`
	model := "gpt-4"
	customProvider := "openai"

	params := CredentialLiteLLMParams{
		APIKey:                &apiKey,
		APIBase:               &apiBase,
		VertexProject:         &vertexProject,
		VertexLocation:        &vertexLocation,
		VertexCredentials:     &vertexCreds,
		Model:                 &model,
		CustomLLMProviderName: &customProvider,
	}

	assert.Equal(t, "sk-test-key", *params.APIKey)
	assert.Equal(t, "https://api.openai.com/v1", *params.APIBase)
	assert.Equal(t, "my-project", *params.VertexProject)
	assert.Equal(t, "us-central1", *params.VertexLocation)
	assert.Equal(t, `{"type":"service_account"}`, *params.VertexCredentials)
	assert.Equal(t, "gpt-4", *params.Model)
	assert.Equal(t, "openai", *params.CustomLLMProviderName)
}

// TestCredentialLiteLLMInfoFields verifies CredentialLiteLLMInfo structure
func TestCredentialLiteLLMInfoFields(t *testing.T) {
	provider := "vertex_ai"
	info := CredentialLiteLLMInfo{
		CustomLLMProvider: &provider,
	}

	assert.Equal(t, "vertex_ai", *info.CustomLLMProvider)
}

// TestCredentialTableFields verifies CredentialTable structure
func TestCredentialTableFields(t *testing.T) {
	credID := "cred-001"
	credName := "my-credential"
	apiKey := "sk-test"

	credIDPtr := &credID
	credNamePtr := &credName
	apiKeyPtr := &apiKey

	params := CredentialLiteLLMParams{
		APIKey: apiKeyPtr,
	}

	provider := "openai"
	info := CredentialLiteLLMInfo{
		CustomLLMProvider: &provider,
	}

	table := CredentialTable{
		CredentialId:     credIDPtr,
		CredentialName:   credNamePtr,
		CredentialParams: &params,
		CredentialInfo:   &info,
	}

	assert.Equal(t, "cred-001", *table.CredentialId)
	assert.Equal(t, "my-credential", *table.CredentialName)
	assert.NotNil(t, table.CredentialParams)
	assert.NotNil(t, table.CredentialInfo)
}

// TestCredentialTable_NilFields verifies CredentialTable with nil fields
func TestCredentialTable_NilFields(t *testing.T) {
	table := CredentialTable{}

	assert.Nil(t, table.CredentialId)
	assert.Nil(t, table.CredentialName)
	assert.Nil(t, table.CredentialParams)
	assert.Nil(t, table.CredentialInfo)
}

// TestCustomPricingLiteLLMParamsFields verifies CustomPricingLiteLLMParams structure
func TestCustomPricingLiteLLMParamsFields(t *testing.T) {
	inputCost := 0.00001
	outputCost := 0.00003
	inputAbove128k := 0.000008
	inputAbove200k := 0.000005
	outputAbove128k := 0.000024
	outputAbove200k := 0.000015

	inputCostPerSecond := 0.0001
	outputCostPerSecond := 0.0003

	cacheReadInput := 0.000001
	cacheReadInputAbove200k := 0.0000008

	inputAudio := 0.000006
	inputAudioPerSecond := 0.00006
	inputAudioAbove128k := 0.000004
	outputAudio := 0.000024
	outputAudioPerSecond := 0.00024

	inputVideoPerSecond := 0.000036
	inputVideoPerSecondAbove15s := 0.000018
	inputVideoPerSecondAbove8s := 0.000024
	outputVideoPerSecond := 0.000108

	inputImage := 0.00001
	inputImageAbove128k := 0.000005
	outputImage := 0.0000425
	outputImageToken := 0.000001
	outputReasoningToken := 0.000018

	params := CustomPricingLiteLLMParams{
		InputCostPerToken:                          &inputCost,
		OutputCostPerToken:                         &outputCost,
		InputCostPerTokenAbove128kTokens:           &inputAbove128k,
		InputCostPerTokenAbove200kTokens:           &inputAbove200k,
		OutputCostPerTokenAbove128kTokens:          &outputAbove128k,
		OutputCostPerTokenAbove200kTokens:          &outputAbove200k,
		InputCostPerSecond:                         &inputCostPerSecond,
		OutputCostPerSecond:                        &outputCostPerSecond,
		CacheReadInputTokenCost:                    &cacheReadInput,
		CacheReadInputTokenCostAbove200kTokens:     &cacheReadInputAbove200k,
		InputCostPerAudioToken:                     &inputAudio,
		InputCostPerAudioPerSecond:                 &inputAudioPerSecond,
		InputCostPerAudioPerSecondAbove128kTokens:  &inputAudioAbove128k,
		OutputCostPerAudioToken:                    &outputAudio,
		OutputCostPerAudioPerSecond:                &outputAudioPerSecond,
		InputCostPerVideoPerSecond:                 &inputVideoPerSecond,
		InputCostPerVideoPerSecondAbove15sInterval: &inputVideoPerSecondAbove15s,
		InputCostPerVideoPerSecondAbove8sInterval:  &inputVideoPerSecondAbove8s,
		OutputCostPerVideoPerSecond:                &outputVideoPerSecond,
		InputCostPerImage:                          &inputImage,
		InputCostPerImageAbove128kTokens:           &inputImageAbove128k,
		OutputCostPerImage:                         &outputImage,
		OutputCostPerImageToken:                    &outputImageToken,
		OutputCostPerReasoningToken:                &outputReasoningToken,
	}

	assert.NotNil(t, params.InputCostPerToken)
	assert.NotNil(t, params.OutputCostPerToken)
	assert.Equal(t, 0.00001, *params.InputCostPerToken)
	assert.Equal(t, 0.00003, *params.OutputCostPerToken)
	assert.Equal(t, 0.000018, *params.OutputCostPerReasoningToken)
}

// TestGenericLiteLLMParamsFields verifies GenericLiteLLMParams structure with embedded types
func TestGenericLiteLLMParamsFields(t *testing.T) {
	apiKey := "sk-test"
	model := "gpt-4"
	provider := "openai"
	tpm := 1000
	rpm := 100

	tpmVal := &tpm
	rpmVal := &rpm

	// Embed credential params
	credParams := CredentialLiteLLMParams{
		APIKey: &apiKey,
		Model:  &model,
	}

	// Create pricing params
	inputCost := 0.00001
	pricingParams := CustomPricingLiteLLMParams{
		InputCostPerToken: &inputCost,
	}

	// Build model_info
	modelInfo := map[string]interface{}{
		"mode":                      "chat",
		"supports_function_calling": true,
	}

	params := GenericLiteLLMParams{
		CredentialLiteLLMParams:    credParams,
		CustomPricingLiteLLMParams: pricingParams,
		CustomLLMProvider:          &provider,
		CredentialName:             nil,
		TPM:                        tpmVal,
		RPM:                        rpmVal,
		ModelInfo:                  modelInfo,
	}

	assert.Equal(t, "sk-test", *params.APIKey)
	assert.Equal(t, "gpt-4", *params.Model)
	assert.Equal(t, "openai", *params.CustomLLMProvider)
	assert.Equal(t, 1000, *params.TPM)
	assert.Equal(t, 100, *params.RPM)
	assert.NotNil(t, params.ModelInfo)
	assert.Equal(t, true, params.ModelInfo["supports_function_calling"])
}

// TestModelTableFields verifies ModelTable structure
func TestModelTableFields(t *testing.T) {
	modelID := "gpt-4"
	modelName := "gpt-4"
	modelInfo := map[string]interface{}{
		"mode": "chat",
	}

	modelIDPtr := &modelID
	modelNamePtr := &modelName

	// Create params
	apiKey := "sk-test"
	apiKeyPtr := &apiKey

	params := GenericLiteLLMParams{
		CredentialLiteLLMParams: CredentialLiteLLMParams{
			APIKey: apiKeyPtr,
		},
	}

	table := ModelTable{
		ModelId:   modelIDPtr,
		ModelName: modelNamePtr,
		LlmParams: &params,
		ModelInfo: modelInfo,
	}

	assert.Equal(t, "gpt-4", *table.ModelId)
	assert.Equal(t, "gpt-4", *table.ModelName)
	assert.NotNil(t, table.LlmParams)
	assert.NotNil(t, table.ModelInfo)
}

// TestModelTable_NilFields verifies ModelTable with nil fields
func TestModelTable_NilFields(t *testing.T) {
	table := ModelTable{}

	assert.Nil(t, table.ModelId)
	assert.Nil(t, table.ModelName)
	assert.Nil(t, table.LlmParams)
	assert.Nil(t, table.ModelInfo)
}

// TestQueryConstantsExist verifies that required query constants are defined
func TestQueryConstantsExist(t *testing.T) {
	// Verify credential queries exist
	assert.NotEmpty(t, QueryCredentialsTable)
	assert.Contains(t, QueryCredentialsTable, "LiteLLM_CredentialsTable")

	// Verify master key query exists
	assert.NotEmpty(t, QueryMasterKey)
	assert.Contains(t, QueryMasterKey, "master_key")

	// Verify model table query exists
	assert.NotEmpty(t, QueryProxyModelTable)
	assert.Contains(t, QueryProxyModelTable, "LiteLLM_ProxyModelTable")
}

// TestSpendLogQueryConstants verifies spend log query constants
func TestSpendLogQueryConstants(t *testing.T) {
	// These are imported from spend_logs.go
	// Verify they are defined (we can't import const from another file directly in test)
	assert.NotEmpty(t, QueryInsertSpendLog)
	assert.NotEmpty(t, QuerySelectUnprocessedRequestIDs)
	assert.NotEmpty(t, QuerySelectUnprocessedSpendLogs)
	assert.NotEmpty(t, QueryUpsertDailyUserSpend)
	assert.NotEmpty(t, QueryUpsertDailyTeamSpend)
	assert.NotEmpty(t, QueryUpsertDailyOrganizationSpend)
	assert.NotEmpty(t, QueryUpsertDailyEndUserSpend)
	assert.NotEmpty(t, QueryUpsertDailyAgentSpend)
	assert.NotEmpty(t, QueryUpsertDailyTagSpend)
	assert.NotEmpty(t, QueryMarkSpendLogsAsProcessed)
}

// TestQueryContainsRequiredFields verifies queries contain required fields
func TestQueryContainsRequiredFields(t *testing.T) {
	// Verify credential table query
	assert.Contains(t, QueryCredentialsTable, "credential_id")
	assert.Contains(t, QueryCredentialsTable, "credential_name")
	assert.Contains(t, QueryCredentialsTable, "credential_values")
	assert.Contains(t, QueryCredentialsTable, "credential_info")

	// Verify model table query
	assert.Contains(t, QueryProxyModelTable, "model_id")
	assert.Contains(t, QueryProxyModelTable, "model_name")
	assert.Contains(t, QueryProxyModelTable, "litellm_params")
	assert.Contains(t, QueryProxyModelTable, "model_info")

	// Verify spend log queries contain required fields
	assert.Contains(t, QuerySelectUnprocessedSpendLogs, "request_id")
	assert.Contains(t, QuerySelectUnprocessedSpendLogs, "prompt_tokens")
	assert.Contains(t, QuerySelectUnprocessedSpendLogs, "completion_tokens")
	assert.Contains(t, QuerySelectUnprocessedSpendLogs, "spend")

	// Verify upsert queries
	assert.Contains(t, QueryUpsertDailyUserSpend, "user_id")
	assert.Contains(t, QueryUpsertDailyTeamSpend, "team_id")
	assert.Contains(t, QueryUpsertDailyOrganizationSpend, "organization_id")
	assert.Contains(t, QueryUpsertDailyEndUserSpend, "end_user_id")
	assert.Contains(t, QueryUpsertDailyAgentSpend, "agent_id")
	assert.Contains(t, QueryUpsertDailyTagSpend, "tag")
}
