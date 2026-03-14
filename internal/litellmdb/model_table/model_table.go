package modeltable

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	cryptoutils "github.com/mixaill76/auto_ai_router/internal/litellmdb/crypto_utils"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"

	manager "github.com/mixaill76/auto_ai_router/internal/models"
)

// ProxyModelTable
// Synchronous (blocking) - token validation must complete before request processing
type ProxyModelTable struct {
	pool   *connection.ConnectionPool
	logger *slog.Logger
}

// NewProxyModelTable creates a new authenticator
func NewProxyModelTable(pool *connection.ConnectionPool, logger *slog.Logger) *ProxyModelTable {
	return &ProxyModelTable{
		pool:   pool,
		logger: logger,
	}
}

func (a *ProxyModelTable) FetchModels(ctx context.Context) ([]queries.ModelTable, error) {
	if !a.pool.IsHealthy() {
		return nil, models.ErrConnectionFailed
	}

	conn, err := a.pool.Acquire(ctx)
	if err != nil {
		a.logger.Error("Failed to acquire connection", "error", err)
		return nil, models.ErrConnectionFailed
	}
	defer conn.Release()

	rows, err := conn.Query(ctx, queries.QueryProxyModelTable)
	if err != nil {
		a.logger.Error("Failed to execute QueryProxyModelTable", "error", err)
		return nil, err
	}
	defer rows.Close()

	var results []queries.ModelTable

	for rows.Next() {
		var m queries.ModelTable
		err := rows.Scan(
			&m.ModelId,
			&m.ModelName,
			&m.LlmParams,
			&m.ModelInfo,
		)
		if err != nil {
			a.logger.Error("Failed to scan row", "error", err)
			continue
		}
		results = append(results, m)
	}

	if err = rows.Err(); err != nil {
		return nil, err
	}

	a.logger.Info("Models loaded from DB", "count", len(results))
	return results, nil
}

func (a *ProxyModelTable) FetchCredentials(ctx context.Context) ([]queries.CredentialTable, error) {
	if !a.pool.IsHealthy() {
		return nil, models.ErrConnectionFailed
	}

	conn, err := a.pool.Acquire(ctx)
	if err != nil {
		a.logger.Error("Failed to acquire connection", "error", err)
		return nil, models.ErrConnectionFailed
	}
	defer conn.Release()

	rows, err := conn.Query(ctx, queries.QueryCredentialsTable)
	if err != nil {
		a.logger.Error("Failed to execute QueryCredentialsTable", "error", err)
		return nil, err
	}
	defer rows.Close()

	var results []queries.CredentialTable

	for rows.Next() {
		var m queries.CredentialTable
		err := rows.Scan(
			&m.CredentialId,
			&m.CredentialName,
			&m.CredentialParams,
			&m.CredentialInfo,
		)
		if err != nil {
			a.logger.Error("Failed to scan row", "error", err)
			continue
		}
		results = append(results, m)
	}

	if err = rows.Err(); err != nil {
		return nil, err
	}

	a.logger.Info("Credentials loaded from DB", "count", len(results))
	return results, nil
}

func (a *ProxyModelTable) FetchModelsForAIR(ctx context.Context, signingKey string) ([]config.CredentialConfig, []config.ModelRPMConfig, map[string]*manager.ModelPrice, error) {
	creds, err := a.FetchCredentials(ctx)
	if err != nil {
		a.logger.Error("Failed to FetchCredentials", "error", err)
		return nil, nil, nil, err
	}
	dbModels, err := a.FetchModels(ctx)
	if err != nil {
		a.logger.Error("Failed to FetchModels", "error", err)
		return nil, nil, nil, err
	}

	// Decrypt named credentials
	for i := range creds {
		if creds[i].CredentialParams == nil {
			continue
		}
		if err := cryptoutils.DecryptCredentialLiteLLMParams(creds[i].CredentialParams, signingKey); err != nil {
			a.logger.Warn("Failed to decrypt credential params",
				"credential", derefStr(creds[i].CredentialName, "<nil>"),
				"error", err,
			)
		}
	}

	// Decrypt inline model credentials.
	// Note: GenericLiteLLMParams.CustomLLMProvider has the same JSON tag as the embedded
	// CredentialLiteLLMParams.CustomLLMProviderName. Go JSON picks the outer field, so
	// CustomLLMProviderName is always nil after unmarshal and DecryptCredentialLiteLLMParams
	// skips it. We must decrypt the outer CustomLLMProvider separately.
	for i := range dbModels {
		if dbModels[i].LlmParams == nil {
			continue
		}
		if err := cryptoutils.DecryptCredentialLiteLLMParams(&dbModels[i].LlmParams.CredentialLiteLLMParams, signingKey); err != nil {
			a.logger.Warn("Failed to decrypt model inline credential",
				"model", derefStr(dbModels[i].ModelName, "<nil>"),
				"error", err,
			)
		}
		// Decrypt outer CustomLLMProvider (shadowed by embedded field in JSON unmarshal).
		p := dbModels[i].LlmParams
		if p.CustomLLMProvider != nil && *p.CustomLLMProvider != "" {
			decrypted, err := cryptoutils.DecryptValueHelper(*p.CustomLLMProvider, "custom_llm_provider", signingKey)
			if err != nil {
				a.logger.Warn("Failed to decrypt model custom_llm_provider",
					"model", derefStr(dbModels[i].ModelName, "<nil>"),
					"error", err,
				)
			} else {
				p.CustomLLMProvider = &decrypted
			}
		}
	}

	// Build named credential map and list
	credByName := make(map[string]bool)
	var airCredentials []config.CredentialConfig

	for _, cred := range creds {
		if cred.CredentialName == nil {
			continue
		}
		cfg := convertCredentialTableToConfig(cred)
		if cfg.Type == "" {
			a.logger.Warn("Skipping credential with unsupported provider",
				"credential", derefStr(cred.CredentialName, "<nil>"),
			)
			continue
		}
		if credByName[*cred.CredentialName] {
			a.logger.Warn("Duplicate credential name in DB, skipping",
				"credential", *cred.CredentialName,
			)
			continue
		}
		credByName[*cred.CredentialName] = true
		airCredentials = append(airCredentials, cfg)
	}

	// Process models → RPM configs, inline credentials, prices
	var airModels []config.ModelRPMConfig
	airPrices := make(map[string]*manager.ModelPrice)

	for _, model := range dbModels {
		if model.ModelName == nil || model.LlmParams == nil {
			continue
		}
		modelName := *model.ModelName

		// Determine which credential this model uses
		var credName string
		if model.LlmParams.CredentialName != nil && *model.LlmParams.CredentialName != "" {
			credName = *model.LlmParams.CredentialName
			if !credByName[credName] {
				a.logger.Warn("Model references unknown credential",
					"model", modelName,
					"credential", credName,
				)
				continue
			}
		} else if hasInlineCredentials(&model.LlmParams.CredentialLiteLLMParams) {
			// Create synthetic credential from model inline params
			syntheticName := fmt.Sprintf("db-model-%s", derefStr(model.ModelId, modelName))
			if !credByName[syntheticName] {
				syntheticCred := convertInlineCredToConfig(syntheticName, model.LlmParams)
				if syntheticCred.Type == "" {
					a.logger.Warn("Skipping model with unsupported inline provider",
						"model", modelName,
					)
					continue
				}
				credByName[syntheticName] = true
				airCredentials = append(airCredentials, syntheticCred)
			}
			credName = syntheticName
		}

		// Build ModelRPMConfig
		rpmCfg := config.ModelRPMConfig{
			Name:       modelName,
			Credential: credName,
		}
		if model.LlmParams.RPM != nil {
			rpmCfg.RPM = *model.LlmParams.RPM

		}
		if rpmCfg.RPM == 0 {
			rpmCfg.RPM = -1
		}
		if model.LlmParams.TPM != nil {
			rpmCfg.TPM = *model.LlmParams.TPM
		}
		if rpmCfg.TPM == 0 {
			rpmCfg.TPM = -1
		}
		// Map real provider model name (e.g. "gemini-2.0-flash" → "vertex_ai/gemini-2.0-flash")
		if model.LlmParams.Model != nil && *model.LlmParams.Model != "" && *model.LlmParams.Model != modelName {
			rpmCfg.Model = *model.LlmParams.Model
		}
		airModels = append(airModels, rpmCfg)

		// Build ModelPrice from CustomPricingLiteLLMParams
		if price := convertPricingToModelPrice(&model.LlmParams.CustomPricingLiteLLMParams); price != nil {
			airPrices[manager.NormalizeModelName(modelName)] = price
		}
	}

	a.logger.Info("FetchModelsForAIR completed",
		"credentials", len(airCredentials),
		"models", len(airModels),
		"prices", len(airPrices),
	)

	return airCredentials, airModels, airPrices, nil
}

// ==================== Helper functions ====================

func derefStr(s *string, fallback string) string {
	if s != nil {
		return *s
	}
	return fallback
}

// mapProviderType converts a LiteLLM custom_llm_provider string to config.ProviderType
func mapProviderType(provider string) config.ProviderType {
	p := strings.ToLower(provider)
	switch {
	case strings.Contains(p, "openai") || strings.Contains(p, "router"):
		return config.ProviderTypeOpenAI
	case strings.Contains(p, "vertex"):
		return config.ProviderTypeVertexAI
	case strings.Contains(p, "google"):
		return config.ProviderTypeGemini
	case strings.Contains(p, "xai"):
		return config.ProviderTypeOpenAI
	default:
		// Unknown/unsupported providers are intentionally dropped for now.
		return ""
	}
}

// fillCredentialFromParams fills a CredentialConfig from CredentialLiteLLMParams
func fillCredentialFromParams(cfg *config.CredentialConfig, params *queries.CredentialLiteLLMParams) {
	if params == nil {
		return
	}
	if params.APIKey != nil {
		cfg.APIKey = *params.APIKey
	}
	if params.APIBase != nil {
		cfg.BaseURL = *params.APIBase
	}
	if params.VertexProject != nil {
		cfg.ProjectID = *params.VertexProject
	}
	if params.VertexLocation != nil {
		cfg.Location = *params.VertexLocation
	}
	if params.VertexCredentials != nil {
		cfg.CredentialsJSON = *params.VertexCredentials
	}
}

// convertCredentialTableToConfig converts a DB CredentialTable row to config.CredentialConfig
func convertCredentialTableToConfig(cred queries.CredentialTable) config.CredentialConfig {
	cfg := config.CredentialConfig{RPM: -1, TPM: -1}

	if cred.CredentialName != nil {
		cfg.Name = *cred.CredentialName
	}

	// Provider type from credential_info
	if cred.CredentialInfo != nil && cred.CredentialInfo.CustomLLMProvider != nil {
		cfg.Type = mapProviderType(*cred.CredentialInfo.CustomLLMProvider)
	}

	fillCredentialFromParams(&cfg, cred.CredentialParams)

	return cfg
}

// convertInlineCredToConfig creates a CredentialConfig from model inline params
func convertInlineCredToConfig(name string, params *queries.GenericLiteLLMParams) config.CredentialConfig {
	cfg := config.CredentialConfig{Name: name, RPM: -1, TPM: -1}

	if params == nil {
		return cfg
	}

	// Determine provider type: prefer top-level CustomLLMProvider, then embedded one
	providerName := ""
	if params.CustomLLMProvider != nil && *params.CustomLLMProvider != "" {
		providerName = *params.CustomLLMProvider
	} else if params.CustomLLMProviderName != nil && *params.CustomLLMProviderName != "" {
		providerName = *params.CustomLLMProviderName
	}
	if providerName != "" {
		cfg.Type = mapProviderType(providerName)
	}

	fillCredentialFromParams(&cfg, &params.CredentialLiteLLMParams)
	return cfg
}

// hasInlineCredentials returns true if the params have any non-empty auth credentials
func hasInlineCredentials(params *queries.CredentialLiteLLMParams) bool {
	if params == nil {
		return false
	}
	return (params.APIKey != nil && *params.APIKey != "") ||
		(params.VertexProject != nil && *params.VertexProject != "") ||
		(params.VertexCredentials != nil && *params.VertexCredentials != "")
}

// convertPricingToModelPrice converts CustomPricingLiteLLMParams to a ModelPrice.
// Returns nil if no pricing data is present.
func convertPricingToModelPrice(p *queries.CustomPricingLiteLLMParams) *manager.ModelPrice {
	if p == nil {
		return nil
	}
	if p.InputCostPerToken == nil && p.OutputCostPerToken == nil {
		return nil
	}

	price := &manager.ModelPrice{}
	if p.InputCostPerToken != nil {
		price.InputCostPerToken = *p.InputCostPerToken
	}
	if p.OutputCostPerToken != nil {
		price.OutputCostPerToken = *p.OutputCostPerToken
	}
	if p.InputCostPerTokenAbove200kTokens != nil {
		price.InputCostPerTokenAbove200k = *p.InputCostPerTokenAbove200kTokens
	}
	if p.OutputCostPerTokenAbove200kTokens != nil {
		price.OutputCostPerTokenAbove200k = *p.OutputCostPerTokenAbove200kTokens
	}
	if p.InputCostPerAudioToken != nil {
		price.InputCostPerAudioToken = *p.InputCostPerAudioToken
	}
	if p.OutputCostPerAudioToken != nil {
		price.OutputCostPerAudioToken = *p.OutputCostPerAudioToken
	}
	if p.OutputCostPerReasoningToken != nil {
		price.OutputCostPerReasoningToken = *p.OutputCostPerReasoningToken
	}
	if p.CacheReadInputTokenCost != nil {
		price.InputCostPerCachedToken = *p.CacheReadInputTokenCost
	}
	if p.OutputCostPerImage != nil {
		price.OutputCostPerImage = *p.OutputCostPerImage
	}
	if p.OutputCostPerImageToken != nil {
		price.OutputCostPerImageToken = *p.OutputCostPerImageToken
	}

	return price
}
