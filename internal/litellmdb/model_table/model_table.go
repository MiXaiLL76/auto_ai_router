// Package modeltable reads model deployments from the LiteLLM database.
package modeltable

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	cryptoutils "github.com/mixaill76/auto_ai_router/internal/litellmdb/crypto_utils"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/scope"

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

	results, err := a.queryModels(ctx, conn, queries.QueryProxyModelTableWithBlocked, true)
	if isUndefinedColumn(err) {
		// Schemas from before LiteLLM added the blocked column.
		a.logger.Debug("LiteLLM_ProxyModelTable has no blocked column, reading without it")
		results, err = a.queryModels(ctx, conn, queries.QueryProxyModelTable, false)
	}
	if err != nil {
		return nil, err
	}

	a.logger.Info("Models loaded from DB", "count", len(results))
	return results, nil
}

func (a *ProxyModelTable) queryModels(ctx context.Context, conn *pgxpool.Conn, query string, withBlocked bool) ([]queries.ModelTable, error) {
	rows, err := conn.Query(ctx, query)
	if err != nil {
		if !isUndefinedColumn(err) {
			a.logger.Error("Failed to execute QueryProxyModelTable", "error", err)
		}
		return nil, err
	}
	defer rows.Close()

	var results []queries.ModelTable

	for rows.Next() {
		var m queries.ModelTable
		var blocked *bool
		targets := []any{&m.ModelID, &m.ModelName, &m.LlmParams, &m.ModelInfo}
		if withBlocked {
			targets = append(targets, &blocked)
		}
		if err := rows.Scan(targets...); err != nil {
			a.logger.Error("Failed to scan row", "error", err)
			continue
		}
		m.Blocked = blocked != nil && *blocked
		results = append(results, m)
	}

	if err = rows.Err(); err != nil {
		if !isUndefinedColumn(err) {
			a.logger.Error("Failed to read QueryProxyModelTable rows", "error", err)
		}
		return nil, err
	}
	return results, nil
}

// isUndefinedColumn reports whether err is PostgreSQL's undefined_column (42703).
func isUndefinedColumn(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42703"
}

// FetchRouterSettings loads model_group_alias and fallbacks from
// LiteLLM_Config.router_settings. A database without that row yields empty settings.
func (a *ProxyModelTable) FetchRouterSettings(ctx context.Context) (queries.RouterSettings, error) {
	if !a.pool.IsHealthy() {
		return queries.RouterSettings{}, models.ErrConnectionFailed
	}

	conn, err := a.pool.Acquire(ctx)
	if err != nil {
		a.logger.Error("Failed to acquire connection", "error", err)
		return queries.RouterSettings{}, models.ErrConnectionFailed
	}
	defer conn.Release()

	var raw []byte
	rows, err := conn.Query(ctx, queries.QueryRouterSettings)
	if err != nil {
		a.logger.Error("Failed to execute QueryRouterSettings", "error", err)
		return queries.RouterSettings{}, err
	}
	defer rows.Close()
	if rows.Next() {
		if err := rows.Scan(&raw); err != nil {
			a.logger.Error("Failed to scan router_settings", "error", err)
			return queries.RouterSettings{}, err
		}
	}
	if err := rows.Err(); err != nil {
		return queries.RouterSettings{}, err
	}

	return queries.ParseRouterSettings(raw)
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
			&m.CredentialID,
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
	// A failed read must fail the whole sync: applying models without their aliases
	// would make every aliased name 404 until the next successful cycle.
	router, err := a.FetchRouterSettings(ctx)
	if err != nil {
		a.logger.Error("Failed to FetchRouterSettings", "error", err)
		return nil, nil, nil, err
	}

	airCredentials, airModels, airPrices := buildAIRModels(a.logger, creds, dbModels, router, signingKey)
	return airCredentials, airModels, airPrices, nil
}

// buildAIRModels turns rows read from the LiteLLM database into AIR credentials,
// per-credential model configs and prices. It performs no I/O so the whole
// conversion can be exercised against exported data.
func buildAIRModels(
	logger *slog.Logger,
	creds []queries.CredentialTable,
	dbModels []queries.ModelTable,
	router queries.RouterSettings,
	signingKey string,
) ([]config.CredentialConfig, []config.ModelRPMConfig, map[string]*manager.ModelPrice) {
	// Decrypt named credentials
	for i := range creds {
		if creds[i].CredentialParams == nil {
			continue
		}
		if err := cryptoutils.DecryptCredentialLiteLLMParams(creds[i].CredentialParams, signingKey); err != nil {
			logger.Warn("Failed to decrypt credential params",
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
			logger.Warn("Failed to decrypt model inline credential",
				"model", derefStr(dbModels[i].ModelName, "<nil>"),
				"error", err,
			)
		}
		// Decrypt outer CustomLLMProvider (shadowed by embedded field in JSON unmarshal).
		p := dbModels[i].LlmParams
		if p.CustomLLMProvider != nil && *p.CustomLLMProvider != "" {
			decrypted, err := cryptoutils.DecryptValueHelper(*p.CustomLLMProvider, "custom_llm_provider", signingKey)
			if err != nil {
				logger.Warn("Failed to decrypt model custom_llm_provider",
					"model", derefStr(dbModels[i].ModelName, "<nil>"),
					"error", err,
				)
			} else {
				p.CustomLLMProvider = &decrypted
			}
		}
		// LiteLLM stores the bound credential name encrypted. A plaintext name (written
		// by another tool) fails authentication and is kept as is.
		if p.LiteLLMCredentialName != nil && *p.LiteLLMCredentialName != "" {
			if decrypted, err := cryptoutils.DecryptValueHelper(*p.LiteLLMCredentialName, "litellm_credential_name", signingKey); err == nil {
				p.LiteLLMCredentialName = &decrypted
			}
		}
	}

	// A named credential whose credential_info carries no provider takes it from the
	// deployments that reference it (LiteLLM keeps custom_llm_provider on the model).
	providerByCredential := make(map[string]string)
	for _, model := range dbModels {
		if model.LlmParams == nil {
			continue
		}
		name := model.LlmParams.EffectiveCredentialName()
		if name == "" {
			continue
		}
		if provider := modelProviderName(model.LlmParams); provider != "" {
			if _, seen := providerByCredential[name]; !seen {
				providerByCredential[name] = provider
			}
		}
	}

	// Build named credential map and list
	credByName := make(map[string]bool)
	credTypeByName := make(map[string]config.ProviderType)
	var airCredentials []config.CredentialConfig

	for _, cred := range creds {
		if cred.CredentialName == nil {
			continue
		}
		cfg := convertCredentialTableToConfig(cred)
		if cfg.Type == "" {
			if provider, ok := providerByCredential[*cred.CredentialName]; ok {
				cfg.Type = mapProviderType(provider)
			}
		}
		if cfg.Type == "" {
			logger.Warn("Skipping credential with unsupported provider",
				"credential", derefStr(cred.CredentialName, "<nil>"),
			)
			continue
		}
		if credByName[*cred.CredentialName] {
			logger.Warn("Duplicate credential name in DB, skipping",
				"credential", *cred.CredentialName,
			)
			continue
		}
		credByName[*cred.CredentialName] = true
		credTypeByName[*cred.CredentialName] = cfg.Type
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

		if model.Blocked {
			logger.Info("Skipping blocked model", "model", modelName, "model_id", derefStr(model.ModelID, ""))
			continue
		}
		if model.Mode() == "rerank" {
			logger.Info("Skipping rerank model: /rerank is not supported", "model", modelName)
			continue
		}

		// Determine which credential this model uses
		var credName string
		if bound := model.LlmParams.EffectiveCredentialName(); bound != "" {
			credName = bound
			if !credByName[credName] {
				logger.Warn("Model references unknown credential",
					"model", modelName,
					"credential", credName,
				)
				continue
			}
		} else if hasInlineEndpoint(model.LlmParams) {
			// Create synthetic credential from model inline params
			syntheticName := fmt.Sprintf("db-model-%s", derefStr(model.ModelID, modelName))
			if !credByName[syntheticName] {
				syntheticCred := convertInlineCredToConfig(syntheticName, model.LlmParams)
				if syntheticCred.Type == "" {
					logger.Warn("Skipping model with unsupported inline provider",
						"model", modelName,
					)
					continue
				}
				credByName[syntheticName] = true
				credTypeByName[syntheticName] = syntheticCred.Type
				airCredentials = append(airCredentials, syntheticCred)
			}
			credName = syntheticName
		}

		// Build ModelRPMConfig
		rpmCfg := config.ModelRPMConfig{
			Name:         modelName,
			DeploymentID: derefStr(model.ModelID, ""),
			Credential:   credName,
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
		if model.LlmParams.Model != nil && *model.LlmParams.Model != "" {
			if realName := stripVLLMPrefix(*model.LlmParams.Model); realName != modelName {
				rpmCfg.Model = realName
			}
		}
		// Default request params are a vLLM-deployment feature only.
		if credTypeByName[credName] == config.ProviderTypeVLLM {
			rpmCfg.DefaultParams = model.LlmParams.DefaultRequestParams()
		}
		airModels = append(airModels, rpmCfg)

		// Build ModelPrice from CustomPricingLiteLLMParams
		if price := convertPricingToModelPrice(&model.LlmParams.CustomPricingLiteLLMParams); price != nil {
			price.LiteLLMProvider = pricingProviderName(model.LlmParams)
			// Key by the exact DB model name, not its normalized form: MergeDB
			// treats each key as an exact override (plus that key's own
			// lowercased form) specifically so a price for "gpt-5-mini" never
			// leaks into a distinctly-keyed "openrouter/gpt-5-mini" alias or
			// vice versa. Normalizing here would collapse both DB rows into
			// the same map key before MergeDB ever sees them, silently
			// dropping one price — defeating that protection entirely.
			airPrices[modelName] = price
		}
	}

	airModels = applyModelGroupAliases(logger, airModels, airPrices, router)

	logger.Info("FetchModelsForAIR completed",
		"credentials", len(airCredentials),
		"models", len(airModels),
		"prices", len(airPrices),
	)

	return airCredentials, airModels, airPrices
}

// applyModelGroupAliases exposes every LiteLLM router_settings.model_group_alias as a
// model of its own: each deployment of the target group is duplicated under the alias
// name, keeping the target's provider-facing model name, credential and default
// params. A real group with the alias's name wins over the alias, as in LiteLLM.
//
// router_settings.fallbacks (group-to-group failover) is deliberately not applied:
// AIR's failover works between credentials of one model, not between models, so
// importing it as credential tiers would misroute. It is logged at debug level so the
// gap is visible.
func applyModelGroupAliases(
	logger *slog.Logger,
	airModels []config.ModelRPMConfig,
	airPrices map[string]*manager.ModelPrice,
	router queries.RouterSettings,
) []config.ModelRPMConfig {
	if len(router.Fallbacks) > 0 {
		logger.Debug("router_settings.fallbacks is not imported (model-level failover unsupported)",
			"groups", len(router.Fallbacks))
	}
	if len(router.ModelGroupAlias) == 0 {
		return airModels
	}

	byGroup := make(map[string][]config.ModelRPMConfig, len(airModels))
	for _, m := range airModels {
		byGroup[m.Name] = append(byGroup[m.Name], m)
	}

	aliases := make([]string, 0, len(router.ModelGroupAlias))
	for alias := range router.ModelGroupAlias {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)

	for _, alias := range aliases {
		target := router.ModelGroupAlias[alias]
		if alias == target {
			continue
		}
		if _, exists := byGroup[alias]; exists {
			logger.Info("model_group_alias ignored: a model group with that name exists",
				"alias", alias, "target", target)
			continue
		}
		group := byGroup[target]
		if len(group) == 0 {
			logger.Warn("model_group_alias target has no usable deployments",
				"alias", alias, "target", target)
			continue
		}
		for _, m := range group {
			aliased := m
			aliased.Name = alias
			if aliased.Model == "" {
				aliased.Model = target
			}
			airModels = append(airModels, aliased)
		}
		if price, ok := airPrices[target]; ok {
			if _, has := airPrices[alias]; !has {
				airPrices[alias] = price
			}
		}
	}

	return airModels
}

// ==================== Helper functions ====================

func derefStr(s *string, fallback string) string {
	if s != nil {
		return *s
	}
	return fallback
}

// modelProviderName returns a deployment's own (decrypted) custom_llm_provider.
func modelProviderName(params *queries.GenericLiteLLMParams) string {
	if params == nil {
		return ""
	}
	if params.CustomLLMProvider != nil && *params.CustomLLMProvider != "" {
		return *params.CustomLLMProvider
	}
	if params.CustomLLMProviderName != nil {
		return *params.CustomLLMProviderName
	}
	return ""
}

// stripVLLMPrefix removes the LiteLLM provider prefix ("hosted_vllm/model") that the
// upstream vLLM server does not know. Only the vLLM prefixes are touched: other
// slashes are part of real model ids (for example "Qwen/Qwen3-8B").
func stripVLLMPrefix(model string) string {
	for _, prefix := range []string{"hosted_vllm/", "vllm/"} {
		if rest, ok := strings.CutPrefix(model, prefix); ok {
			return rest
		}
	}
	return model
}

// hasInlineEndpoint reports whether a deployment carries its own connection details.
// Besides secrets (hasInlineCredentials) a vLLM deployment counts with just an
// api_base: vLLM is routinely run without an API key.
func hasInlineEndpoint(params *queries.GenericLiteLLMParams) bool {
	if params == nil {
		return false
	}
	if hasInlineCredentials(&params.CredentialLiteLLMParams) {
		return true
	}
	return mapProviderType(modelProviderName(params)) == config.ProviderTypeVLLM &&
		params.APIBase != nil && *params.APIBase != ""
}

// mapProviderType converts a LiteLLM custom_llm_provider string to config.ProviderType
func mapProviderType(provider string) config.ProviderType {
	p := strings.ToLower(provider)
	switch {
	case p == "hosted_vllm" || p == "vllm" || p == "hosted-vllm":
		return config.ProviderTypeVLLM
	case p == "air" || p == "aar" || strings.Contains(p, "auto_ai_router") || strings.Contains(p, "auto-ai-router"):
		return config.ProviderTypeAIR
	case strings.Contains(p, "openai") || strings.Contains(p, "router"):
		return config.ProviderTypeOpenAI
	case strings.Contains(p, "vertex"):
		return config.ProviderTypeVertexAI
	case config.IsGoogleGeminiProvider(p):
		return config.ProviderTypeGemini
	case strings.Contains(p, "cometapi") || strings.Contains(p, "comet-api"):
		return config.ProviderTypeCometAPI
	case strings.Contains(p, "proman") || strings.Contains(p, "pro-man") || strings.Contains(p, "pro_man"):
		return config.ProviderTypeProMan
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
	if cred.CredentialInfo != nil {
		cfg.Scopes = scope.NormalizeList(cred.CredentialInfo.AirScopes)
		cfg.DeniedScopes = scope.NormalizeList(append(cred.CredentialInfo.AirDeniedScopes, cred.CredentialInfo.AirForbiddenScopes...))
		cfg.ReasoningOnly = cred.CredentialInfo.AirReasoningOnly
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
	if p.InputCostPerToken == nil && p.OutputCostPerToken == nil && p.OutputCostPerVideoPerSecond == nil && len(p.SearchContextCostPerQuery) == 0 {
		return nil
	}

	price := &manager.ModelPrice{}
	if p.InputCostPerToken != nil {
		price.InputCostPerToken = *p.InputCostPerToken
	}
	if p.OutputCostPerToken != nil {
		price.OutputCostPerToken = *p.OutputCostPerToken
	}
	if p.InputCostPerTokenAbove32kTokens != nil {
		price.InputCostPerTokenAbove32k = *p.InputCostPerTokenAbove32kTokens
	}
	if p.OutputCostPerTokenAbove32kTokens != nil {
		price.OutputCostPerTokenAbove32k = *p.OutputCostPerTokenAbove32kTokens
	}
	if p.InputCostPerTokenAbove128kTokens != nil {
		price.InputCostPerTokenAbove128k = *p.InputCostPerTokenAbove128kTokens
	}
	if p.OutputCostPerTokenAbove128kTokens != nil {
		price.OutputCostPerTokenAbove128k = *p.OutputCostPerTokenAbove128kTokens
	}
	if p.InputCostPerTokenAbove200kTokens != nil {
		price.InputCostPerTokenAbove200k = *p.InputCostPerTokenAbove200kTokens
	}
	if p.OutputCostPerTokenAbove200kTokens != nil {
		price.OutputCostPerTokenAbove200k = *p.OutputCostPerTokenAbove200kTokens
	}
	if p.InputCostPerTokenAbove256kTokens != nil {
		price.InputCostPerTokenAbove256k = *p.InputCostPerTokenAbove256kTokens
	}
	if p.OutputCostPerTokenAbove256kTokens != nil {
		price.OutputCostPerTokenAbove256k = *p.OutputCostPerTokenAbove256kTokens
	}
	if p.InputCostPerTokenAbove272kTokens != nil {
		price.InputCostPerTokenAbove272k = *p.InputCostPerTokenAbove272kTokens
	}
	if p.OutputCostPerTokenAbove272kTokens != nil {
		price.OutputCostPerTokenAbove272k = *p.OutputCostPerTokenAbove272kTokens
	}
	if p.InputCostPerTokenAbove512kTokens != nil {
		price.InputCostPerTokenAbove512k = *p.InputCostPerTokenAbove512kTokens
	}
	if p.OutputCostPerTokenAbove512kTokens != nil {
		price.OutputCostPerTokenAbove512k = *p.OutputCostPerTokenAbove512kTokens
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
	if p.CacheCreationInputTokenCost != nil {
		price.CacheCreationInputTokenCost = *p.CacheCreationInputTokenCost
	}
	if p.CacheReadInputTokenCostAbove32kTokens != nil {
		price.CacheReadInputTokenCostAbove32k = *p.CacheReadInputTokenCostAbove32kTokens
	}
	if p.CacheCreationInputTokenCostAbove32kTokens != nil {
		price.CacheCreationInputTokenCostAbove32k = *p.CacheCreationInputTokenCostAbove32kTokens
	}
	if p.CacheReadInputTokenCostAbove128kTokens != nil {
		price.CacheReadInputTokenCostAbove128k = *p.CacheReadInputTokenCostAbove128kTokens
	}
	if p.CacheCreationInputTokenCostAbove128kTokens != nil {
		price.CacheCreationInputTokenCostAbove128k = *p.CacheCreationInputTokenCostAbove128kTokens
	}
	if p.CacheReadInputTokenCostAbove200kTokens != nil {
		price.CacheReadInputTokenCostAbove200k = *p.CacheReadInputTokenCostAbove200kTokens
	}
	if p.CacheCreationInputTokenCostAbove200kTokens != nil {
		price.CacheCreationInputTokenCostAbove200k = *p.CacheCreationInputTokenCostAbove200kTokens
	}
	if p.CacheReadInputTokenCostAbove256kTokens != nil {
		price.CacheReadInputTokenCostAbove256k = *p.CacheReadInputTokenCostAbove256kTokens
	}
	if p.CacheCreationInputTokenCostAbove256kTokens != nil {
		price.CacheCreationInputTokenCostAbove256k = *p.CacheCreationInputTokenCostAbove256kTokens
	}
	if p.CacheCreationInputTokenCostAbove1hr != nil {
		price.CacheCreationInputTokenCostAbove1hr = *p.CacheCreationInputTokenCostAbove1hr
	}
	if p.CacheCreationInputTokenCostAbove1hrAbove200kTokens != nil {
		price.CacheCreationInputTokenCostAbove1hrAbove200k = *p.CacheCreationInputTokenCostAbove1hrAbove200kTokens
	}
	if p.CacheReadInputTokenCostAbove272kTokens != nil {
		price.CacheReadInputTokenCostAbove272k = *p.CacheReadInputTokenCostAbove272kTokens
	}
	if p.CacheCreationInputTokenCostAbove272kTokens != nil {
		price.CacheCreationInputTokenCostAbove272k = *p.CacheCreationInputTokenCostAbove272kTokens
	}
	if p.CacheReadInputTokenCostAbove512kTokens != nil {
		price.CacheReadInputTokenCostAbove512k = *p.CacheReadInputTokenCostAbove512kTokens
	}
	if p.CacheCreationInputTokenCostAbove512kTokens != nil {
		price.CacheCreationInputTokenCostAbove512k = *p.CacheCreationInputTokenCostAbove512kTokens
	}
	if p.CacheReadInputAudioTokenCost != nil {
		price.CacheReadInputAudioTokenCost = *p.CacheReadInputAudioTokenCost
	}
	if p.OutputCostPerImage != nil {
		price.OutputCostPerImage = *p.OutputCostPerImage
	}
	if p.OutputCostPerImageToken != nil {
		price.OutputCostPerImageToken = *p.OutputCostPerImageToken
	}
	if p.OutputCostPerVideoPerSecond != nil {
		price.OutputCostPerVideoPerSecond = *p.OutputCostPerVideoPerSecond
	}
	if len(p.SearchContextCostPerQuery) > 0 {
		price.SearchContextCostPerQuery = p.SearchContextCostPerQuery
	}
	if p.WebSearchBillingUnit != nil {
		price.WebSearchBillingUnit = *p.WebSearchBillingUnit
	}

	return price
}

func pricingProviderName(params *queries.GenericLiteLLMParams) string {
	if params == nil {
		return ""
	}
	if params.CustomLLMProvider != nil && *params.CustomLLMProvider != "" {
		return *params.CustomLLMProvider
	}
	if params.CustomLLMProviderName != nil && *params.CustomLLMProviderName != "" {
		return *params.CustomLLMProviderName
	}
	if params.Model != nil {
		if slash := strings.IndexByte(*params.Model, '/'); slash > 0 {
			return (*params.Model)[:slash]
		}
	}
	return ""
}
