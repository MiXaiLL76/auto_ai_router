package models

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

type OrganizationPolicyRegistry struct {
	byOrganization map[string]*OrganizationPolicy
}

type OrganizationPolicy struct {
	OrganizationID  string
	PriceProfileID  string
	ModelPricesLink string
	ProfileSHA256   string
	AllowlistSet    bool

	allowlist          map[string]struct{}
	mappings           map[string]string
	prices             map[string]*ModelPrice
	credentialDenylist []string
}

type OrganizationModelResolution struct {
	PublicModelID    string
	CanonicalModelID string
	ModelID          string
	RealModelID      string
	PriceModelID     string
	ModelPrice       *ModelPrice
}

type OrganizationPolicyLoadOptions struct {
	LiteLLMDBEnabled      bool
	LiteLLMDBRequired     bool
	DisableSpendLogsWrite bool
}

// Several organizations routinely share one model_prices_link / price
// profile. Load, parse, duplicate-key scan and SHA-256 each distinct link
// only once; the resulting price map is read-only after construction, so it
// is safe to share across policies.
type loadedProfile struct {
	prices   map[string]*ModelPrice
	identity profileIdentity
}

func NewOrganizationPolicyRegistry() *OrganizationPolicyRegistry {
	return &OrganizationPolicyRegistry{byOrganization: make(map[string]*OrganizationPolicy)}
}

func LoadOrganizationPolicies(
	policies []config.OrganizationPolicyConfig,
	manager *Manager,
	opts OrganizationPolicyLoadOptions,
) (*OrganizationPolicyRegistry, error) {
	registry := NewOrganizationPolicyRegistry()
	if len(policies) == 0 {
		return registry, nil
	}
	if !opts.LiteLLMDBEnabled || !opts.LiteLLMDBRequired || opts.DisableSpendLogsWrite {
		return nil, fmt.Errorf("organization policies require litellm_db.enabled=true, litellm_db.is_required=true, and litellm_db.disable_spend_logs_write=false")
	}
	if manager == nil {
		return nil, fmt.Errorf("organization policies require a model manager")
	}

	profileByLink := make(map[string]loadedProfile, len(policies))

	profiles := make(map[string]profileIdentity)
	for _, cfg := range policies {
		loaded, cached := profileByLink[cfg.ModelPricesLink]
		if !cached {
			priceRows, identity, err := loadStrictOrganizationPriceProfile(cfg.PriceProfileID, cfg.ModelPricesLink)
			if err != nil {
				return nil, fmt.Errorf("organization policy %q: %w", cfg.OrganizationID, err)
			}
			loaded = loadedProfile{prices: priceRows, identity: identity}
			profileByLink[cfg.ModelPricesLink] = loaded
		}
		priceRows := loaded.prices
		identity := loaded.identity
		identity.id = cfg.PriceProfileID
		if previous, exists := profiles[cfg.PriceProfileID]; exists && previous != identity {
			return nil, fmt.Errorf("organization policy %q: profile %q has conflicting source or digest", cfg.OrganizationID, cfg.PriceProfileID)
		}
		profiles[cfg.PriceProfileID] = identity

		policy := &OrganizationPolicy{
			OrganizationID:     cfg.OrganizationID,
			PriceProfileID:     cfg.PriceProfileID,
			ModelPricesLink:    cfg.ModelPricesLink,
			ProfileSHA256:      identity.sha256,
			AllowlistSet:       cfg.AllowlistSet,
			allowlist:          make(map[string]struct{}, len(cfg.ModelAllowlist)),
			mappings:           make(map[string]string, len(cfg.ModelMappings)),
			prices:             priceRows,
			credentialDenylist: append([]string(nil), cfg.CredentialDenylist...),
		}
		for _, modelID := range cfg.ModelAllowlist {
			policy.allowlist[modelID] = struct{}{}
		}
		for source, target := range cfg.ModelMappings {
			policy.mappings[source] = target
			if _, ok := manager.ResolveOrganizationMappingTarget(target); !ok {
				return nil, fmt.Errorf("organization policy %q: model mapping %q target %q is not an active direct route or active model_alias", cfg.OrganizationID, source, target)
			}
		}
		if cfg.AllowlistSet {
			for _, modelID := range cfg.ModelAllowlist {
				if !policy.inSurface(manager, modelID) {
					return nil, fmt.Errorf("organization policy %q: allowlisted model %q is not routable", cfg.OrganizationID, modelID)
				}
				if _, ok := policy.prices[modelID]; !ok {
					return nil, fmt.Errorf("organization policy %q: allowlisted model %q has no exact profile price", cfg.OrganizationID, modelID)
				}
			}
		}
		registry.byOrganization[policy.OrganizationID] = policy
	}
	return registry, nil
}

func (p *OrganizationPolicy) CredentialDenylist() []string {
	if p == nil || len(p.credentialDenylist) == 0 {
		return nil
	}
	return append([]string(nil), p.credentialDenylist...)
}

func (r *OrganizationPolicyRegistry) Empty() bool {
	return r == nil || len(r.byOrganization) == 0
}

func (r *OrganizationPolicyRegistry) HasOrganization(organizationID string) bool {
	if r == nil {
		return false
	}
	_, ok := r.byOrganization[strings.TrimSpace(organizationID)]
	return ok
}

func (r *OrganizationPolicyRegistry) Policy(organizationID string) (*OrganizationPolicy, bool) {
	if r == nil {
		return nil, false
	}
	policy, ok := r.byOrganization[strings.TrimSpace(organizationID)]
	return policy, ok
}

func (p *OrganizationPolicy) MappingTarget(modelID string) (string, bool) {
	if p == nil {
		return "", false
	}
	target, ok := p.mappings[modelID]
	return target, ok
}

func (p *OrganizationPolicy) Price(modelID string) (*ModelPrice, bool) {
	if p == nil {
		return nil, false
	}
	price, ok := p.prices[modelID]
	return price, ok
}

func (p *OrganizationPolicy) CacheKey() string {
	if p == nil {
		return ""
	}
	return p.OrganizationID + "|" + p.PriceProfileID + "|" + p.ProfileSHA256
}

func (p *OrganizationPolicy) inSurface(manager *Manager, publicID string) bool {
	if p == nil || manager == nil {
		return false
	}
	if p.AllowlistSet {
		if _, ok := p.allowlist[publicID]; !ok {
			return false
		}
	}
	if _, ok := p.mappings[publicID]; ok {
		return true
	}
	return manager.IsClientModelIDRoutable(publicID)
}

func (m *Manager) ResolveOrganizationModel(policy *OrganizationPolicy, publicID string) (OrganizationModelResolution, error) {
	if policy == nil {
		return OrganizationModelResolution{}, fmt.Errorf("organization policy is required")
	}
	publicID = strings.TrimSpace(publicID)
	if !policy.inSurface(m, publicID) {
		return OrganizationModelResolution{}, ErrOrganizationModelNotFound
	}

	canonicalID := publicID
	if target, mapped := policy.MappingTarget(publicID); mapped {
		resolved, active := m.ResolveOrganizationMappingTarget(target)
		if !active {
			return OrganizationModelResolution{}, ErrOrganizationModelNotFound
		}
		canonicalID = target
		publicRouteID := resolved
		realID := publicRouteID
		if realName, ok := m.GetRealModelName(publicRouteID); ok {
			realID = realName
		}
		price, _ := policy.Price(publicID)
		return OrganizationModelResolution{
			PublicModelID:    publicID,
			CanonicalModelID: canonicalID,
			ModelID:          publicRouteID,
			RealModelID:      realID,
			PriceModelID:     publicID,
			ModelPrice:       price,
		}, nil
	}

	resolvedCanonical, isAlias, err := m.ResolvePublicModelAlias(publicID)
	if err != nil {
		return OrganizationModelResolution{}, ErrOrganizationModelNotFound
	}
	if isAlias {
		canonicalID = resolvedCanonical
	}
	modelID := canonicalID
	if resolved, isModelAlias := m.ResolveAlias(modelID); isModelAlias {
		modelID = resolved
	}
	realID := modelID
	if realName, ok := m.GetRealModelName(modelID); ok {
		realID = realName
	}
	price, _ := policy.Price(publicID)
	return OrganizationModelResolution{
		PublicModelID:    publicID,
		CanonicalModelID: canonicalID,
		ModelID:          modelID,
		RealModelID:      realID,
		PriceModelID:     publicID,
		ModelPrice:       price,
	}, nil
}

func (m *Manager) ResolveOrganizationModelScoped(
	policy *OrganizationPolicy,
	publicID string,
	visibility scope.Context,
) (OrganizationModelResolution, error) {
	resolution, err := m.ResolveOrganizationModel(policy, publicID)
	if err != nil {
		return OrganizationModelResolution{}, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if !m.modelVisibleInScopeLocked(resolution.ModelID, visibility) {
		return OrganizationModelResolution{}, ErrOrganizationModelNotFound
	}
	return resolution, nil
}

func (m *Manager) ResolveOrganizationMappingTarget(target string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if _, ok := m.publicModelAliases[target]; ok {
		return "", false
	}
	if _, ok := m.acceptedModelAliases[target]; ok {
		return "", false
	}
	return m.clientCanonicalRouteTargetLocked(target)
}

func (m *Manager) IsAnyModelIDAllowedByScope(candidates []string, allowedModelIDs []string) bool {
	for _, candidate := range candidates {
		if m.IsModelIDAllowedByScope(candidate, allowedModelIDs) {
			return true
		}
	}
	return false
}

func (m *Manager) GetAllModelsScopedForOrganization(visibility scope.Context, policy *OrganizationPolicy) ModelsResponse {
	if policy == nil {
		return m.GetAllModelsScoped(visibility)
	}
	if response, ok := m.getCachedScopedAllModelsForOrganization(visibility, policy); ok {
		return response
	}
	response := m.GetAllModelsScoped(visibility)
	projected := m.projectOrganizationCatalog(response, visibility, policy)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.scopedAllModelsCache.Add(m.scopedAllModelsCacheKeyForOrganizationLocked(visibility, policy), allModelsCache{
		response:  copyModelsResponse(projected),
		expiresAt: utils.NowUTC().Add(allModelsCacheTTL),
	})
	return projected
}

// GetAllModelsWithAccessGroupsScopedForOrganization deliberately ignores
// include_model_access_groups for organization-scoped keys: the access-group
// projection is an administrative view over internal routes, and an
// organization catalog is an explicit curated surface (allowlist + mappings +
// prices). Returning access-group pseudo-models would re-introduce backend IDs
// through a query parameter, exactly as GetAllModelsWithAccessGroupsScoped
// already suppresses them once a client model surface is configured.
func (m *Manager) GetAllModelsWithAccessGroupsScopedForOrganization(visibility scope.Context, policy *OrganizationPolicy) ModelsResponse {
	return m.GetAllModelsScopedForOrganization(visibility, policy)
}

func (m *Manager) getCachedScopedAllModelsForOrganization(visibility scope.Context, policy *OrganizationPolicy) (ModelsResponse, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cacheKey := m.scopedAllModelsCacheKeyForOrganizationLocked(visibility, policy)
	cached, ok := m.scopedAllModelsCache.Get(cacheKey)
	if !ok || cached.expiresAt.IsZero() || !utils.NowUTC().Before(cached.expiresAt) {
		if ok {
			m.scopedAllModelsCache.Remove(cacheKey)
		}
		return ModelsResponse{}, false
	}
	return copyModelsResponse(cached.response), true
}

func (m *Manager) scopedAllModelsCacheKeyForOrganizationLocked(visibility scope.Context, policy *OrganizationPolicy) string {
	return m.scopedAllModelsCacheKeyLocked(visibility) + "|op:" + policy.CacheKey()
}

func (m *Manager) projectOrganizationCatalog(response ModelsResponse, visibility scope.Context, policy *OrganizationPolicy) ModelsResponse {
	if policy == nil {
		return response
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	// globalByID indexes every model reachable in this scope by its real
	// internal id so a mapped entry can borrow created/owned_by from the route
	// it points at. response.Data is keyed by public/client id, which is not
	// the route id returned by resolveOrganizationMappingTargetLocked.
	globalByID := make(map[string]Model, len(m.allModels)+len(response.Data))
	for _, model := range m.allModels {
		if model.ID != "" {
			globalByID[model.ID] = model
		}
	}
	for _, model := range response.Data {
		if model.ID != "" {
			globalByID[model.ID] = model
		}
	}

	publicByID := make(map[string]Model, len(response.Data)+len(policy.mappings))
	for _, model := range response.Data {
		if !policy.allowlistAdmitsLocked(model.ID) {
			continue
		}
		if _, priced := policy.prices[model.ID]; !priced {
			continue
		}
		publicByID[model.ID] = model
	}
	for source, target := range policy.mappings {
		if policy.AllowlistSet {
			if _, ok := policy.allowlist[source]; !ok {
				continue
			}
		}
		if _, priced := policy.prices[source]; !priced {
			continue
		}
		routeID, active := m.resolveOrganizationMappingTargetLocked(target)
		if !active || !m.modelVisibleInScopeLocked(routeID, visibility) {
			continue
		}
		model := Model{ID: source, Object: "model", OwnedBy: "system"}
		if sourceModel, ok := globalByID[routeID]; ok {
			model = sourceModel
			model.ID = source
		} else if sourceModel, ok := globalByID[target]; ok {
			model = sourceModel
			model.ID = source
		}
		if model.Object == "" {
			model.Object = "model"
		}
		publicByID[source] = model
	}

	result := make([]Model, 0, len(publicByID))
	for _, model := range publicByID {
		result = append(result, model)
	}
	slices.SortFunc(result, func(left, right Model) int {
		return strings.Compare(left.ID, right.ID)
	})
	return ModelsResponse{Object: response.Object, Data: result}
}

// allowlistAdmitsLocked is only the allowlist gate: it returns true for any id
// when no allowlist is configured. It is NOT a lock-held twin of inSurface and
// performs no mapping/routability check, so callers must pass ids that are
// already known to be routable (projectOrganizationCatalog feeds it entries
// straight from the routable global catalog).
func (p *OrganizationPolicy) allowlistAdmitsLocked(publicID string) bool {
	if p.AllowlistSet {
		_, ok := p.allowlist[publicID]
		return ok
	}
	return true
}

func (m *Manager) resolveOrganizationMappingTargetLocked(target string) (string, bool) {
	if _, ok := m.publicModelAliases[target]; ok {
		return "", false
	}
	if _, ok := m.acceptedModelAliases[target]; ok {
		return "", false
	}
	return m.clientCanonicalRouteTargetLocked(target)
}

func (m *Manager) modelVisibleInScopeLocked(modelID string, visibility scope.Context) bool {
	visibleCreds := m.visibleCredentialNamesLocked(visibility)
	return m.modelVisibleLocked(modelID, visibleCreds, visibility)
}

var (
	ErrOrganizationModelNotFound    = errors.New("organization model not found")
	ErrOrganizationPriceUnavailable = errors.New("organization price unavailable")
)

type profileIdentity struct {
	id     string
	source string
	sha256 string
}

func loadStrictOrganizationPriceProfile(profileID, link string) (map[string]*ModelPrice, profileIdentity, error) {
	data, err := LoadModelPriceBytes(link)
	if err != nil {
		return nil, profileIdentity{}, err
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return nil, profileIdentity{}, err
	}

	var raw map[string]json.RawMessage
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&raw); err != nil {
		return nil, profileIdentity{}, fmt.Errorf("failed to parse organization tariff JSON: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return nil, profileIdentity{}, err
	}

	prices := make(map[string]*ModelPrice, len(raw))
	for modelID, row := range raw {
		price, err := decodeStrictPriceRow(modelID, row)
		if err != nil {
			return nil, profileIdentity{}, err
		}
		prices[modelID] = price
	}
	sum := sha256.Sum256(data)
	return prices, profileIdentity{id: profileID, source: link, sha256: hex.EncodeToString(sum[:])}, nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	if decoder == nil {
		return nil
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("organization tariff JSON contains trailing data")
	}
	return nil
}

func decodeStrictPriceRow(modelID string, row json.RawMessage) (*ModelPrice, error) {
	if bytes.Equal(bytes.TrimSpace(row), []byte("null")) {
		return nil, fmt.Errorf("organization tariff %q: null price row", modelID)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(row, &fields); err != nil {
		return nil, fmt.Errorf("organization tariff %q: price row must be an object: %w", modelID, err)
	}
	if len(fields) == 0 {
		return nil, fmt.Errorf("organization tariff %q: empty price row", modelID)
	}
	known := modelPriceJSONFields()
	for field := range fields {
		if _, ok := known[field]; !ok {
			return nil, fmt.Errorf("organization tariff %q: unknown price field %q", modelID, field)
		}
	}
	var price ModelPrice
	decoder := json.NewDecoder(bytes.NewReader(row))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&price); err != nil {
		return nil, fmt.Errorf("organization tariff %q: %w", modelID, err)
	}
	hasPriceField := false
	for field := range fields {
		if known[field] {
			hasPriceField = true
			break
		}
	}
	if !hasPriceField {
		return nil, fmt.Errorf("organization tariff %q: no recognized price field", modelID)
	}
	return &price, nil
}

func modelPriceJSONFields() map[string]bool {
	result := make(map[string]bool)
	t := reflect.TypeOf(ModelPrice{})
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("json")
		name, _, _ := strings.Cut(tag, ",")
		if name == "" || name == "-" {
			continue
		}
		isPriceField := name != "litellm_provider" && name != "reasoning_tokens_additive" && name != "web_search_billing_unit"
		result[name] = isPriceField
	}
	return result
}

func rejectDuplicateJSONKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	type frame struct {
		object    bool
		keys      map[string]struct{}
		expectKey bool
	}
	var stack []frame
	completeValue := func() {
		if len(stack) > 0 && stack[len(stack)-1].object && !stack[len(stack)-1].expectKey {
			stack[len(stack)-1].expectKey = true
		}
	}
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		switch value := token.(type) {
		case json.Delim:
			switch value {
			case '{':
				stack = append(stack, frame{object: true, keys: map[string]struct{}{}, expectKey: true})
			case '}':
				stack = stack[:len(stack)-1]
				completeValue()
			case '[':
				stack = append(stack, frame{})
			case ']':
				stack = stack[:len(stack)-1]
				completeValue()
			}
		case string:
			if len(stack) == 0 || !stack[len(stack)-1].object {
				completeValue()
				continue
			}
			top := &stack[len(stack)-1]
			if top.expectKey {
				if _, exists := top.keys[value]; exists {
					return fmt.Errorf("duplicate JSON key %q", value)
				}
				top.keys[value] = struct{}{}
				top.expectKey = false
			} else {
				top.expectKey = true
			}
		default:
			completeValue()
		}
	}
}
