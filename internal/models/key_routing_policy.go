package models

import (
	"fmt"
	"slices"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

type keyRoutingRule struct {
	models      []string
	identities  map[string]bool
	credentials []string
}

type keyRoutingPolicy struct {
	id    string
	rules []keyRoutingRule
}

// KeyRoutingPolicyRegistry is an immutable, server-owned binding. It is never
// populated from request headers, bodies, aliases or client-editable metadata.
type KeyRoutingPolicyRegistry struct {
	byHash  map[string]keyRoutingPolicy
	manager *Manager
}

func LoadKeyRoutingPolicies(policies []config.KeyRoutingPolicyConfig, credentials []config.CredentialConfig, manager *Manager) (*KeyRoutingPolicyRegistry, error) {
	if err := config.ValidateKeyRoutingPolicies(policies, credentials); err != nil {
		return nil, err
	}
	registry := &KeyRoutingPolicyRegistry{byHash: make(map[string]keyRoutingPolicy), manager: manager}
	if len(policies) > 0 && manager == nil {
		return nil, fmt.Errorf("key routing policies require a model catalog")
	}
	for i, entry := range policies {
		hash, err := entry.ResolveKeyHash()
		if err != nil {
			return nil, fmt.Errorf("key_routing_policies[%d]: %w", i, err)
		}
		policy := keyRoutingPolicy{id: entry.ID}
		for j, cfg := range entry.Rules {
			rule := keyRoutingRule{models: slices.Clone(cfg.Models), credentials: slices.Clone(cfg.CredentialAllowlist), identities: make(map[string]bool)}
			for _, name := range cfg.Models {
				resolved, err := registry.resolve(name)
				if err != nil {
					return nil, fmt.Errorf("key_routing_policies[%d].rules[%d]: model is not uniquely routable", i, j)
				}
				supported := manager.GetCredentialsForModel(resolved)
				if !slices.ContainsFunc(cfg.CredentialAllowlist, func(credential string) bool { return slices.Contains(supported, credential) }) {
					return nil, fmt.Errorf("key_routing_policies[%d].rules[%d]: model has no permitted direct route", i, j)
				}
				rule.identities[name] = true
				for identity := range manager.keyRoutingModelIdentities(resolved) {
					rule.identities[identity] = true
				}
			}
			policy.rules = append(policy.rules, rule)
		}
		registry.byHash[hash] = policy
	}
	return registry, nil
}

func (r *KeyRoutingPolicyRegistry) resolve(name string) (string, error) {
	canonical, _, err := r.manager.ResolvePublicModelAlias(name)
	if err != nil {
		return "", err
	}
	resolved, _ := r.manager.ResolveAlias(canonical)
	return resolved, nil
}

// Restrict intersects all matching rules. Public names and the admitted route
// identity are both considered so neither an accepted alias nor an organization
// model mapping can widen a caller's permitted credential pool.
func (r *KeyRoutingPolicyRegistry) Restrict(visibility scope.Context, keyHash, publicModel, routeModel string) (scope.Context, string) {
	if r == nil {
		return visibility, ""
	}
	policy, exists := r.byHash[keyHash]
	if !exists {
		return visibility, ""
	}
	identities := r.manager.keyRoutingModelIdentities(routeModel)
	if resolved, err := r.resolve(publicModel); err == nil {
		if !identities[resolved] {
			for identity := range r.manager.keyRoutingModelIdentities(resolved) {
				identities[identity] = true
			}
		}
	}
	identities[publicModel] = true
	matched := false
	for _, rule := range policy.rules {
		matches := false
		for identity := range rule.identities {
			matches = matches || identities[identity]
		}
		// Catalog refreshes can retarget an accepted alias. Include the current
		// target without dropping the original identity on a transient outage.
		for _, name := range rule.models {
			if resolved, err := r.resolve(name); err == nil {
				matches = matches || identities[resolved]
			}
		}
		if matches {
			visibility = visibility.RestrictCredentials(rule.credentials)
			matched = true
		}
	}
	if matched {
		return visibility, policy.id
	}
	return visibility, ""
}

// Match deployment aliases using the real model name of EACH credential, not
// the pre-selection global name. One logical route can use different wire names
// on different providers. Resolve the connected aliases under one catalog lock
// so a refresh cannot leave a partially expanded identity set.
func (m *Manager) keyRoutingModelIdentities(model string) map[string]bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	realName := func(alias, credential string) string {
		if name, ok := m.modelRealNamesPerCred[credential][alias]; ok {
			return name
		}
		if name, ok := m.modelRealNames[alias]; ok {
			return name
		}
		return alias
	}
	identities := map[string]bool{model: true}
	queue := []string{model}
	for i := 0; i < len(queue); i++ {
		current := queue[i]
		for _, credential := range m.modelToCredentials[current] {
			wireName := realName(current, credential)
			for _, alias := range m.credentialModels[credential] {
				if identities[alias] || realName(alias, credential) != wireName {
					continue
				}
				identities[alias] = true
				queue = append(queue, alias)
			}
		}
	}
	return identities
}
