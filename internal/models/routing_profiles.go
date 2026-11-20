package models

import (
	"errors"
	"fmt"
	"slices"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

type routingRule struct {
	models      []string
	identities  map[string]bool
	credentials []string
}

type routingProfile struct {
	rules []routingRule
}

// RoutingProfileRegistry holds immutable operator-owned profiles. Only their
// opaque IDs are read from authenticated database metadata, never request data.
type RoutingProfileRegistry struct {
	byID    map[string]routingProfile
	manager *Manager
}

func LoadRoutingProfiles(policies []config.RoutingProfileConfig, credentials []config.CredentialConfig, manager *Manager) (*RoutingProfileRegistry, error) {
	if err := config.ValidateRoutingProfiles(policies, credentials); err != nil {
		return nil, err
	}
	registry := &RoutingProfileRegistry{byID: make(map[string]routingProfile), manager: manager}
	if len(policies) > 0 && manager == nil {
		return nil, fmt.Errorf("routing profiles require a model catalog")
	}
	for i, entry := range policies {
		policy := routingProfile{}
		for j, cfg := range entry.Rules {
			allowed := slices.Clone(cfg.CredentialAllowlist)
			if len(cfg.CredentialAllowlist) == 0 {
				for _, credential := range credentials {
					if !credential.IsProxyLike() {
						allowed = append(allowed, credential.Name)
					}
				}
			}
			allowed = slices.DeleteFunc(allowed, func(name string) bool { return slices.Contains(cfg.CredentialDenylist, name) })
			rule := routingRule{models: slices.Clone(cfg.Models), credentials: allowed, identities: make(map[string]bool)}
			for _, name := range cfg.Models {
				resolved, err := registry.resolve(name)
				if err != nil {
					return nil, fmt.Errorf("routing_profiles[%d].rules[%d]: model is not uniquely routable", i, j)
				}
				supported := manager.GetCredentialsForModel(resolved)
				if len(supported) == 0 {
					return nil, fmt.Errorf("routing_profiles[%d].rules[%d]: model has no configured route", i, j)
				}
				rule.identities[name] = true
				for identity := range manager.keyRoutingModelIdentities(resolved) {
					rule.identities[identity] = true
				}
			}
			policy.rules = append(policy.rules, rule)
		}
		registry.byID[entry.ID] = policy
	}
	return registry, nil
}

func (r *RoutingProfileRegistry) resolve(name string) (string, error) {
	canonical, _, err := r.manager.ResolvePublicModelAlias(name)
	if err != nil {
		return "", err
	}
	resolved, _ := r.manager.ResolveAlias(canonical)
	return resolved, nil
}

var ErrInvalidRoutingProfile = errors.New("routing profile is unavailable")

// Restrict fails closed for any missing profile, including an empty registry.
// Empty bindings preserve ordinary routing. Profile IDs are not credentials.
func (r *RoutingProfileRegistry) Restrict(visibility scope.Context, profileIDs []string, publicModel, routeModel string) (scope.Context, error) {
	if len(profileIDs) == 0 {
		return visibility, nil
	}
	if r == nil {
		return visibility.RestrictCredentials(nil), ErrInvalidRoutingProfile
	}
	for _, id := range profileIDs {
		if _, exists := r.byID[id]; !exists {
			return visibility.RestrictCredentials(nil), ErrInvalidRoutingProfile
		}
	}
	identities := r.manager.keyRoutingModelIdentities(routeModel)
	if resolved, err := r.resolve(publicModel); err == nil && !identities[resolved] {
		for identity := range r.manager.keyRoutingModelIdentities(resolved) {
			identities[identity] = true
		}
	}
	identities[publicModel] = true
	for _, id := range profileIDs {
		for _, rule := range r.byID[id].rules {
			matches := false
			for identity := range rule.identities {
				matches = matches || identities[identity]
			}
			// Include a refreshed alias without forgetting its startup identity.
			for _, name := range rule.models {
				if resolved, err := r.resolve(name); err == nil {
					matches = matches || identities[resolved]
				}
			}
			if matches {
				visibility = visibility.RestrictCredentials(rule.credentials)
			}
		}
	}
	return visibility, nil
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
