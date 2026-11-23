package models

import (
	"errors"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

const RoutingModelsField = "air_routing_models"
const RoutingProfilesField = "air_routing_profiles"

var ErrRoutingPolicy = errors.New("routing policy is invalid or unavailable")

type RoutingPolicy struct {
	Models []string
	Pools  []string
}

// ParseKeyRoutingPolicy reads only trusted authentication metadata. The optional
// model selector opts a key into strict isolation; profiles alone retain their
// existing soft-selection semantics. Strict selectors are key-specific.
func ParseKeyRoutingPolicy(key, team, organization map[string]interface{}) (*RoutingPolicy, error) {
	// The common case is a key with no routing metadata at all; bail out before
	// allocating anything.
	if _, hasSelector := key[RoutingModelsField]; !hasSelector &&
		team[RoutingModelsField] == nil && organization[RoutingModelsField] == nil {
		return nil, nil
	}
	for _, scoped := range []struct {
		name     string
		metadata map[string]interface{}
	}{{"team", team}, {"organization", organization}} {
		if scoped.metadata == nil {
			continue
		}
		if _, exists := scoped.metadata[RoutingModelsField]; exists {
			return nil, fmt.Errorf("%w: reason=selector_on_%s field=%s", ErrRoutingPolicy, scoped.name, RoutingModelsField)
		}
	}
	value, exists := key[RoutingModelsField]
	if !exists {
		return nil, nil
	}
	modelNames, ok := strictRoutingList(value, 64)
	if !ok {
		return nil, fmt.Errorf("%w: reason=bad_shape field=%s", ErrRoutingPolicy, RoutingModelsField)
	}
	// A model selector without a usable profile list selects nothing: it cannot
	// name a pool, and treating it as "restrict to an empty allowlist" would take
	// the whole key down, models the selector never mentioned included. Fall back
	// to ordinary routing rather than failing a key that used the wrong field.
	if emptyRoutingList(key[RoutingProfilesField]) {
		return nil, nil
	}
	pools, ok := strictRoutingList(key[RoutingProfilesField], 16)
	if !ok {
		return nil, fmt.Errorf("%w: reason=bad_shape field=%s", ErrRoutingPolicy, RoutingProfilesField)
	}
	for _, model := range modelNames {
		if len(model) > 256 || strings.ContainsAny(model, " \t\r\n*?\x00") {
			return nil, fmt.Errorf("%w: reason=bad_shape field=%s", ErrRoutingPolicy, RoutingModelsField)
		}
	}
	for _, pool := range pools {
		if !config.PoolIDPattern.MatchString(pool) {
			return nil, fmt.Errorf("%w: reason=bad_shape field=%s", ErrRoutingPolicy, RoutingProfilesField)
		}
	}
	return &RoutingPolicy{Models: modelNames, Pools: pools}, nil
}

// emptyRoutingList reports whether a profile value names no pool at all: absent,
// null or an empty list. Anything else must go through strictRoutingList, which
// rejects the wrong shape.
func emptyRoutingList(value interface{}) bool {
	switch values := value.(type) {
	case nil:
		return true
	case []string:
		return len(values) == 0
	case []interface{}:
		return len(values) == 0
	default:
		return false
	}
}

func strictRoutingList(value interface{}, maximum int) ([]string, bool) {
	var result []string
	switch values := value.(type) {
	case []string:
		result = values
	case []interface{}:
		if len(values) > maximum {
			return nil, false
		}
		for _, value := range values {
			text, ok := value.(string)
			if !ok {
				return nil, false
			}
			result = append(result, text)
		}
	default:
		return nil, false
	}
	if len(result) == 0 || len(result) > maximum {
		return nil, false
	}
	for _, value := range result {
		if value == "" || strings.TrimSpace(value) != value {
			return nil, false
		}
	}
	return result, true
}

// The catalog index and pool membership are replaced under the Manager lock.
// Known identity edges are retained for this process lifetime: removing an alias
// bridge during a DB refresh must not silently widen an existing restriction.
type routingPoolIndex struct {
	identities map[string]string
	pools      map[string]bool
	members    map[string]map[string]map[string]bool // identity -> pool -> credentials
}

func (m *Manager) rebuildRoutingPoolsLocked() {
	parent := make(map[string]string)
	var find func(string) string
	find = func(name string) string {
		if parent[name] == "" {
			parent[name] = name
		}
		if parent[name] != name {
			parent[name] = find(parent[name])
		}
		return parent[name]
	}
	join := func(a, b string) {
		if a != "" && b != "" {
			parent[find(a)] = find(b)
		}
	}
	if m.routingPools != nil {
		for name, identity := range m.routingPools.identities {
			join(name, identity)
		}
	}
	for _, aliases := range []map[string]string{m.modelAliases, m.publicModelAliases, m.acceptedModelAliases, m.modelRealNames} {
		for name, target := range aliases {
			join(name, target)
		}
	}
	// Wire names identify equivalent aliases only within the SAME credential.
	type wireIdentity struct{ credential, wire string }
	wires := make(map[wireIdentity]string)
	for name, entries := range m.modelLimits {
		find(name)
		for _, entry := range entries {
			if entry.Credential == "" {
				continue
			}
			wire := name
			if mapped := m.modelRealNamesPerCred[entry.Credential][name]; mapped != "" {
				wire = mapped
			} else if mapped := m.modelRealNames[name]; mapped != "" {
				wire = mapped
			}
			key := wireIdentity{entry.Credential, wire}
			if other := wires[key]; other != "" {
				join(name, other)
			} else {
				wires[key] = name
			}
		}
	}
	index := &routingPoolIndex{identities: make(map[string]string), pools: make(map[string]bool), members: make(map[string]map[string]map[string]bool)}
	for name := range parent {
		index.identities[name] = find(name)
	}
	for name, entries := range m.modelLimits {
		identity := index.identities[name]
		for _, entry := range entries {
			for _, pool := range entry.Pools {
				if !config.PoolIDPattern.MatchString(pool) {
					// Silently dropping the tag leaves the deployment looking
					// unpooled and a strict key answering a bare 503 with nothing
					// to trace the typo back to.
					m.logger.Warn("Ignoring a pool tag that is not a valid ID",
						"model", name, "credential", entry.Credential, "pool", pool)
					continue
				}
				index.pools[pool] = true
				// Global tags remain valid for legacy selection but do not
				// authorize every account for a strictly isolated request.
				if entry.Credential == "" {
					continue
				}
				if index.members[identity] == nil {
					index.members[identity] = make(map[string]map[string]bool)
				}
				if index.members[identity][pool] == nil {
					index.members[identity][pool] = make(map[string]bool)
				}
				index.members[identity][pool][entry.Credential] = true
			}
		}
	}
	m.routingPools = index
}

// RestrictToModelPools narrows existing visibility for explicitly selected
// models. A matching model with zero deployments denies ALL routes; an unrelated
// model retains its existing routing. Multiple pools are alternatives.
func (m *Manager) RestrictToModelPools(visibility scope.Context, policy *RoutingPolicy, publicModel, routeModel string) (scope.Context, error) {
	if policy == nil {
		return visibility, nil
	}
	if m == nil {
		return visibility.RestrictCredentials(nil), fmt.Errorf("%w: reason=no_model_manager", ErrRoutingPolicy)
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	index := m.routingPools
	if index == nil {
		return visibility.RestrictCredentials(nil), fmt.Errorf("%w: reason=no_catalog_index", ErrRoutingPolicy)
	}
	matches := false
	for _, model := range policy.Models {
		identity, known := index.identities[model]
		if !known {
			return visibility.RestrictCredentials(nil), fmt.Errorf("%w: reason=unknown_selected_model field=%s", ErrRoutingPolicy, RoutingModelsField)
		}
		matches = matches || identity == index.identities[publicModel] || identity == index.identities[routeModel]
	}
	if !matches {
		return visibility, nil
	}
	var allowed []string
	for _, pool := range policy.Pools {
		if !index.pools[pool] {
			return visibility.RestrictCredentials(nil), fmt.Errorf("%w: reason=unknown_pool field=%s", ErrRoutingPolicy, RoutingProfilesField)
		}
		for credential := range index.members[index.identities[routeModel]][pool] {
			allowed = append(allowed, credential)
		}
	}
	return visibility.RestrictCredentials(allowed), nil
}
