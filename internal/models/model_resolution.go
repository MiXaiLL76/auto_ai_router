package models

// ModelResolution describes a request's names before credential selection.
// RealModelID is the global default; direct providers may override it per credential.
// ModelPrice is populated only when an organization supplies an exact tariff.
type ModelResolution struct {
	PublicModelID    string
	CanonicalModelID string
	ModelID          string
	RealModelID      string
	PriceModelID     string
	ModelPrice       *ModelPrice
}

// ResolveModel owns client-surface checks and alias precedence for both ordinary
// and organization requests. A registered route takes precedence over model_alias;
// only a trusted internal request can also bypass accepted_model_alias.
func (m *Manager) ResolveModel(modelID string, policy *OrganizationPolicy, trustedInternal bool) (ModelResolution, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	canonicalID := modelID
	mapped := false
	if policy != nil {
		trustedInternal = false
		if !policy.allowlistAdmitsLocked(modelID) {
			return ModelResolution{}, false
		}
		canonicalID, mapped = policy.MappingTarget(modelID)
		if !mapped {
			canonicalID = modelID
		} else if _, accepted := m.acceptedModelAliases[canonicalID]; accepted {
			// Organization mappings target routes or model_alias, not accepted aliases.
			return ModelResolution{}, false
		}
	}
	if !mapped && !(trustedInternal && len(m.modelToCredentials[modelID]) > 0) {
		if target, accepted := m.acceptedModelAliases[modelID]; accepted {
			canonicalID = target
		}
		if !trustedInternal && m.clientModelSurfaceConfigured {
			if _, allowed := m.clientModelIDs[canonicalID]; !allowed {
				return ModelResolution{}, false
			}
		}
	}
	routeID, active := m.clientCanonicalRouteTargetLocked(canonicalID)
	if !active {
		return ModelResolution{}, false
	}
	realID := routeID
	if real, ok := m.modelRealNames[routeID]; ok {
		realID = real
	}
	resolution := ModelResolution{
		PublicModelID: modelID, CanonicalModelID: canonicalID,
		ModelID: routeID, RealModelID: realID,
	}
	if policy != nil {
		resolution.PriceModelID = modelID
		resolution.ModelPrice, _ = policy.Price(modelID)
	}
	return resolution, true
}
