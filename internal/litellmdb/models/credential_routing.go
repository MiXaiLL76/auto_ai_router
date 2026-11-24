package models

import (
	"maps"
	"slices"

	"github.com/mixaill76/auto_ai_router/internal/scope"
)

const allowedCredentialMetadataField = "allowed_cred"

// AllowedCredentialRoutingIDs returns the intersection of the restrictions
// specified in the key's own metadata and its organization's and team's
// metadata. The second result reports whether any restriction is present.
// Empty and null lists are ignored. Malformed values deny all credentials.
func (t *TokenInfo) AllowedCredentialRoutingIDs() ([]string, bool) {
	if t == nil {
		return nil, false
	}

	var allowed map[string]struct{}
	for _, metadata := range []map[string]interface{}{t.OrganizationMetadata, t.TeamMetadata, t.Metadata} {
		value := metadata[allowedCredentialMetadataField]
		if value == nil {
			continue
		}
		ids := credentialRoutingIDs(value)
		if ids == nil {
			return nil, true
		}
		if len(ids) == 0 {
			continue
		}
		if allowed == nil {
			allowed = ids
			continue
		}
		for id := range allowed {
			if _, present := ids[id]; !present {
				delete(allowed, id)
			}
		}
		if len(allowed) == 0 {
			return nil, true
		}
	}
	if allowed == nil {
		return nil, false
	}
	return slices.Sorted(maps.Keys(allowed)), true
}

func credentialRoutingIDs(value interface{}) map[string]struct{} {
	ids := make(map[string]struct{})
	switch list := value.(type) {
	case []string:
		for _, id := range list {
			if !scope.ValidCredentialRoutingID(id) {
				return nil
			}
			ids[id] = struct{}{}
		}
	case []interface{}:
		for _, item := range list {
			id, ok := item.(string)
			if !ok || !scope.ValidCredentialRoutingID(id) {
				return nil
			}
			ids[id] = struct{}{}
		}
	default:
		return nil
	}
	return ids
}
