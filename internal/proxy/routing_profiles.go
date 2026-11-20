package proxy

import (
	"errors"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

var errRoutingProfile = errors.New("credential is outside the permitted routing pool")

type routingRestrictedContextKey struct{}

// routingProfilesFromMetadata reads only the server-side authentication record.
// A malformed field is never treated as an absent restriction. Management APIs
// must reserve this field for operators, including deletion/replacement updates.
func routingProfilesFromMetadata(layers ...map[string]interface{}) ([]string, error) {
	const field = "air_routing_profiles"
	const maxProfiles = 32
	var ids []string
	seen := make(map[string]bool)
	for _, metadata := range layers {
		value, exists := metadata[field]
		if !exists {
			continue
		}
		var values []string
		switch v := value.(type) {
		case []string:
			values = v
		case []interface{}:
			if len(v) > maxProfiles {
				return nil, errRoutingProfile
			}
			for _, item := range v {
				id, ok := item.(string)
				if !ok {
					return nil, errRoutingProfile
				}
				values = append(values, id)
			}
		default:
			return nil, errRoutingProfile
		}
		if len(values) > maxProfiles {
			return nil, errRoutingProfile
		}
		for _, id := range values {
			if !config.ValidRoutingProfileID(id) {
				return nil, errRoutingProfile
			}
			if !seen[id] {
				ids = append(ids, id)
				seen[id] = true
			}
		}
	}
	return ids, nil
}

func (p *Proxy) applyRoutingProfile(logCtx *RequestLogContext, modelID string) {
	if logCtx.TokenInfo == nil || logCtx.TokenInfo.IsMasterKey {
		return
	}
	info := logCtx.TokenInfo
	ids, err := routingProfilesFromMetadata(info.OrganizationMetadata, info.TeamMetadata, info.Metadata)
	if err == nil {
		logCtx.Scope, err = p.routingProfiles.Restrict(logCtx.Scope, ids, logCtx.PublicModelID, modelID)
	}
	if err != nil {
		logCtx.Scope = logCtx.Scope.RestrictCredentials(nil)
		p.logger.WarnContext(logCtx.Context(), "Routing profile binding is invalid or unavailable")
	}
}

// Defense in depth at the send boundary. Normal selections already apply this
// filter through CredentialConfig.VisibleTo, including exact/sticky picks.
func checkRoutingCredential(logCtx *RequestLogContext, credential *config.CredentialConfig) error {
	if logCtx != nil && logCtx.Scope.HasCredentialRestriction() && (credential == nil || !credential.VisibleTo(logCtx.Scope)) {
		return errRoutingProfile
	}
	return nil
}

func rejectRoutingSend(w http.ResponseWriter, logCtx *RequestLogContext) {
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusServiceUnavailable
	logCtx.ErrorMsg = "No permitted route available"
	WriteErrorServiceUnavailable(w, logCtx.ErrorMsg)
}
