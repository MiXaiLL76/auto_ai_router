package proxy

import (
	"context"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

// keyIdentityFromTokenInfo maps a validated token to its per-key metrics
// identity. Only a prefix of the stored hash is exposed (see
// monitoring.KeyHashPrefixLen); the master key never exposes any part of its
// hash, since HashToken of a non-"sk-" master key is the key itself.
func keyIdentityFromTokenInfo(info *models.TokenInfo) monitoring.KeyIdentity {
	if info == nil {
		return monitoring.KeyIdentity{}
	}
	if info.IsMasterKey {
		return monitoring.KeyIdentity{Key: monitoring.KeyLabelMaster, KeyAlias: monitoring.KeyLabelMaster}
	}
	if len(info.Token) < monitoring.KeyHashPrefixLen {
		return monitoring.KeyIdentity{}
	}
	return monitoring.KeyIdentity{
		Key:            info.Token[:monitoring.KeyHashPrefixLen],
		KeyAlias:       info.KeyAlias,
		UserID:         info.UserID,
		UserEmail:      info.UserEmail,
		TeamID:         info.TeamID,
		TeamAlias:      info.TeamAlias,
		OrganizationID: info.OrganizationID,
	}
}

// noteRequestKey attributes the current request to info's key for per-key
// metrics; the enclosing HTTP middleware or WebSocket turn records it.
func (p *Proxy) noteRequestKey(ctx context.Context, info *models.TokenInfo) {
	if p.keyMetrics == nil || info == nil {
		return
	}
	monitoring.SetRequestKeyIdentity(ctx, keyIdentityFromTokenInfo(info))
}

// observeKeyRequest counts one finished request directly, for flows that are
// not an HTTP request of their own (native WebSocket turns).
func (p *Proxy) observeKeyRequest(info *models.TokenInfo, statusCode int) {
	if p.keyMetrics == nil || info == nil {
		return
	}
	p.keyMetrics.ObserveStatus(keyIdentityFromTokenInfo(info), statusCode)
}
