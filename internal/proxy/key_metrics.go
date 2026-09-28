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

// withKeyTurn gives one WebSocket turn its own key identity slot, so the turn
// is counted like a standalone HTTP request. finish records the turn with its
// final status (0 = implicit 200) and is a no-op if the turn never
// authenticated. The caller decides when a turn is a client abort (499): a
// hijacked request's context is not canceled when the client disconnects, so
// ctx cannot tell.
func (p *Proxy) withKeyTurn(ctx context.Context) (context.Context, func(status int)) {
	if p.keyMetrics == nil {
		return ctx, func(int) {}
	}
	turnCtx, identity := monitoring.WithKeyIdentitySlot(ctx)
	return turnCtx, func(status int) {
		if id, ok := identity(); ok {
			p.keyMetrics.ObserveStatus(id, status)
		}
	}
}
