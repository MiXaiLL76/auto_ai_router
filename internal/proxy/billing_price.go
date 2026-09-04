package proxy

import "github.com/mixaill76/auto_ai_router/internal/models"

// lookupBillingModelPrice resolves the price row to bill a request against.
// Candidates are tried strictly in the order publicModelID, modelID,
// realModelID (client-facing alias first, provider deployment name last) —
// callers must preserve this argument order, since it IS the priority
// contract; passing the strings in a different order silently changes which
// price wins.
//
// The underlying GetPriceAny tries each candidate in full (exact raw-lowercase
// key, then normalised/provider-prefix-stripped key) before moving to the next
// candidate, so a higher-priority candidate's normalised match always wins
// over a lower-priority candidate's raw match — e.g. a provider-prefixed
// publicModelID whose normalised form collides with a cheaper sibling (e.g.
// "openrouter/gpt-5-mini" → "gpt-5-mini") is still found before falling
// through to modelID, while explicit raw entries like
// "google/gemini-3-flash-preview-highlimits" still win over the normalised
// fallback for that same candidate.
func lookupBillingModelPrice(registry *models.ModelPriceRegistry, publicModelID, modelID, realModelID string) (string, *models.ModelPrice) {
	if registry == nil {
		return modelID, nil
	}

	if realModelID == modelID {
		realModelID = ""
	}

	if matchedID, modelPrice := registry.GetPriceAny(publicModelID, modelID, realModelID); modelPrice != nil {
		return matchedID, modelPrice
	}

	return modelID, nil
}

// resolveBillingPrice resolves and caches the billing price on logCtx so that
// budget reservation (called once per request, before the provider call) and
// final spend logging (called once per request, after the response) agree on
// the exact same price row instead of independently re-querying the price
// registry — which could otherwise straddle a concurrent registry reload and
// bill the reservation and the final spend log against different prices.
func (p *Proxy) resolveBillingPrice(logCtx *RequestLogContext, publicModelID, modelID, realModelID string) (string, *models.ModelPrice) {
	if logCtx.billingPriceResolved {
		return logCtx.billingPriceModelID, logCtx.billingPrice
	}
	priceModelID, modelPrice := lookupBillingModelPrice(p.priceRegistry, publicModelID, modelID, realModelID)
	logCtx.billingPriceResolved = true
	logCtx.billingPriceModelID = priceModelID
	logCtx.billingPrice = modelPrice
	return priceModelID, modelPrice
}
