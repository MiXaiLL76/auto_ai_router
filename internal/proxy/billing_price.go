package proxy

import (
	"net/http"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/models"
)

// teamModelPriceSource is the spend log's billing_price_source of a request
// billed at its team's model price.
const teamModelPriceSource = "team_model_prices"

// lookupBillingModelPrice resolves the price row to bill a request against.
// Candidates are tried strictly in the order publicModelID, modelID,
// realModelID (client-facing alias first, provider deployment name last) —
// callers must preserve this argument order, since it IS the priority
// contract; passing the strings in a different order silently changes which
// price wins.
//
// The underlying GetPriceAny tries each candidate in full (exact raw-lowercase
// key, then normalised/prefix-stripped key) before moving to the next
// candidate, so a higher-priority candidate's normalised match always wins
// over a lower-priority candidate's raw match — e.g. a provider-prefixed
// publicModelID whose normalised form collides with a cheaper sibling (e.g.
// "openrouter/gpt-5-mini" → "gpt-5-mini") is still found before falling
// through to modelID, while explicit raw entries like
// "google/gemini-3-flash-preview-highlimits" still win over the normalised
// fallback for that same candidate.
func lookupBillingModelPrice(registry *models.ModelPriceRegistry, publicModelID, modelID, realModelID string) (string, *models.ModelPrice) {
	matchedID, modelPrice, _ := lookupBillingModelPriceWithImageToolRow(registry, publicModelID, modelID, realModelID)
	return matchedID, modelPrice
}

// lookupBillingModelPriceWithImageToolRow is lookupBillingModelPrice that also
// returns the row the found row names as its image tool's price, from the
// same generation of the price list.
func lookupBillingModelPriceWithImageToolRow(registry *models.ModelPriceRegistry, publicModelID, modelID, realModelID string) (string, *models.ModelPrice, *models.ModelPrice) {
	if registry == nil {
		return modelID, nil, nil
	}

	if realModelID == modelID {
		realModelID = ""
	}

	if matchedID, modelPrice, imageToolRow := registry.GetPriceAnyWithImageToolRow(publicModelID, modelID, realModelID); modelPrice != nil {
		return matchedID, modelPrice, imageToolRow
	}

	return modelID, nil, nil
}

// missingPriceReferenceWarnings holds the price source and referenced model
// pairs already reported missing, so a misconfigured row is logged once
// rather than on every request using it. The spend rows still carry the
// unbilled usage (e.g. image_tool_generations with a zero image_tool_cost).
var missingPriceReferenceWarnings sync.Map

// billingPriceResolver looks up price rows that the billed row refers to by
// name (ModelPrice.ImageGenerationToolModel) in the same price source the
// request is billed from: the organization tariff when one applies, the
// default price list otherwise, so one request never mixes two tariffs.
func (p *Proxy) billingPriceResolver(logCtx *RequestLogContext) models.PriceResolver {
	return func(modelID string) *models.ModelPrice {
		var price *models.ModelPrice
		priceSource := "default"
		if logCtx != nil && logCtx.OrganizationPolicy.HasCustomPricing() {
			price, _ = logCtx.OrganizationPolicy.Price(modelID)
			priceSource = logCtx.OrganizationPolicy.CacheKey()
		} else if p.priceRegistry != nil {
			price = p.priceRegistry.GetPrice(modelID)
		}
		if price != nil || p.logger == nil {
			return price
		}
		if _, warned := missingPriceReferenceWarnings.LoadOrStore(priceSource+"\x00"+modelID, struct{}{}); !warned {
			args := []any{"referenced_model", modelID, "price_source", priceSource}
			if logCtx != nil {
				args = append(args, "model", logCtx.ModelID, "request_id", logCtx.RequestID)
			}
			p.logger.Warn("Referenced price row not found; usage billed at its tariff is not charged (logged once)", args...)
		}
		return nil
	}
}

// resolveBillingPrice resolves and caches the billing price on logCtx so that
// budget reservation (called once per request, before the provider call) and
// final spend logging (called once per request, after the response) agree on
// the exact same price row instead of independently re-querying the price
// registry — which could otherwise straddle a concurrent registry reload and
// bill the reservation and the final spend log against different prices.
// With a team model price pinned (admitTeamModelPrice) the cached row is the
// price list row with the team's prices (models.TeamModelPrice).
func (p *Proxy) resolveBillingPrice(logCtx *RequestLogContext, publicModelID, modelID, realModelID string) (string, *models.ModelPrice) {
	if logCtx.billingPriceResolved {
		return logCtx.billingPriceModelID, logCtx.billingPrice
	}
	priceModelID, modelPrice, imageToolRow := lookupBillingModelPriceWithImageToolRow(p.priceRegistry, publicModelID, modelID, realModelID)
	if modelPrice != nil && logCtx.teamModelPrice != nil {
		logCtx.billedAtTeamModelPrice = true
		teamPrice := models.TeamModelPrice(modelPrice, logCtx.teamModelPrice)
		// A row naming itself (under another name) as its image tool's price
		// is not charged for those images. The team row is a new row, so
		// settle that here, against the generation of the price list the row
		// came from.
		if imageToolRow == modelPrice {
			teamPrice.ImageGenerationToolModel = ""
		}
		modelPrice = teamPrice
	}
	logCtx.billingPriceResolved = true
	logCtx.billingPriceModelID = priceModelID
	logCtx.billingPrice = modelPrice
	return priceModelID, modelPrice
}

// admitTeamModelPrice pins the team's ready price for the requested model
// (TokenInfo.TeamModelPrices) on logCtx, keyed by the model name the client
// requested. A field AIR cannot use refuses every request of the team instead
// of billing it at the default price, and so does a team price next to an
// organization tariff: one request never mixes two tariffs.
func (p *Proxy) admitTeamModelPrice(w http.ResponseWriter, r *http.Request, publicModelID string, logCtx *RequestLogContext) bool {
	if logCtx.TokenInfo == nil || logCtx.TokenInfo.TeamModelPrices == nil {
		return true
	}
	prices, err := models.DecodeTeamModelPrices(logCtx.TokenInfo.TeamModelPrices)
	price := prices[publicModelID]
	if err == nil && price == nil {
		return true
	}
	if err == nil && !logCtx.OrganizationPolicy.HasCustomPricing() {
		logCtx.teamModelPrice = price
		return true
	}
	args := []any{
		"error_code", http.StatusServiceUnavailable,
		"team_id", logCtx.TokenInfo.TeamID,
		"model", publicModelID,
		"request_id", logCtx.RequestID,
	}
	if err != nil {
		p.logger.ErrorContext(r.Context(), "Team model prices unusable; request rejected", append(args, "error", err)...)
	} else {
		p.logger.ErrorContext(r.Context(), "Team model price conflicts with the organization tariff; request rejected", args...)
	}
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusServiceUnavailable
	logCtx.ErrorMsg = "model pricing unavailable"
	logCtx.Logged = true
	WriteErrorServiceUnavailable(w, "Model pricing unavailable")
	return false
}
