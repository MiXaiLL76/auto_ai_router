package proxy

import "github.com/mixaill76/auto_ai_router/internal/models"

func lookupBillingModelPrice(registry *models.ModelPriceRegistry, publicModelID, modelID, realModelID string) (string, *models.ModelPrice) {
	if registry == nil {
		return modelID, nil
	}

	if publicModelID != "" {
		modelPrice := registry.GetPrice(publicModelID)
		if modelPrice != nil {
			return publicModelID, modelPrice
		}
	}

	modelPrice := registry.GetPrice(modelID)
	if modelPrice != nil {
		return modelID, modelPrice
	}

	if realModelID != "" && realModelID != modelID {
		modelPrice = registry.GetPrice(realModelID)
		if modelPrice != nil {
			return realModelID, modelPrice
		}
	}

	return modelID, nil
}
