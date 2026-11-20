package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ToolPricesConfig is the router's internal tool price table: the price of
// the tools the ROUTER runs on the model's behalf (the provider's own
// Search/Fetch endpoints), as opposed to the price of the model's tokens.
//
// It is a separate document from the model prices on purpose. The model
// prices answer "what does a token cost"; these answer "what does the
// router's own call to a provider's search API cost", and the two are
// billed from different counters. Keeping them apart also keeps the model
// price file in the LiteLLM price format, which other tools read.
//
// The document:
//
//	{
//	  "tool_prices": {
//	    "moonshot": {
//	      "search_cost_per_call": 0.013,
//	      "search_pro_cost_per_call": 0.013,
//	      "fetch_cost_per_call": 0.013,
//	      "rate": 1.3
//	    }
//	  }
//	}
//
// The three cost_per_call values are the FINAL client price: the rate is
// already folded into them, and it is NOT applied again. The rate is kept in
// the document because it is what the rest of the tariff is derived from --
// the same shape the model prices use -- and so that a reader can see that
// $0.013 is $0.01 x 1.3 and not a typo.
type ToolPricesConfig struct {
	// ToolPrices holds the per-provider tool prices, keyed by the provider
	// name ("moonshot").
	ToolPrices map[string]ToolPricesProvider `json:"tool_prices"`
}

// ToolPricesProvider is one provider's tool prices.
type ToolPricesProvider struct {
	// SearchCostPerCall is the price of one successful Basic search.
	SearchCostPerCall float64 `json:"search_cost_per_call"`
	// SearchProCostPerCall is the price of one successful Pro search.
	SearchProCostPerCall float64 `json:"search_pro_cost_per_call"`
	// FetchCostPerCall is the price of one successful fetch.
	FetchCostPerCall float64 `json:"fetch_cost_per_call"`
	// Rate is the markup the provider's own price was multiplied by to
	// reach the cost_per_call values. It is documentation: the costs above
	// are final and the rate is not applied again.
	Rate float64 `json:"rate"`
}

// MoonshotToolPrices returns Moonshot's tool prices, and whether the
// document named them at all.
//
// A provider that is absent from the document has no tool prices: its
// router-run tools are free, which is the same thing as a price of zero.
func (c *ToolPricesConfig) MoonshotToolPrices() (ToolPricesProvider, bool) {
	if c == nil || len(c.ToolPrices) == 0 {
		return ToolPricesProvider{}, false
	}
	prices, ok := c.ToolPrices["moonshot"]
	if !ok {
		return ToolPricesProvider{}, false
	}
	return prices, true
}

// ValidateToolPrices checks the tool price document.
//
// The rule that matters: a cost_per_call is the FINAL client price, so the
// rate must not be applied to it again. A document that lists a rate and a
// cost that does not follow from it is rejected rather than silently
// double-charged -- the mistake is expensive and invisible in the total.
func ValidateToolPrices(doc *ToolPricesConfig) error {
	if doc == nil {
		return nil
	}
	for provider, prices := range doc.ToolPrices {
		for name, cost := range map[string]float64{
			"search_cost_per_call":     prices.SearchCostPerCall,
			"search_pro_cost_per_call": prices.SearchProCostPerCall,
			"fetch_cost_per_call":      prices.FetchCostPerCall,
		} {
			if cost < 0 {
				return fmt.Errorf("tool_prices.%s.%s must not be negative, got: %v", provider, name, cost)
			}
		}
		if prices.Rate < 0 {
			return fmt.Errorf("tool_prices.%s.rate must not be negative, got: %v", provider, prices.Rate)
		}
	}
	return nil
}

// LoadToolPrices loads the tool price document from a file path.
//
// The path is the operator's (a CLI flag or an environment variable), and it
// is validated the same way the model price list's path is: a traversal
// segment is refused, and the file's size is bounded.
func LoadToolPrices(path string) (*ToolPricesConfig, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	if hasPathTraversal(path) {
		return nil, fmt.Errorf("tool prices path contains traversal segments: %s", path)
	}
	cleanPath := filepath.Clean(path)
	stat, err := os.Stat(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("reading tool prices file: %w", err)
	}
	if stat.Size() > MaxFileSizeBytes {
		return nil, fmt.Errorf("tool prices file exceeds %d bytes: %d", MaxFileSizeBytes, stat.Size())
	}
	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("reading tool prices file: %w", err)
	}
	return ParseToolPrices(data)
}

// ParseToolPrices parses and validates the tool price document.
func ParseToolPrices(data []byte) (*ToolPricesConfig, error) {
	var doc ToolPricesConfig
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("failed to parse tool prices JSON: %w", err)
	}
	if err := ValidateToolPrices(&doc); err != nil {
		return nil, err
	}
	return &doc, nil
}
