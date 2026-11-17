package models

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// TestParseToolPricesReadsTheMoonshotPrices checks that the tool price
// document is read as it is written.
func TestParseToolPricesReadsTheMoonshotPrices(t *testing.T) {
	document := `{
		"tool_prices": {
			"moonshot": {
				"search_cost_per_call": 0.013,
				"search_pro_cost_per_call": 0.013,
				"fetch_cost_per_call": 0.013,
				"rate": 1.3
			}
		}
	}`
	doc, err := ParseToolPrices([]byte(document))
	if err != nil {
		t.Fatalf("ParseToolPrices: %v", err)
	}
	prices, ok := doc.MoonshotToolPrices()
	if !ok {
		t.Fatal("the document does not name Moonshot")
	}
	if prices.SearchCostPerCall != 0.013 ||
		prices.SearchProCostPerCall != 0.013 ||
		prices.FetchCostPerCall != 0.013 {
		t.Errorf("unexpected prices: %+v", prices)
	}
	if prices.Rate != 1.3 {
		t.Errorf("the rate = %v, want 1.3", prices.Rate)
	}
}

// TestParseToolPricesRejectsNegativePrices checks the validation: a
// negative price would silently refund the client.
func TestParseToolPricesRejectsNegativePrices(t *testing.T) {
	document := `{"tool_prices":{"moonshot":{"search_cost_per_call":-0.013}}}`
	if _, err := ParseToolPrices([]byte(document)); err == nil {
		t.Error("a negative price must be rejected")
	}
}

// TestParseToolPricesAcceptsAnEmptyDocument checks that a document without
// Moonshot is valid: the tools are simply free.
func TestParseToolPricesAcceptsAnEmptyDocument(t *testing.T) {
	doc, err := ParseToolPrices([]byte(`{"tool_prices":{}}`))
	if err != nil {
		t.Fatalf("ParseToolPrices: %v", err)
	}
	if _, ok := doc.MoonshotToolPrices(); ok {
		t.Error("an empty document must not name Moonshot")
	}
}

// TestLoadToolPricesReadsTheFile checks the loader.
func TestLoadToolPricesReadsTheFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tool_prices.json")
	document := `{"tool_prices":{"moonshot":{"search_cost_per_call":0.013,"search_pro_cost_per_call":0.013,"fetch_cost_per_call":0.013,"rate":1.3}}}`
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatalf("writing the document: %v", err)
	}
	doc, err := LoadToolPrices(path)
	if err != nil {
		t.Fatalf("LoadToolPrices: %v", err)
	}
	prices, ok := doc.MoonshotToolPrices()
	if !ok {
		t.Fatal("the document does not name Moonshot")
	}
	if prices.SearchCostPerCall != 0.013 {
		t.Errorf("the search price = %v, want 0.013", prices.SearchCostPerCall)
	}
}

// TestLoadToolPricesWithoutAPathIsEmpty checks that an unconfigured path is
// not an error: the tools are simply free.
func TestLoadToolPricesWithoutAPathIsEmpty(t *testing.T) {
	doc, err := LoadToolPrices("")
	if err != nil {
		t.Fatalf("LoadToolPrices: %v", err)
	}
	if doc != nil {
		t.Errorf("an absent path must load nothing, got %+v", doc)
	}
}

// TestMoonshotToolPricesRideOnTheModelRow checks that the router's own tool
// prices reach the model's row, so the ordinary cost calculation prices the
// router's calls like any other tool charge -- and only for Moonshot's own
// models.
func TestMoonshotToolPricesRideOnTheModelRow(t *testing.T) {
	registry := NewModelPriceRegistry()
	registry.SetMoonshotToolPrices(&ToolPricesProvider{
		SearchCostPerCall:    0.013,
		SearchProCostPerCall: 0.013,
		FetchCostPerCall:     0.013,
		Rate:                 1.3,
	})
	registry.Update(map[string]*ModelPrice{
		"kimi-k2": {InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6},
		"gpt-4o":  {InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6},
	})

	kimi := registry.GetPrice("kimi-k2")
	if kimi == nil {
		t.Fatal("the registry has no kimi-k2")
	}
	if kimi.ToolCostPerCall[ToolMoonshotSearch] != 0.013 ||
		kimi.ToolCostPerCall[ToolMoonshotSearchPro] != 0.013 ||
		kimi.ToolCostPerCall[ToolMoonshotFetch] != 0.013 {
		t.Errorf("the Moonshot tool prices did not reach the model's row: %+v", kimi.ToolCostPerCall)
	}

	other := registry.GetPrice("gpt-4o")
	if other == nil {
		t.Fatal("the registry has no gpt-4o")
	}
	if other.ToolCostPerCall[ToolMoonshotSearch] != 0 {
		t.Errorf("another provider's model got Moonshot's tool prices: %+v", other.ToolCostPerCall)
	}
}

// TestMoonshotToolPricesSurviveTheDBMerge checks the other path the prices
// take: a price row that comes from the database must carry them too, or a
// database-sourced Moonshot model would price the router's calls at zero.
func TestMoonshotToolPricesSurviveTheDBMerge(t *testing.T) {
	registry := NewModelPriceRegistry()
	registry.SetMoonshotToolPrices(&ToolPricesProvider{
		SearchCostPerCall:    0.013,
		SearchProCostPerCall: 0.013,
		FetchCostPerCall:     0.013,
	})
	registry.MergeDB(map[string]*ModelPrice{
		"kimi-k2": {InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6},
	})
	kimi := registry.GetPrice("kimi-k2")
	if kimi == nil {
		t.Fatal("the registry has no kimi-k2")
	}
	if kimi.ToolCostPerCall[ToolMoonshotSearch] != 0.013 {
		t.Errorf("the Moonshot tool prices did not survive the DB merge: %+v", kimi.ToolCostPerCall)
	}
}

// TestMoonshotToolCallsArePriced checks the cost calculation: the router's
// own calls are priced by their count, and the counters are the billable
// ones.
func TestMoonshotToolCallsArePriced(t *testing.T) {
	registry := NewModelPriceRegistry()
	registry.SetMoonshotToolPrices(&ToolPricesProvider{
		SearchCostPerCall:    0.013,
		SearchProCostPerCall: 0.013,
		FetchCostPerCall:     0.013,
	})
	registry.Update(map[string]*ModelPrice{
		"kimi-k2": {InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6},
	})
	price := registry.GetPrice("kimi-k2")
	if price == nil {
		t.Fatal("the registry has no kimi-k2")
	}

	usage := &converter.TokenUsage{
		PromptTokens:              100,
		CompletionTokens:          20,
		MoonshotSearchRequests:    1,
		MoonshotSearchProRequests: 1,
		MoonshotFetchRequests:     1,
	}
	costs := CalculateTokenCosts(usage, price)
	if diff := costs.MoonshotSearchCost - 0.013; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("the search cost = %v, want 0.013", costs.MoonshotSearchCost)
	}
	if diff := costs.MoonshotSearchProCost - 0.013; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("the search_pro cost = %v, want 0.013", costs.MoonshotSearchProCost)
	}
	if diff := costs.MoonshotFetchCost - 0.013; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("the fetch cost = %v, want 0.013", costs.MoonshotFetchCost)
	}
	// The three tools enter the total once, through the tool usage cost.
	wantToolCost := 0.039
	if diff := costs.ToolUsageCost - wantToolCost; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("the tool usage cost = %v, want %v", costs.ToolUsageCost, wantToolCost)
	}
	wantTotal := 100*1e-6 + 20*2e-6 + wantToolCost
	if diff := costs.TotalCost - wantTotal; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("the total cost = %v, want %v", costs.TotalCost, wantTotal)
	}
}

// TestMoonshotToolsWithoutPricesAreFree checks the other end: a router with
// no tool price document charges nothing for its own calls, rather than
// inventing a price.
func TestMoonshotToolsWithoutPricesAreFree(t *testing.T) {
	registry := NewModelPriceRegistry()
	registry.Update(map[string]*ModelPrice{
		"kimi-k2": {InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6},
	})
	price := registry.GetPrice("kimi-k2")
	if price == nil {
		t.Fatal("the registry has no kimi-k2")
	}
	usage := &converter.TokenUsage{
		PromptTokens:           100,
		CompletionTokens:       20,
		MoonshotSearchRequests: 1,
	}
	costs := CalculateTokenCosts(usage, price)
	if costs.MoonshotSearchCost != 0 {
		t.Errorf("the search cost = %v, want 0 (no price document)", costs.MoonshotSearchCost)
	}
}
