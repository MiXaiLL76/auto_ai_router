package models

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"reflect"
	"strings"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// teamModelPricesCacheSize bounds the parsed air_model_prices fields kept for
// reuse. One entry serves every request carrying the same field.
const teamModelPricesCacheSize = 1024

// teamModelPricesSource names a team's air_model_prices in price row errors.
const teamModelPricesSource = "air_model_prices"

// teamModelPrices is a parsed air_model_prices field, or why it is unusable.
type teamModelPrices struct {
	prices map[string]*ModelPrice
	err    error
}

var teamModelPricesCache = newTeamModelPricesCache()

func newTeamModelPricesCache() *lru.Cache[string, teamModelPrices] {
	cache, err := lru.New[string, teamModelPrices](teamModelPricesCacheSize)
	if err != nil {
		panic(fmt.Sprintf("models: failed to create team model prices cache: %v", err))
	}
	return cache
}

// DecodeTeamModelPrices parses a team's metadata.air_model_prices: an object
// of price rows keyed by the model name clients request. Any error makes the
// whole field unusable, so the team's requests are refused instead of being
// billed at the default price. A field is parsed once and the result is
// shared by every request carrying it: callers must not modify it.
func DecodeTeamModelPrices(raw json.RawMessage) (map[string]*ModelPrice, error) {
	key := string(raw)
	if parsed, ok := teamModelPricesCache.Get(key); ok {
		return parsed.prices, parsed.err
	}
	prices, err := decodeTeamModelPrices(raw)
	teamModelPricesCache.Add(key, teamModelPrices{prices: prices, err: err})
	return prices, err
}

func decodeTeamModelPrices(raw json.RawMessage) (map[string]*ModelPrice, error) {
	var rows map[string]json.RawMessage
	if err := json.Unmarshal(raw, &rows); err != nil {
		return nil, fmt.Errorf("air_model_prices must be an object: %w", err)
	}
	if rows == nil {
		return nil, errors.New("air_model_prices must be an object, not null")
	}
	prices := make(map[string]*ModelPrice, len(rows))
	for modelID, row := range rows {
		if strings.TrimSpace(modelID) == "" {
			return nil, errors.New("air_model_prices: empty model name")
		}
		price, err := decodeStrictPriceRow(teamModelPricesSource, modelID, row)
		if err != nil {
			return nil, err
		}
		if err := validatePriceOnlyRow(row, price); err != nil {
			return nil, fmt.Errorf("%s %q: %w", teamModelPricesSource, modelID, err)
		}
		prices[modelID] = price
	}
	return prices, nil
}

// TeamModelPrice is the row a request billed at a team model price is billed
// by: the general row with every price replaced by the team row's. A price
// the team row does not list is absent, as in any price row, so the team row
// is the whole price of the model. How the model is billed (litellm_provider,
// image_request_defaults, ...) stays the general row's, and so do the
// router's own Moonshot tool prices, which are not the model's price.
func TeamModelPrice(general, team *ModelPrice) *ModelPrice {
	price := *general
	source := reflect.ValueOf(team).Elem()
	target := reflect.ValueOf(&price).Elem()
	for _, i := range priceFieldIndexes() {
		target.Field(i).Set(source.Field(i))
	}
	price.ToolCostPerCall = maps.Clone(team.ToolCostPerCall)
	for tool := range moonshotToolPrices {
		cost, ok := general.ToolCostPerCall[tool]
		if !ok {
			continue
		}
		if price.ToolCostPerCall == nil {
			price.ToolCostPerCall = make(map[string]float64, len(moonshotToolPrices))
		}
		price.ToolCostPerCall[tool] = cost
	}
	return &price
}

// priceFieldIndexes are the ModelPrice fields that hold prices (see
// modelPriceJSONFields).
var priceFieldIndexes = sync.OnceValue(func() []int {
	known := modelPriceJSONFields()
	t := reflect.TypeOf(ModelPrice{})
	var indexes []int
	for i := 0; i < t.NumField(); i++ {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("json"), ",")
		if known[name] {
			indexes = append(indexes, i)
		}
	}
	return indexes
})

// validatePriceOnlyRow checks what a team row adds to the strict price row
// format: it holds ready prices only, so it cannot change how a model is
// billed. No field other than a price, no null, no negative amount, and Web
// Search prices under keys the calculator looks up.
func validatePriceOnlyRow(row json.RawMessage, price *ModelPrice) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(row, &fields); err != nil {
		return err
	}
	known := modelPriceJSONFields()
	for field := range fields {
		if !known[field] {
			return fmt.Errorf("%q is not a price field", field)
		}
	}
	if err := rejectNullsAndNegativeAmounts(fields); err != nil {
		return err
	}
	return validateSearchContextSizes(price)
}

// rejectNullsAndNegativeAmounts rejects a null or a negative number in the
// fields of a price row: each of their numbers is an amount or a size. The
// one exception is the when conditions of image price tiers: they may hold
// any request value, such as a seed of -1, and validateStrictImagePricing
// checks them.
func rejectNullsAndNegativeAmounts(fields map[string]json.RawMessage) error {
	for field, value := range fields {
		var tiers []map[string]json.RawMessage
		if field != "output_cost_per_image_tiers" || bytes.Equal(bytes.TrimSpace(value), []byte("null")) || json.Unmarshal(value, &tiers) != nil {
			if err := rejectNullsAndNegativeNumbers(value); err != nil {
				return fmt.Errorf("%q: %w", field, err)
			}
			continue
		}
		for i, tier := range tiers {
			for key, tierValue := range tier {
				if key == "when" {
					continue
				}
				if err := rejectNullsAndNegativeNumbers(tierValue); err != nil {
					return fmt.Errorf("%q[%d].%s: %w", field, i, key, err)
				}
			}
		}
	}
	return nil
}

// rejectNullsAndNegativeNumbers rejects a null or a negative number anywhere
// in the JSON value.
func rejectNullsAndNegativeNumbers(value json.RawMessage) error {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		switch token := token.(type) {
		case nil:
			return errors.New("null value")
		case json.Number:
			if strings.HasPrefix(token.String(), "-") {
				return fmt.Errorf("negative amount %s", token)
			}
		}
	}
}

// validateSearchContextSizes rejects a Web Search price under a key the
// calculator never looks up, which would bill every search at zero.
func validateSearchContextSizes(price *ModelPrice) error {
	for key := range price.SearchContextCostPerQuery {
		size, ok := strings.CutPrefix(key, searchContextSizeKeyPrefix)
		if !ok || converter.NormalizeWebSearchContextSize(size) != size {
			return fmt.Errorf("search_context_cost_per_query: unknown key %q (%slow, _medium or _high)", key, searchContextSizeKeyPrefix)
		}
	}
	return nil
}
