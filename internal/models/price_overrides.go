package models

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

type priceField struct {
	index int
	value float64
}
type modelPatches map[string][]priceField
type priceOverrides map[string]map[string]modelPatches

// Compile field offsets once at load time. Only scalar cost fields are editable.
func (o *priceOverrides) UnmarshalJSON(data []byte) error {
	var groups map[string]map[string]map[string]map[string]*float64
	if err := json.Unmarshal(data, &groups); err != nil {
		return fmt.Errorf("invalid $overrides: %w", err)
	}
	if groups == nil {
		return fmt.Errorf("$overrides must be an object")
	}
	fields := map[string]int{}
	for f := range reflect.TypeFor[ModelPrice]().Fields() {
		name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
		if f.Type.Kind() == reflect.Float64 && strings.Contains(name, "cost") {
			fields[name] = f.Index[0]
		}
	}
	parsed := priceOverrides{}
	for group, owners := range groups {
		if (group != "teams" && group != "organizations" && group != "keys") || owners == nil {
			return fmt.Errorf("invalid price override group %q", group)
		}
		parsed[group] = map[string]modelPatches{}
		for owner, models := range owners {
			if owner == "" || models == nil {
				return fmt.Errorf("invalid owner in %s", group)
			}
			patches := modelPatches{}
			for model, row := range models {
				id := strings.ToLower(strings.TrimSpace(model))
				if _, dup := patches[id]; dup || id == "" || row == nil {
					return fmt.Errorf("invalid or duplicate override for model %q", model)
				}
				var patch []priceField
				for name, value := range row {
					i, ok := fields[name]
					if !ok || value == nil || *value < 0 {
						return fmt.Errorf("invalid price override for model %q field %q", model, name)
					}
					patch = append(patch, priceField{i, *value})
				}
				patches[id] = patch
			}
			parsed[group][owner] = patches
		}
	}
	*o = parsed
	return nil
}

// BillingPrices pins the registry generation and the caller's applicable patches.
type BillingPrices struct {
	prices map[string]*ModelPrice
	layers [3]modelPatches
}

func (r *ModelPriceRegistry) BillingPrices(org, team, key string) *BillingPrices {
	if r == nil {
		return &BillingPrices{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return &BillingPrices{
		prices: r.prices,
		layers: [3]modelPatches{
			r.overrides["organizations"][org],
			r.overrides["teams"][team],
			r.overrides["keys"][key],
		},
	}
}

func (b *BillingPrices) Apply(id string, base *ModelPrice) *ModelPrice {
	if base == nil {
		return nil
	}
	price := base
	for _, layer := range b.layers {
		if len(layer) == 0 {
			continue
		}
		for _, field := range layer[strings.ToLower(strings.TrimSpace(id))] {
			if price == base {
				copyPrice := *base
				price = &copyPrice
			}
			reflect.ValueOf(price).Elem().Field(field.index).SetFloat(field.value)
		}
	}
	return price
}

func (b *BillingPrices) Lookup(ids ...string) (string, *ModelPrice) {
	for _, id := range ids {
		if id == "" {
			continue
		}
		raw := strings.ToLower(strings.TrimSpace(id))
		if price := b.prices[raw]; price != nil {
			return id, b.Apply(raw, price)
		}
		normalized := NormalizeModelName(id)
		if price := b.prices[normalized]; price != nil {
			return id, b.Apply(normalized, price)
		}
	}
	return "", nil
}
