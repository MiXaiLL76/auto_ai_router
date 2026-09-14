package models

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// maxImageParamValueLen bounds request parameter values considered for price
// tiers: tiers match short enumerations ("2k", "medium", true), never prompts.
const maxImageParamValueLen = 64

// ImagePriceTier is one per-image price rule for image generation/edit
// requests. Every condition it declares must hold; conditions it omits match
// anything.
type ImagePriceTier struct {
	// Operation restricts the tier to "generation" or "edit" requests.
	Operation string `json:"operation,omitempty"`
	// When lists request parameters and the values they must have. Names are
	// exact; string values compare case-insensitively, and the JSON type must
	// match (true is not "true"). ModelPrice.ImageRequestDefaults fill in
	// parameters the request omitted.
	When ImageRequestParams `json:"when,omitempty"`
	// MaxPixels limits the tier to images of at most this many pixels
	// (width×height of the delivered image). Images of unknown size never
	// match a tier that sets it.
	MaxPixels int64 `json:"max_pixels,omitempty"`
	// OutputCostPerImage is the price of one delivered image in this tier.
	OutputCostPerImage float64 `json:"output_cost_per_image"`
}

// ImagePriceTiers is an ordered tier list; the first matching tier wins.
//
// A tier listed after one that covers it (see ImagePriceTier.covers) could
// never match, so a list written as "base price, then discounts" would bill
// every image at the base price. Such a tier is moved just ahead of the first
// tier covering it, which only changes the price of the images it was written
// for; a tier with exactly the conditions of an earlier one is dropped, and the
// earlier one keeps winning. Tiers that merely overlap keep their listed order.
// Both repairs are logged so the price map can be fixed.
//
// It unmarshals leniently: the whole price map is decoded in one pass, so a
// malformed tier must not take every other model's price down with it. A tier
// that fails validation is dropped as a whole (never just its bad condition,
// which would widen it to more images) and logged; an image that then matches
// no tier falls back to the model's flat output_cost_per_image.
type ImagePriceTiers []ImagePriceTier

func (t *ImagePriceTiers) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		warnImagePricingOnce("Ignoring output_cost_per_image_tiers: not an array", data, err)
		*t = nil
		return nil
	}
	tiers := make(ImagePriceTiers, 0, len(raw))
	// kept[i] is the JSON of tiers[i], quoted when a later tier conflicts with it.
	kept := make([]json.RawMessage, 0, len(raw))
	for _, item := range raw {
		tier, err := decodeImagePriceTier(item)
		if err != nil {
			warnImagePricingOnce("Ignoring malformed image price tier", item, err)
			continue
		}
		index := tiers.firstCovering(tier)
		switch {
		case index < 0:
			tiers = append(tiers, tier)
			kept = append(kept, item)
		case tier.covers(tiers[index]):
			warnImagePricingOnce("Ignoring image price tier with the same conditions as an earlier tier", item,
				fmt.Errorf("same conditions as %s", bytes.TrimSpace(kept[index])))
		default:
			warnImagePricingOnce("Image price tier moved ahead of a tier that covers it", item,
				fmt.Errorf("listed after %s, which matches every image it does", bytes.TrimSpace(kept[index])))
			tiers = slices.Insert(tiers, index, tier)
			kept = slices.Insert(kept, index, item)
		}
	}
	*t = tiers
	return nil
}

// covers reports whether t matches every image other matches, whatever the
// request and image size — so other, listed after t, could never be reached.
// Request defaults don't widen coverage: a request can always set a parameter
// to something other than its default.
func (t ImagePriceTier) covers(other ImagePriceTier) bool {
	if t.Operation != "" && t.Operation != other.Operation {
		return false
	}
	for key, want := range t.When {
		if got, ok := other.When[key]; !ok || got != want {
			return false
		}
	}
	return t.MaxPixels == 0 || (other.MaxPixels > 0 && other.MaxPixels <= t.MaxPixels)
}

// firstCovering returns the index of the first tier covering tier, or -1.
func (t ImagePriceTiers) firstCovering(tier ImagePriceTier) int {
	for i, candidate := range t {
		if candidate.covers(tier) {
			return i
		}
	}
	return -1
}

var errImageTierInvalid = errors.New("invalid image price tier")

// imagePricingWarnings remembers warnings already logged: the price map is
// re-decoded on every sync interval, so a persistent config mistake must not
// flood the log.
var imagePricingWarnings sync.Map

// warnImagePricingOnce logs a dropped or repaired image pricing value once per
// process, quoting the offending JSON so it can be found in the price map.
func warnImagePricingOnce(msg string, raw []byte, err error) {
	snippet := string(bytes.TrimSpace(raw))
	if len(snippet) > 200 {
		snippet = snippet[:200] + "..."
	}
	key := msg + "\x00" + snippet + "\x00" + fmt.Sprint(err)
	if _, loaded := imagePricingWarnings.LoadOrStore(key, struct{}{}); loaded {
		return
	}
	slog.Warn(msg, "value", snippet, "error", err)
}

// decodeImagePriceTier decodes and validates one tier strictly: unknown keys
// (a typo such as "max_pixel"), non-scalar conditions, an unknown operation or
// a missing/negative price all reject the tier.
func decodeImagePriceTier(item json.RawMessage) (ImagePriceTier, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(item, &fields); err != nil {
		return ImagePriceTier{}, fmt.Errorf("%w: tier must be an object", errImageTierInvalid)
	}
	cost, ok := fields["output_cost_per_image"]
	if !ok || bytes.Equal(bytes.TrimSpace(cost), []byte("null")) {
		return ImagePriceTier{}, fmt.Errorf("%w: output_cost_per_image is required", errImageTierInvalid)
	}
	if when, hasWhen := fields["when"]; hasWhen {
		if err := validateImageParamsObject(when); err != nil {
			return ImagePriceTier{}, fmt.Errorf("%w: when: %v", errImageTierInvalid, err)
		}
	}

	var tier ImagePriceTier
	decoder := json.NewDecoder(bytes.NewReader(item))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&tier); err != nil {
		return ImagePriceTier{}, fmt.Errorf("%w: %v", errImageTierInvalid, err)
	}
	tier.Operation = strings.ToLower(strings.TrimSpace(tier.Operation))
	switch {
	case tier.Operation != "" && tier.Operation != converter.ImageOperationGeneration && tier.Operation != converter.ImageOperationEdit:
		return ImagePriceTier{}, fmt.Errorf("%w: unknown operation %q", errImageTierInvalid, tier.Operation)
	case tier.OutputCostPerImage < 0:
		return ImagePriceTier{}, fmt.Errorf("%w: negative output_cost_per_image", errImageTierInvalid)
	case tier.MaxPixels < 0:
		return ImagePriceTier{}, fmt.Errorf("%w: negative max_pixels", errImageTierInvalid)
	}
	return tier, nil
}

// validateImageParamsObject requires an object of scalar values usable as
// request parameter conditions.
func validateImageParamsObject(raw json.RawMessage) error {
	var values map[string]any
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		return errors.New("must be an object")
	}
	for key, value := range values {
		if strings.TrimSpace(key) == "" {
			return errors.New("empty parameter name")
		}
		if _, ok := NormalizeImageParamValue(value); !ok {
			return fmt.Errorf("parameter %q must be a short string, boolean or number", key)
		}
	}
	return nil
}

// ImageRequestParams maps request parameter names to canonical values (see
// NormalizeImageParamValue). It unmarshals leniently from an object of
// scalars, skipping values that can't be compared, and marshals back to the
// same JSON so prices round-trip unchanged.
type ImageRequestParams map[string]string

func (p *ImageRequestParams) UnmarshalJSON(data []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		warnImagePricingOnce("Ignoring image request parameters: not an object", data, err)
		*p = nil
		return nil
	}
	params := make(ImageRequestParams, len(raw))
	for key, value := range raw {
		name := strings.TrimSpace(key)
		normalized, ok := NormalizeImageParamValue(value)
		if name == "" || !ok {
			warnImagePricingOnce("Ignoring image request parameter that is not a short scalar", data, fmt.Errorf("parameter %q", key))
			continue
		}
		params[name] = normalized
	}
	*p = params
	return nil
}

func (p ImageRequestParams) MarshalJSON() ([]byte, error) {
	if p == nil {
		return []byte("null"), nil
	}
	raw := make(map[string]json.RawMessage, len(p))
	for key, value := range p {
		if !json.Valid([]byte(value)) {
			encoded, err := json.Marshal(value)
			if err != nil {
				return nil, err
			}
			value = string(encoded)
		}
		raw[key] = json.RawMessage(value)
	}
	return json.Marshal(raw)
}

// NormalizeImageParamValue turns a decoded JSON scalar into the canonical form
// used to match price tiers: the JSON literal of the value, with strings
// trimmed and lower-cased. Keeping the literal keeps types apart, so a string
// "true" never matches a boolean condition. ok=false for values that can't
// select a tier (objects, arrays, null, empty or long strings).
func NormalizeImageParamValue(value any) (string, bool) {
	switch v := value.(type) {
	case string:
		s := strings.ToLower(strings.TrimSpace(v))
		if s == "" || len(s) > maxImageParamValueLen {
			return "", false
		}
		encoded, err := json.Marshal(s)
		if err != nil {
			return "", false
		}
		return string(encoded), true
	case bool:
		return strconv.FormatBool(v), true
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), true
	case json.Number:
		f, err := v.Float64()
		if err != nil {
			return "", false
		}
		return strconv.FormatFloat(f, 'f', -1, 64), true
	default:
		return "", false
	}
}

// NormalizeImageFormValue canonicalizes a multipart form field. Form fields are
// untyped, so booleans and numbers are read the way providers parse them.
func NormalizeImageFormValue(value string) (string, bool) {
	trimmed := strings.TrimSpace(value)
	switch strings.ToLower(trimmed) {
	case "true":
		return NormalizeImageParamValue(true)
	case "false":
		return NormalizeImageParamValue(false)
	}
	if f, err := strconv.ParseFloat(trimmed, 64); err == nil {
		return NormalizeImageParamValue(f)
	}
	return NormalizeImageParamValue(trimmed)
}

// validateStrictImagePricing rejects image pricing fields that the lenient
// decoders above would have partly dropped or reordered, for loaders that must
// fail closed on any malformed tariff.
func validateStrictImagePricing(fields map[string]json.RawMessage, price *ModelPrice) error {
	if raw, ok := fields["output_cost_per_image_tiers"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return errors.New("output_cost_per_image_tiers must be an array")
		}
		tiers := make(ImagePriceTiers, 0, len(items))
		for i, item := range items {
			tier, err := decodeImagePriceTier(item)
			if err != nil {
				return fmt.Errorf("output_cost_per_image_tiers[%d]: %w", i, err)
			}
			if covering := tiers.firstCovering(tier); covering >= 0 {
				return fmt.Errorf("output_cost_per_image_tiers[%d]: %w: output_cost_per_image_tiers[%d] matches every image it does, so it can never apply; list tiers from most to least specific",
					i, errImageTierInvalid, covering)
			}
			tiers = append(tiers, tier)
		}
	}
	if raw, ok := fields["image_request_defaults"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if err := validateImageParamsObject(raw); err != nil {
			return fmt.Errorf("image_request_defaults: %w", err)
		}
	}
	if price.InputImagesFreePerRequest < 0 {
		return errors.New("input_images_free_per_request must not be negative")
	}
	if price.InputCostPerImage < 0 {
		return errors.New("input_cost_per_image must not be negative")
	}
	return nil
}

// outputImageCost prices imageCount delivered images. Without tiers it is the
// flat imageCount × OutputCostPerImage; with tiers each image is priced by the
// first tier matching the request and that image's pixel count.
func (p *ModelPrice) outputImageCost(imageCount int, details *converter.ImageBillingDetails) float64 {
	if imageCount <= 0 {
		return 0
	}
	if len(p.OutputCostPerImageTiers) == 0 {
		return float64(imageCount) * p.OutputCostPerImage
	}
	total := 0.0
	for i := 0; i < imageCount; i++ {
		var pixels int64
		if details != nil && i < len(details.OutputPixels) {
			pixels = details.OutputPixels[i]
		}
		total += p.outputCostPerImage(details, pixels)
	}
	return total
}

// outputCostPerImage returns the price of a single delivered image.
func (p *ModelPrice) outputCostPerImage(details *converter.ImageBillingDetails, pixels int64) float64 {
	for _, tier := range p.OutputCostPerImageTiers {
		if p.imageTierMatches(tier, details, pixels) {
			return tier.OutputCostPerImage
		}
	}
	return p.OutputCostPerImage
}

func (p *ModelPrice) imageTierMatches(tier ImagePriceTier, details *converter.ImageBillingDetails, pixels int64) bool {
	if tier.Operation != "" && (details == nil || tier.Operation != details.Operation) {
		return false
	}
	for key, want := range tier.When {
		if got, ok := p.imageRequestParam(details, key); !ok || got != want {
			return false
		}
	}
	if tier.MaxPixels > 0 && (pixels <= 0 || pixels > tier.MaxPixels) {
		return false
	}
	return true
}

func (p *ModelPrice) imageRequestParam(details *converter.ImageBillingDetails, key string) (string, bool) {
	if details != nil {
		if value, ok := details.RequestParams[key]; ok {
			return value, true
		}
	}
	value, ok := p.ImageRequestDefaults[key]
	return value, ok
}

// inputImageCost prices the source images of an image request beyond the
// free allowance.
func (p *ModelPrice) inputImageCost(details *converter.ImageBillingDetails) float64 {
	if details == nil || p.InputCostPerImage <= 0 {
		return 0
	}
	free := p.InputImagesFreePerRequest
	if free < 0 {
		free = 0
	}
	billable := details.InputImages - free
	if billable <= 0 {
		return 0
	}
	return float64(billable) * p.InputCostPerImage
}
