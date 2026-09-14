package models

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Prices below are provider list prices × 1.3, written exactly as a price map
// would publish them.
const imagePriceMapJSON = `{
	"pixel-tier-model": {
		"output_cost_per_image": 0.117,
		"input_cost_per_image": 0.0039,
		"input_images_free_per_request": 1,
		"output_cost_per_image_tiers": [
			{"when": {"layer_decomposition": true}, "max_pixels": 2610000, "output_cost_per_image": 0.02925},
			{"when": {"layer_decomposition": true}, "output_cost_per_image": 0.0585},
			{"max_pixels": 2610000, "output_cost_per_image": 0.0585},
			{"output_cost_per_image": 0.117}
		]
	},
	"quality-tier-model": {
		"output_cost_per_image": 0.104,
		"input_cost_per_image": 0.013,
		"image_request_defaults": {"resolution": "1k", "quality": "auto"},
		"output_cost_per_image_tiers": [
			{"operation": "edit", "when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.078},
			{"operation": "edit", "when": {"resolution": "2k", "quality": "auto"}, "output_cost_per_image": 0.104},
			{"when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.052},
			{"when": {"resolution": "2k", "quality": "auto"}, "output_cost_per_image": 0.078},
			{"when": {"resolution": "1k", "quality": "low"}, "output_cost_per_image": 0.052},
			{"when": {"resolution": "2k", "quality": "low"}, "output_cost_per_image": 0.078},
			{"when": {"resolution": "1k", "quality": "medium"}, "output_cost_per_image": 0.078},
			{"when": {"resolution": "2k", "quality": "medium"}, "output_cost_per_image": 0.104}
		]
	},
	"resolution-tier-model": {
		"output_cost_per_image": 0.091,
		"input_cost_per_image": 0.013,
		"image_request_defaults": {"resolution": "1k"},
		"output_cost_per_image_tiers": [
			{"when": {"resolution": "1k"}, "output_cost_per_image": 0.065},
			{"when": {"resolution": "2k"}, "output_cost_per_image": 0.091}
		]
	},
	"flat-model": {
		"output_cost_per_image": 0.052
	}
}`

func loadImagePrices(t *testing.T) map[string]*ModelPrice {
	t.Helper()
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(imagePriceMapJSON), &prices))
	return prices
}

func imageCost(t *testing.T, price *ModelPrice, usage converter.TokenUsage) float64 {
	t.Helper()
	costs := CalculateTokenCosts(&usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, costs.ImageCost, costs.TotalCost, 1e-12, "image requests must not pick up token costs")
	return costs.TotalCost
}

// canon builds request parameters in the canonical form the proxy produces.
func canon(values map[string]any) map[string]string {
	params := make(map[string]string, len(values))
	for key, value := range values {
		if normalized, ok := NormalizeImageParamValue(value); ok {
			params[key] = normalized
		}
	}
	return params
}

func generation(params map[string]string, pixels []int64, inputImages int) *converter.ImageBillingDetails {
	return &converter.ImageBillingDetails{
		Operation:     converter.ImageOperationGeneration,
		RequestParams: params,
		OutputPixels:  pixels,
		InputImages:   inputImages,
	}
}

func TestImagePricing_PixelTiers(t *testing.T) {
	price := loadImagePrices(t)["pixel-tier-model"]

	tests := []struct {
		name  string
		usage converter.TokenUsage
		want  float64
	}{
		{
			name:  "default 2K output above the pixel threshold",
			usage: converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{2496 * 1664}, 0)},
			want:  0.117,
		},
		{
			name:  "1K output within the pixel threshold",
			usage: converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{1152 * 864}, 0)},
			want:  0.0585,
		},
		{
			name:  "threshold is inclusive",
			usage: converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{2610000}, 0)},
			want:  0.0585,
		},
		{
			name:  "each image priced by its own size",
			usage: converter.TokenUsage{ImageCount: 2, ImageBilling: generation(nil, []int64{1024 * 1024, 2048 * 2048}, 0)},
			want:  0.0585 + 0.117,
		},
		{
			name:  "first source image is free, the rest are billed",
			usage: converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{1424 * 800}, 3)},
			want:  0.0585 + 2*0.0039,
		},
		{
			name: "layer decomposition prices base image and layers at layer rates",
			usage: converter.TokenUsage{ImageCount: 3, ImageBilling: generation(
				canon(map[string]any{"layer_decomposition": true}),
				[]int64{2048 * 2048, 1273 * 265, 492 * 98},
				1,
			)},
			want: 0.0585 + 0.02925 + 0.02925,
		},
		{
			name:  "unknown size falls back to the highest tier",
			usage: converter.TokenUsage{ImageCount: 2, ImageBilling: generation(nil, []int64{1024 * 1024}, 0)},
			want:  0.0585 + 0.117,
		},
		{
			name:  "no billing details uses the catch-all tier",
			usage: converter.TokenUsage{ImageCount: 1},
			want:  0.117,
		},
		{
			name:  "source images are not billed when nothing was delivered",
			usage: converter.TokenUsage{ImageCount: 0, ImageBilling: generation(nil, nil, 4)},
			want:  0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, imageCost(t, price, tt.usage), 1e-12)
		})
	}
}

// Expected values reproduce provider-reported costs observed live
func TestImagePricing_ResolutionAndQualityTiers(t *testing.T) {
	prices := loadImagePrices(t)
	quality := prices["quality-tier-model"]
	resolution := prices["resolution-tier-model"]

	edit := func(params map[string]string, inputImages int) *converter.ImageBillingDetails {
		return &converter.ImageBillingDetails{Operation: converter.ImageOperationEdit, RequestParams: params, InputImages: inputImages}
	}

	tests := []struct {
		name  string
		price *ModelPrice
		usage converter.TokenUsage
		want  float64
	}{
		{"omitted params use provider defaults (auto → low for generation)", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(map[string]string{}, nil, 0)}, 0.04 * 1.3},
		{"explicit low", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"quality": "low"}), nil, 0)}, 0.04 * 1.3},
		{"explicit medium", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"quality": "medium"}), nil, 0)}, 0.06 * 1.3},
		{"2k low", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"resolution": "2k", "quality": "low"}), nil, 0)}, 0.06 * 1.3},
		{"2k medium", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"resolution": "2k", "quality": "medium"}), nil, 0)}, 0.08 * 1.3},
		{"n=2 at defaults", quality,
			converter.TokenUsage{ImageCount: 2, ImageBilling: generation(map[string]string{}, nil, 0)}, 0.08 * 1.3},
		{"edit at defaults (auto → medium) plus one source image", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: edit(map[string]string{}, 1)}, 0.07 * 1.3},
		{"unsupported quality value falls back to the highest price", quality,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"quality": "ultra"}), nil, 0)}, 0.104},
		{"resolution only: default 1k", resolution,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(map[string]string{}, nil, 0)}, 0.05 * 1.3},
		{"resolution only: 2k", resolution,
			converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"resolution": "2k"}), nil, 0)}, 0.07 * 1.3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, imageCost(t, tt.price, tt.usage), 1e-12)
		})
	}
}

func TestImagePricing_FlatPriceUnchanged(t *testing.T) {
	price := loadImagePrices(t)["flat-model"]
	usage := converter.TokenUsage{ImageCount: 3, ImageBilling: generation(map[string]string{"size": "2k"}, []int64{4194304, 4194304, 4194304}, 5)}
	assert.InDelta(t, 3*0.052, imageCost(t, price, usage), 1e-12)
	assert.InDelta(t, 2*0.052, imageCost(t, price, converter.TokenUsage{ImageCount: 2}), 1e-12)
}

// The whole price map is decoded in one pass: a malformed tier must not break
// other models, and a tier without an explicit price must never make images free.
func TestImagePricing_LenientDecoding(t *testing.T) {
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{
		"broken-tiers": {
			"output_cost_per_image": 0.2,
			"image_request_defaults": {"resolution": "2K", "nested": {"x": 1}, "flag": false, "count": 2},
			"output_cost_per_image_tiers": [
				{"max_pixels": "2610000", "output_cost_per_image": 0.01},
				{"max_pixels": 2610000},
				{"max_pixels": 2610000, "output_cost_per_image": null},
				{"max_pixels": 2610000, "output_cost_per_image": "0.01"},
				{"max_pixels": 2610000, "output_cost_per_image": -1},
				{"max_pixel": 2610000, "output_cost_per_image": 0.01},
				{"operation": "edits", "output_cost_per_image": 0.01},
				{"when": ["resolution"], "output_cost_per_image": 0.03},
				{"when": {"resolution": ["1k", "2k"]}, "output_cost_per_image": 0.04},
				"not-an-object",
				{"operation": " EDIT ", "when": {"Resolution": "2K"}, "output_cost_per_image": 0.05},
				{"max_pixels": 2610000, "output_cost_per_image": 0.1}
			]
		},
		"not-a-list": {"output_cost_per_image": 0.3, "output_cost_per_image_tiers": {"max_pixels": 1}},
		"healthy": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002}
	}`), &prices))

	broken := prices["broken-tiers"]
	require.Len(t, broken.OutputCostPerImageTiers, 2)
	assert.Equal(t, ImageRequestParams{"resolution": `"2k"`, "flag": "false", "count": "2"}, broken.ImageRequestDefaults)
	assert.Equal(t, "edit", broken.OutputCostPerImageTiers[0].Operation)
	assert.Equal(t, ImageRequestParams{"Resolution": `"2k"`}, broken.OutputCostPerImageTiers[0].When)
	assert.Equal(t, int64(2610000), broken.OutputCostPerImageTiers[1].MaxPixels)

	// Only the well-formed tiers take part: small image → 0.1, large → flat 0.2.
	assert.InDelta(t, 0.1, imageCost(t, broken, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{1000}, 0)}), 1e-12)
	assert.InDelta(t, 0.2, imageCost(t, broken, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{4000000}, 0)}), 1e-12)

	assert.Empty(t, prices["not-a-list"].OutputCostPerImageTiers)
	assert.InDelta(t, 0.3, imageCost(t, prices["not-a-list"], converter.TokenUsage{ImageCount: 1}), 1e-12)
	assert.InDelta(t, 0.000002, prices["healthy"].OutputCostPerToken, 1e-15)
}

func TestNormalizeImageParamValue(t *testing.T) {
	for _, tt := range []struct {
		in   any
		want string
		ok   bool
	}{
		{" 2K ", `"2k"`, true},
		{"true", `"true"`, true},
		{true, "true", true},
		{false, "false", true},
		{float64(2), "2", true},
		{float64(0.5), "0.5", true},
		{json.Number("3"), "3", true},
		{"", "", false},
		{string(make([]byte, 65)), "", false},
		{nil, "", false},
		{[]any{"a"}, "", false},
		{map[string]any{"a": 1}, "", false},
	} {
		got, ok := NormalizeImageParamValue(tt.in)
		assert.Equal(t, tt.ok, ok, "%#v", tt.in)
		assert.Equal(t, tt.want, got, "%#v", tt.in)
	}
}

// A typo in a tier key or operation must drop the tier, never widen it into a
// tier that matches every image.
func TestImagePricing_MalformedTierNeverBecomesCatchAll(t *testing.T) {
	for _, tier := range []string{
		`{"max_pixel": 2610000, "output_cost_per_image": 0.0585}`,
		`{"wehn": {"quality": "low"}, "output_cost_per_image": 0.0585}`,
		`{"when": {"quality": ["low"]}, "output_cost_per_image": 0.0585}`,
		`{"operation": "edits", "output_cost_per_image": 0.0585}`,
	} {
		t.Run(tier, func(t *testing.T) {
			var prices map[string]*ModelPrice
			require.NoError(t, json.Unmarshal([]byte(`{"m":{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[`+tier+`]}}`), &prices))
			usage := converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"quality": "low"}), []int64{1024}, 0)}
			assert.InDelta(t, 0.117, imageCost(t, prices["m"], usage), 1e-12)
		})
	}
}

// Conditions compare JSON types: a string "true" is not the boolean true, so a
// mistyped parameter a provider might ignore can't select a cheaper tier.
func TestImagePricing_ConditionsKeepJSONTypes(t *testing.T) {
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{"m":{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[
		{"when": {"layer_decomposition": true}, "output_cost_per_image": 0.0585},
		{"when": {"n": 2}, "output_cost_per_image": 0.05}
	]}}`), &prices))
	price := prices["m"]

	assert.InDelta(t, 0.0585, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"layer_decomposition": true}), nil, 0)}), 1e-12)
	assert.InDelta(t, 0.117, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"layer_decomposition": "true"}), nil, 0)}), 1e-12)
	assert.InDelta(t, 0.05, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"n": float64(2)}), nil, 0)}), 1e-12)
	assert.InDelta(t, 0.117, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"n": "2"}), nil, 0)}), 1e-12)
}

// Parameter names are exact: the provider reads "resolution", not "Resolution".
func TestImagePricing_ParameterNamesAreExact(t *testing.T) {
	price := loadImagePrices(t)["resolution-tier-model"]
	usage := converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"Resolution": "2k"}), nil, 0)}
	assert.InDelta(t, 0.05*1.3, imageCost(t, price, usage), 1e-12)
}

func TestImageRequestParams_RoundTrip(t *testing.T) {
	var price ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{"image_request_defaults":{"resolution":"1K","layer_decomposition":false,"n":2},"output_cost_per_image_tiers":[{"when":{"quality":"Low"},"output_cost_per_image":0.1}]}`), &price))
	encoded, err := json.Marshal(price)
	require.NoError(t, err)
	var decoded ModelPrice
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	assert.Equal(t, price.ImageRequestDefaults, decoded.ImageRequestDefaults)
	assert.Equal(t, price.OutputCostPerImageTiers, decoded.OutputCostPerImageTiers)
	assert.Equal(t, ImageRequestParams{"resolution": `"1k"`, "layer_decomposition": "false", "n": "2"}, decoded.ImageRequestDefaults)
}

func TestNormalizeImageFormValue(t *testing.T) {
	for in, want := range map[string]string{
		"true":  "true",
		"FALSE": "false",
		"2":     "2",
		" 2K ":  `"2k"`,
		"low":   `"low"`,
	} {
		got, ok := NormalizeImageFormValue(in)
		require.True(t, ok, in)
		assert.Equal(t, want, got, in)
	}
	_, ok := NormalizeImageFormValue("   ")
	assert.False(t, ok)
}

// Organization tariffs are loaded strictly: a malformed image pricing field
// must fail the tariff instead of being silently dropped.
func TestDecodeStrictPriceRow_ImagePricing(t *testing.T) {
	valid := `{"output_cost_per_image":0.117,"input_cost_per_image":0.0039,"input_images_free_per_request":1,"image_request_defaults":{"resolution":"1k"},"output_cost_per_image_tiers":[{"operation":"edit","when":{"quality":"auto"},"max_pixels":2610000,"output_cost_per_image":0.0585}]}`
	price, err := decodeStrictPriceRow("image-model", json.RawMessage(valid))
	require.NoError(t, err)
	require.Len(t, price.OutputCostPerImageTiers, 1)

	for name, row := range map[string]string{
		"tier typo":             `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"max_pixel":2610000,"output_cost_per_image":0.0585}]}`,
		"tier without price":    `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"max_pixels":2610000}]}`,
		"unknown operation":     `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"operation":"edits","output_cost_per_image":0.0585}]}`,
		"tiers not an array":    `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":{"output_cost_per_image":0.0585}}`,
		"non-scalar default":    `{"output_cost_per_image":0.117,"image_request_defaults":{"resolution":["1k"]}}`,
		"negative free images":  `{"output_cost_per_image":0.117,"input_images_free_per_request":-1}`,
		"only non-price fields": `{"image_request_defaults":{"resolution":"1k"},"input_images_free_per_request":1}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := decodeStrictPriceRow("image-model", json.RawMessage(row))
			assert.Error(t, err)
		})
	}
}

// The price map is re-decoded on every sync, so a malformed tier is reported
// once rather than on every reload.
func TestImagePricing_MalformedTierWarnsOnce(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })

	body := []byte(`{"m":{"output_cost_per_image":0.1,"output_cost_per_image_tiers":[{"max_pixel":123456789,"output_cost_per_image":0.01}]}}`)
	for range 3 {
		var prices map[string]*ModelPrice
		require.NoError(t, json.Unmarshal(body, &prices))
		assert.Empty(t, prices["m"].OutputCostPerImageTiers)
	}
	assert.Equal(t, 1, strings.Count(logs.String(), "Ignoring malformed image price tier"))
	assert.Contains(t, logs.String(), "max_pixel")
}

// Tiers are matched first-wins, but a tier listed after one that covers all
// of its images could never match. Such a tier is moved ahead of the tier
// covering it, so a natural "base price, then discounts" order still bills
// the discounts.
func TestImagePriceTiers_ShadowedTiersAreMovedAhead(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })

	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{"m":{"output_cost_per_image":0.2,"output_cost_per_image_tiers":[
		{"output_cost_per_image": 0.117},
		{"max_pixels": 2000000, "output_cost_per_image": 0.0585},
		{"when": {"quality": "low"}, "output_cost_per_image": 0.05},
		{"when": {"quality": "low"}, "max_pixels": 1000000, "output_cost_per_image": 0.02},
		{"operation": "edit", "output_cost_per_image": 0.3}
	]}}`), &prices))
	price := prices["m"]

	low := canon(map[string]any{"quality": "low"})
	for _, tt := range []struct {
		name    string
		billing *converter.ImageBillingDetails
		want    float64
	}{
		{"large image", generation(nil, []int64{4000000}, 0), 0.117},
		{"small image", generation(nil, []int64{1500000}, 0), 0.0585},
		{"unknown size", generation(nil, nil, 0), 0.117},
		{"low quality large image", generation(low, []int64{4000000}, 0), 0.05},
		// Both the pixel and the quality tier match; neither covers the other,
		// so the one listed first still wins.
		{"low quality small image", generation(low, []int64{1500000}, 0), 0.0585},
		{"low quality tiny image", generation(low, []int64{800000}, 0), 0.02},
		{"edit", &converter.ImageBillingDetails{Operation: converter.ImageOperationEdit}, 0.3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: tt.billing}), 1e-12)
		})
	}
	assert.Contains(t, logs.String(), "Image price tier moved ahead of a tier that covers it")
}

// Tiers that don't cover one another keep their listed order, including when
// they overlap: the first listed one still wins for images both match.
func TestImagePriceTiers_OverlappingTiersKeepOrder(t *testing.T) {
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{"m":{"output_cost_per_image":0.2,"output_cost_per_image_tiers":[
		{"when": {"quality": "low"}, "output_cost_per_image": 0.05},
		{"max_pixels": 2000000, "output_cost_per_image": 0.0585},
		{"operation": "generation", "output_cost_per_image": 0.1},
		{"when": {"resolution": "1k"}, "output_cost_per_image": 0.07}
	]}}`), &prices))
	price := prices["m"]
	require.Len(t, price.OutputCostPerImageTiers, 4)
	assert.InDelta(t, 0.05, price.OutputCostPerImageTiers[0].OutputCostPerImage, 1e-12)
	assert.InDelta(t, 0.0585, price.OutputCostPerImageTiers[1].OutputCostPerImage, 1e-12)
	assert.InDelta(t, 0.1, price.OutputCostPerImageTiers[2].OutputCostPerImage, 1e-12)
	assert.InDelta(t, 0.07, price.OutputCostPerImageTiers[3].OutputCostPerImage, 1e-12)

	low := canon(map[string]any{"quality": "low"})
	assert.InDelta(t, 0.05, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(low, []int64{1500000}, 0)}), 1e-12)
	assert.InDelta(t, 0.0585, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, []int64{1500000}, 0)}), 1e-12)
}

// A tier with exactly the conditions of an earlier one can never match and
// has no unambiguous place to move to: it is dropped and the first one kept.
func TestImagePriceTiers_DuplicateConditionsKeepFirst(t *testing.T) {
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{"m":{"output_cost_per_image":0.2,"output_cost_per_image_tiers":[
		{"operation": "edit", "when": {"quality": "low", "n": 2}, "max_pixels": 1000, "output_cost_per_image": 0.05},
		{"operation": "EDIT", "when": {"n": 2, "quality": "LOW"}, "max_pixels": 1000, "output_cost_per_image": 0.01}
	]}}`), &prices))
	price := prices["m"]
	require.Len(t, price.OutputCostPerImageTiers, 1)
	assert.InDelta(t, 0.05, price.OutputCostPerImageTiers[0].OutputCostPerImage, 1e-12)
}

// ImageRequestDefaults don't change which tier covers which: a request can
// always set a parameter to a value other than the default.
func TestImagePriceTiers_DefaultsDoNotCreateCoverage(t *testing.T) {
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(`{"m":{"output_cost_per_image":0.2,"image_request_defaults":{"resolution":"1k"},"output_cost_per_image_tiers":[
		{"when": {"resolution": "1k"}, "output_cost_per_image": 0.05},
		{"output_cost_per_image": 0.1}
	]}}`), &prices))
	price := prices["m"]
	require.Len(t, price.OutputCostPerImageTiers, 2)
	assert.InDelta(t, 0.05, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(nil, nil, 0)}), 1e-12)
	assert.InDelta(t, 0.1, imageCost(t, price, converter.TokenUsage{ImageCount: 1, ImageBilling: generation(canon(map[string]any{"resolution": "2k"}), nil, 0)}), 1e-12)
}

// Organization tariffs fail closed instead of being reordered.
func TestDecodeStrictPriceRow_RejectsUnreachableTiers(t *testing.T) {
	for name, row := range map[string]string{
		"catch-all first":      `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"output_cost_per_image":0.117},{"max_pixels":2000000,"output_cost_per_image":0.0585}]}`,
		"wider pixel limit":    `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"max_pixels":4000000,"output_cost_per_image":0.1},{"max_pixels":2000000,"output_cost_per_image":0.0585}]}`,
		"duplicate conditions": `{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"when":{"quality":"low"},"output_cost_per_image":0.1},{"when":{"quality":"Low"},"output_cost_per_image":0.05}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := decodeStrictPriceRow("image-model", json.RawMessage(row))
			require.Error(t, err)
			assert.Contains(t, err.Error(), "output_cost_per_image_tiers[1]")
		})
	}

	_, err := decodeStrictPriceRow("image-model", json.RawMessage(`{"output_cost_per_image":0.117,"output_cost_per_image_tiers":[{"max_pixels":2000000,"output_cost_per_image":0.0585},{"output_cost_per_image":0.117}]}`))
	assert.NoError(t, err)
}
