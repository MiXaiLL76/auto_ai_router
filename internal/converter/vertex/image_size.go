package vertex

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

type geminiImageRatio struct {
	value               string
	ratioWidth          int
	ratioHeight         int
	resolution1KWidth   int
	resolution1KHeight  int
	resolutionOverrides map[string]geminiImageDimensions
}

type geminiImageDimensions struct {
	width  int
	height int
}

type geminiImageResolution struct {
	value string
	scale float64
}

type geminiImageProfile struct {
	ratios      []geminiImageRatio
	resolutions []geminiImageResolution
}

type geminiImageConfig struct {
	aspectRatio string
	imageSize   string
}

var gemini25ImageRatios = []geminiImageRatio{
	{value: "1:1", ratioWidth: 1, ratioHeight: 1, resolution1KWidth: 1024, resolution1KHeight: 1024},
	{value: "2:3", ratioWidth: 2, ratioHeight: 3, resolution1KWidth: 832, resolution1KHeight: 1248},
	{value: "3:2", ratioWidth: 3, ratioHeight: 2, resolution1KWidth: 1248, resolution1KHeight: 832},
	{value: "3:4", ratioWidth: 3, ratioHeight: 4, resolution1KWidth: 864, resolution1KHeight: 1184},
	{value: "4:3", ratioWidth: 4, ratioHeight: 3, resolution1KWidth: 1184, resolution1KHeight: 864},
	{value: "4:5", ratioWidth: 4, ratioHeight: 5, resolution1KWidth: 896, resolution1KHeight: 1152},
	{value: "5:4", ratioWidth: 5, ratioHeight: 4, resolution1KWidth: 1152, resolution1KHeight: 896},
	{value: "9:16", ratioWidth: 9, ratioHeight: 16, resolution1KWidth: 768, resolution1KHeight: 1344},
	{value: "16:9", ratioWidth: 16, ratioHeight: 9, resolution1KWidth: 1344, resolution1KHeight: 768},
	{value: "21:9", ratioWidth: 21, ratioHeight: 9, resolution1KWidth: 1536, resolution1KHeight: 672},
}

var gemini3ImageRatios = []geminiImageRatio{
	{value: "1:1", ratioWidth: 1, ratioHeight: 1, resolution1KWidth: 1024, resolution1KHeight: 1024},
	{value: "2:3", ratioWidth: 2, ratioHeight: 3, resolution1KWidth: 848, resolution1KHeight: 1264},
	{value: "3:2", ratioWidth: 3, ratioHeight: 2, resolution1KWidth: 1264, resolution1KHeight: 848},
	{value: "3:4", ratioWidth: 3, ratioHeight: 4, resolution1KWidth: 896, resolution1KHeight: 1200},
	{value: "4:3", ratioWidth: 4, ratioHeight: 3, resolution1KWidth: 1200, resolution1KHeight: 896},
	{value: "4:5", ratioWidth: 4, ratioHeight: 5, resolution1KWidth: 928, resolution1KHeight: 1152},
	{value: "5:4", ratioWidth: 5, ratioHeight: 4, resolution1KWidth: 1152, resolution1KHeight: 928},
	{value: "9:16", ratioWidth: 9, ratioHeight: 16, resolution1KWidth: 768, resolution1KHeight: 1376},
	{value: "16:9", ratioWidth: 16, ratioHeight: 9, resolution1KWidth: 1376, resolution1KHeight: 768},
	{value: "21:9", ratioWidth: 21, ratioHeight: 9, resolution1KWidth: 1584, resolution1KHeight: 672},
}

var gemini31FlashImageRatios = []geminiImageRatio{
	{value: "1:1", ratioWidth: 1, ratioHeight: 1, resolution1KWidth: 1024, resolution1KHeight: 1024},
	{value: "1:4", ratioWidth: 1, ratioHeight: 4, resolution1KWidth: 512, resolution1KHeight: 2048},
	{value: "1:8", ratioWidth: 1, ratioHeight: 8, resolution1KWidth: 384, resolution1KHeight: 3072},
	{value: "2:3", ratioWidth: 2, ratioHeight: 3, resolution1KWidth: 848, resolution1KHeight: 1264},
	{value: "3:2", ratioWidth: 3, ratioHeight: 2, resolution1KWidth: 1264, resolution1KHeight: 848},
	{value: "3:4", ratioWidth: 3, ratioHeight: 4, resolution1KWidth: 896, resolution1KHeight: 1200},
	{value: "4:1", ratioWidth: 4, ratioHeight: 1, resolution1KWidth: 2048, resolution1KHeight: 512},
	{value: "4:3", ratioWidth: 4, ratioHeight: 3, resolution1KWidth: 1200, resolution1KHeight: 896},
	{value: "4:5", ratioWidth: 4, ratioHeight: 5, resolution1KWidth: 928, resolution1KHeight: 1152},
	{value: "5:4", ratioWidth: 5, ratioHeight: 4, resolution1KWidth: 1152, resolution1KHeight: 928},
	{value: "8:1", ratioWidth: 8, ratioHeight: 1, resolution1KWidth: 3072, resolution1KHeight: 384},
	{value: "9:16", ratioWidth: 9, ratioHeight: 16, resolution1KWidth: 768, resolution1KHeight: 1376},
	{value: "16:9", ratioWidth: 16, ratioHeight: 9, resolution1KWidth: 1376, resolution1KHeight: 768},
	{
		value:              "21:9",
		ratioWidth:         21,
		ratioHeight:        9,
		resolution1KWidth:  1584,
		resolution1KHeight: 672,
		resolutionOverrides: map[string]geminiImageDimensions{
			"512": {width: 792, height: 168},
		},
	},
}

var (
	fixed1KImageResolutions = []geminiImageResolution{{scale: 1}}
	oneKImageResolutions    = []geminiImageResolution{{value: "1K", scale: 1}}
	gemini3ImageResolutions = []geminiImageResolution{
		{value: "1K", scale: 1},
		{value: "2K", scale: 2},
		{value: "4K", scale: 4},
	}
	gemini31FlashImageResolutions = []geminiImageResolution{
		{value: "512", scale: 0.5},
		{value: "1K", scale: 1},
		{value: "2K", scale: 2},
		{value: "4K", scale: 4},
	}
)

func geminiImageProfileForModel(model string) geminiImageProfile {
	lower := strings.ToLower(model)
	switch {
	case strings.Contains(lower, "gemini-3.1-flash-lite-image"):
		return geminiImageProfile{ratios: gemini3ImageRatios, resolutions: oneKImageResolutions}
	case strings.Contains(lower, "gemini-3.1-flash-image"):
		return geminiImageProfile{ratios: gemini31FlashImageRatios, resolutions: gemini31FlashImageResolutions}
	case strings.Contains(lower, "gemini-3-pro-image"):
		return geminiImageProfile{ratios: gemini3ImageRatios, resolutions: gemini3ImageResolutions}
	default:
		return geminiImageProfile{ratios: gemini25ImageRatios, resolutions: fixed1KImageResolutions}
	}
}

func mapGeminiImageSize(model, size string) (*geminiImageConfig, error) {
	size = strings.TrimSpace(size)
	if size == "" || strings.EqualFold(size, "auto") {
		return nil, nil
	}

	width, height, ratioOnly, err := parseGeminiImageSize(size)
	if err != nil {
		return nil, err
	}

	profile := geminiImageProfileForModel(model)
	if !ratioOnly {
		if config, ok := exactGeminiImageConfig(profile, width, height); ok {
			return config, nil
		}
	}

	ratio := nearestGeminiImageRatio(profile.ratios, width, height)
	resolution := defaultGeminiImageResolution(profile.resolutions)
	if !ratioOnly {
		resolution = nearestGeminiImageResolution(profile.resolutions, ratio, width, height)
	}

	return &geminiImageConfig{aspectRatio: ratio.value, imageSize: resolution.value}, nil
}

func exactGeminiImageConfig(profile geminiImageProfile, width, height int) (*geminiImageConfig, bool) {
	for _, ratio := range profile.ratios {
		for _, resolution := range profile.resolutions {
			dimensions := geminiImageDimensions{
				width:  int(math.Round(float64(ratio.resolution1KWidth) * resolution.scale)),
				height: int(math.Round(float64(ratio.resolution1KHeight) * resolution.scale)),
			}
			if override, ok := ratio.resolutionOverrides[resolution.value]; ok {
				dimensions = override
			}
			if dimensions.width == width && dimensions.height == height {
				return &geminiImageConfig{
					aspectRatio: ratio.value,
					imageSize:   resolution.value,
				}, true
			}
		}
	}

	return nil, false
}

func parseGeminiImageSize(size string) (int, int, bool, error) {
	normalized := strings.ToLower(strings.TrimSpace(size))
	normalized = strings.ReplaceAll(normalized, "×", "x")

	separator := "x"
	ratioOnly := false
	for _, candidate := range []string{":", "/"} {
		if strings.Contains(normalized, candidate) {
			separator = candidate
			ratioOnly = true
			break
		}
	}

	left, right, ok := strings.Cut(normalized, separator)
	if !ok || strings.Contains(right, separator) {
		return 0, 0, false, fmt.Errorf("invalid image size %q: expected WxH or W:H", size)
	}
	width, widthErr := strconv.Atoi(strings.TrimSpace(left))
	height, heightErr := strconv.Atoi(strings.TrimSpace(right))
	if widthErr != nil || heightErr != nil || width <= 0 || height <= 0 {
		return 0, 0, false, fmt.Errorf("invalid image size %q: dimensions must be positive integers", size)
	}
	if separator == "x" && width <= 32 && height <= 32 {
		ratioOnly = true
	}

	return width, height, ratioOnly, nil
}

func nearestGeminiImageRatio(ratios []geminiImageRatio, width, height int) geminiImageRatio {
	requested := float64(width) / float64(height)
	nearest := ratios[0]
	nearestDistance := math.Inf(1)
	for _, ratio := range ratios {
		candidate := float64(ratio.ratioWidth) / float64(ratio.ratioHeight)
		distance := math.Abs(math.Log(requested / candidate))
		if distance < nearestDistance {
			nearest = ratio
			nearestDistance = distance
		}
	}
	return nearest
}

func defaultGeminiImageResolution(resolutions []geminiImageResolution) geminiImageResolution {
	for _, resolution := range resolutions {
		if resolution.scale == 1 {
			return resolution
		}
	}
	return resolutions[0]
}

func nearestGeminiImageResolution(
	resolutions []geminiImageResolution,
	ratio geminiImageRatio,
	width, height int,
) geminiImageResolution {
	nearest := resolutions[0]
	nearestDistance := math.Inf(1)
	for _, resolution := range resolutions {
		candidateWidth := float64(ratio.resolution1KWidth) * resolution.scale
		candidateHeight := float64(ratio.resolution1KHeight) * resolution.scale
		widthDistance := math.Log(float64(width) / candidateWidth)
		heightDistance := math.Log(float64(height) / candidateHeight)
		distance := widthDistance*widthDistance + heightDistance*heightDistance
		if distance < nearestDistance {
			nearest = resolution
			nearestDistance = distance
		}
	}
	return nearest
}

func applyGeminiImageSize(genConfig map[string]interface{}, model, size string) error {
	config, err := mapGeminiImageSize(model, size)
	if err != nil {
		return err
	}
	if config == nil {
		return nil
	}

	imageConfig := map[string]interface{}{"aspectRatio": config.aspectRatio}
	if config.imageSize != "" {
		imageConfig["imageSize"] = config.imageSize
	}
	genConfig["image_config"] = imageConfig
	return nil
}
