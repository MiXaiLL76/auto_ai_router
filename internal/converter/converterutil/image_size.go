package converterutil

import (
	"strconv"
	"strings"
)

// ParseImageDimensions parses an image size written as two positive integers
// around one separator: "x", "X" or "×" for pixel dimensions ("1024x768"), or
// ":" or "/" for an aspect ratio ("16:9"), which ratio reports. Spaces around
// either number are allowed. ok=false for anything else ("2K", "auto",
// "1024x768x2").
func ParseImageDimensions(size string) (width, height int, ratio, ok bool) {
	normalized := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(size)), "×", "x")

	separator := "x"
	for _, candidate := range []string{":", "/"} {
		if strings.Contains(normalized, candidate) {
			separator = candidate
			ratio = true
			break
		}
	}

	left, right, found := strings.Cut(normalized, separator)
	if !found {
		return 0, 0, false, false
	}
	width, widthErr := strconv.Atoi(strings.TrimSpace(left))
	height, heightErr := strconv.Atoi(strings.TrimSpace(right))
	if widthErr != nil || heightErr != nil || width <= 0 || height <= 0 {
		return 0, 0, false, false
	}
	return width, height, ratio, true
}
