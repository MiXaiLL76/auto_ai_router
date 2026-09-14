package converterutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseImageDimensions(t *testing.T) {
	type result struct {
		width, height int
		ratio, ok     bool
	}
	for size, want := range map[string]result{
		"1024x768":               {1024, 768, false, true},
		" 1424X800 ":             {1424, 800, false, true},
		"1024×1536":              {1024, 1536, false, true},
		"1024 x 1024":            {1024, 1024, false, true},
		"16:9":                   {16, 9, true, true},
		"3/2":                    {3, 2, true, true},
		"":                       {},
		"auto":                   {},
		"2K":                     {},
		"1024":                   {},
		"x768":                   {},
		"1024x":                  {},
		"0x100":                  {},
		"-5x100":                 {},
		"1024x768x2":             {},
		"16:9:1":                 {},
		"16:9x2":                 {},
		"1e3x100":                {},
		"99999999999999999999x1": {},
	} {
		width, height, ratio, ok := ParseImageDimensions(size)
		assert.Equal(t, want, result{width, height, ratio, ok}, size)
	}
}
