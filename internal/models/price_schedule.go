package models

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"strings"
	"time"
)

const secondsPerDay = 24 * 60 * 60

// PricingWindow is one time-of-day tariff of a model's pricing_schedule.
// The interval is [start_utc, end_utc) in UTC: an end earlier than the start
// wraps around midnight, start == end covers the whole day. Price fields the
// window does not declare are inherited from the model's base price.
type PricingWindow struct {
	Name     string `json:"name,omitempty"`
	StartUTC string `json:"start_utc"`
	EndUTC   string `json:"end_utc"`

	// Raw window object, replayed over the base price by compileSchedule so
	// that only the keys it actually declares are overridden.
	overrides json.RawMessage
}

// modelPriceFields is ModelPrice without its methods: decoding into it leaves
// every field the JSON does not mention untouched.
type modelPriceFields ModelPrice

// UnmarshalJSON decodes the window's own fields and keeps the raw object.
func (w *PricingWindow) UnmarshalJSON(data []byte) error {
	type pricingWindowFields PricingWindow
	var fields pricingWindowFields
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	*w = PricingWindow(fields)
	w.overrides = append(json.RawMessage(nil), data...)
	return nil
}

// MarshalJSON re-emits the original object so per-window rates survive a
// JSON round-trip.
func (w PricingWindow) MarshalJSON() ([]byte, error) {
	if len(w.overrides) > 0 {
		return w.overrides, nil
	}
	type pricingWindowFields PricingWindow
	return json.Marshal(pricingWindowFields(w))
}

// compiledPricingWindow is a window resolved at load time: its bounds in
// seconds since UTC midnight, and the price row to bill while it is active.
type compiledPricingWindow struct {
	name     string
	startSec int
	endSec   int
	price    *ModelPrice
}

// contains reports whether secondOfDay falls inside [startSec, endSec),
// wrapping around midnight when the end is earlier than the start.
func (w compiledPricingWindow) contains(secondOfDay int) bool {
	switch {
	case w.startSec == w.endSec:
		return true
	case w.startSec < w.endSec:
		return secondOfDay >= w.startSec && secondOfDay < w.endSec
	default:
		return secondOfDay >= w.startSec || secondOfDay < w.endSec
	}
}

// segments returns the linear [from, to) intervals of the day the window
// covers — two of them when it wraps around midnight.
func (w compiledPricingWindow) segments() [][2]int {
	switch {
	case w.startSec == w.endSec:
		return [][2]int{{0, secondsPerDay}}
	case w.startSec < w.endSec:
		return [][2]int{{w.startSec, w.endSec}}
	default:
		return [][2]int{{w.startSec, secondsPerDay}, {0, w.endSec}}
	}
}

// compileSchedule validates pricing_schedule and precomputes every window's
// price. Each window is a snapshot of the model's fields, so it must run after
// the caller has finished filling those in (see LoadModelPrices). On error the
// entry must be dropped rather than billed at base rates for half the day.
func (p *ModelPrice) compileSchedule() error {
	if p == nil {
		return nil
	}
	if len(p.PricingSchedule) == 0 {
		p.compiledSchedule = nil
		return nil
	}

	// Base row the windows inherit from; the schedule is cleared so a window
	// price can never recurse into another lookup.
	base := *p
	base.PricingSchedule = nil
	base.compiledSchedule = nil
	base.windowName = ""

	compiled := make([]compiledPricingWindow, 0, len(p.PricingSchedule))
	for i, window := range p.PricingSchedule {
		if window == nil {
			return fmt.Errorf("pricing_schedule[%d]: window is null", i)
		}
		startSec, err := parseWindowTime(window.StartUTC)
		if err != nil {
			return fmt.Errorf("pricing_schedule[%d] (%s): start_utc: %w", i, window.Name, err)
		}
		endSec, err := parseWindowTime(window.EndUTC)
		if err != nil {
			return fmt.Errorf("pricing_schedule[%d] (%s): end_utc: %w", i, window.Name, err)
		}

		effective := base
		// json.Unmarshal reuses a non-nil map, so clone it to keep a window's
		// web search prices out of the base row.
		effective.SearchContextCostPerQuery = maps.Clone(base.SearchContextCostPerQuery)
		if len(window.overrides) > 0 {
			if err := json.Unmarshal(window.overrides, (*modelPriceFields)(&effective)); err != nil {
				return fmt.Errorf("pricing_schedule[%d] (%s): %w", i, window.Name, err)
			}
		}
		effective.PricingSchedule = nil
		effective.compiledSchedule = nil
		effective.windowName = window.Name

		compiled = append(compiled, compiledPricingWindow{
			name:     window.Name,
			startSec: startSec,
			endSec:   endSec,
			price:    &effective,
		})
	}

	warnOverlappingWindows(compiled)
	p.compiledSchedule = compiled
	return nil
}

// warnOverlappingWindows reports windows covering the same second of the day.
// Not fatal — EffectiveAt takes the first match — but almost always a typo.
func warnOverlappingWindows(windows []compiledPricingWindow) {
	for i := range windows {
		for j := i + 1; j < len(windows); j++ {
			if !windowsOverlap(windows[i], windows[j]) {
				continue
			}
			slog.Warn("pricing_schedule windows overlap: the earlier window wins",
				"window", windows[i].name,
				"overlapping_window", windows[j].name,
			)
		}
	}
}

func windowsOverlap(a, b compiledPricingWindow) bool {
	for _, left := range a.segments() {
		for _, right := range b.segments() {
			if left[0] < right[1] && right[0] < left[1] {
				return true
			}
		}
	}
	return false
}

// EffectiveAt returns the price row to bill a request that started at the
// given instant — its start, never the time the response finished. Models
// without a pricing_schedule return themselves unchanged.
func (p *ModelPrice) EffectiveAt(at time.Time) *ModelPrice {
	if p == nil || len(p.compiledSchedule) == 0 {
		return p
	}
	secondOfDay := secondsSinceUTCMidnight(at)
	for _, window := range p.compiledSchedule {
		if window.contains(secondOfDay) {
			return window.price
		}
	}
	// Hours no window covers keep the model's base rates.
	return p
}

// PricingWindowName returns the window a price row came from, or "" for a
// base row.
func (p *ModelPrice) PricingWindowName() string {
	if p == nil {
		return ""
	}
	return p.windowName
}

// secondsSinceUTCMidnight drops sub-second precision, so 14:00:00.5 belongs to
// the window starting at 14:00:00.
func secondsSinceUTCMidnight(at time.Time) int {
	utc := at.UTC()
	return utc.Hour()*3600 + utc.Minute()*60 + utc.Second()
}

// parseWindowTime parses "HH:MM" or "HH:MM:SS" UTC into seconds since
// midnight. "24:00" is accepted for end-of-day and normalised to 0, which the
// matcher already reads as midnight.
func parseWindowTime(value string) (int, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "24:00" || trimmed == "24:00:00" {
		return 0, nil
	}
	layout := "15:04"
	if strings.Count(trimmed, ":") == 2 {
		layout = "15:04:05"
	}
	parsed, err := time.Parse(layout, trimmed)
	if err != nil {
		return 0, fmt.Errorf("invalid time %q: want HH:MM or HH:MM:SS in UTC: %w", value, err)
	}
	return parsed.Hour()*3600 + parsed.Minute()*60 + parsed.Second(), nil
}
