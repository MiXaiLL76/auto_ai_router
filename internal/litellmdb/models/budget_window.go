package models

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// BudgetLimit is one entry of a key's LiteLLM budget_limits column
// (LiteLLM's BudgetLimitEntry): a spend cap that applies per window of
// Duration instead of over the key's lifetime. Duration is kept as stored: it
// names the window's LiteLLM_BudgetWindowSpend row and counter, as in LiteLLM.
type BudgetLimit struct {
	Duration  string
	MaxBudget float64
}

// ParseBudgetLimits reads LiteLLM_VerificationToken.budget_limits. Empty or
// null input yields nil. Unlike ParseCostMarginConfig it is all-or-nothing:
// a limit that cannot be read must not leave the key unlimited, so the caller
// fails the key closed on error.
//
// Entries with the same budget_duration share one window row and counter, so
// they are merged into the first of them with the smaller max_budget: a spend
// under both limits is a spend under the smaller one.
func ParseBudgetLimits(raw []byte) ([]BudgetLimit, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var entries []struct {
		BudgetDuration string   `json:"budget_duration"`
		MaxBudget      *float64 `json:"max_budget"`
	}
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("parse budget_limits: %w", err)
	}
	limits := make([]BudgetLimit, 0, len(entries))
	index := make(map[string]int, len(entries))
	for i, entry := range entries {
		if entry.BudgetDuration == "" || entry.MaxBudget == nil {
			return nil, fmt.Errorf("parse budget_limits[%d]: budget_duration and max_budget are required", i)
		}
		if j, ok := index[entry.BudgetDuration]; ok {
			limits[j].MaxBudget = min(limits[j].MaxBudget, *entry.MaxBudget)
			continue
		}
		index[entry.BudgetDuration] = len(limits)
		limits = append(limits, BudgetLimit{Duration: entry.BudgetDuration, MaxBudget: *entry.MaxBudget})
	}
	return limits, nil
}

// BudgetCalendar is where key budget windows start: the timezone of the
// calendar (nil means UTC) and the time of day day, week and month windows
// reset at (litellm_db.budget_timezone and budget_reset_time, LiteLLM's
// litellm_settings.timezone and budget_reset_time).
type BudgetCalendar struct {
	Location  *time.Location
	ResetTime time.Duration
}

type budgetUnit int

const (
	budgetUnitMinute budgetUnit = iota + 1
	budgetUnitHour
	budgetUnitDay
	budgetUnitWeek
	budgetUnitMonth
)

// BudgetDuration is a budget_duration whose windows LiteLLM resets on a fixed
// calendar grid, so their bounds follow from the time alone.
type BudgetDuration struct {
	unit  budgetUnit
	value int
}

// budgetDurationAliases are LiteLLM's word spellings
// (duration_parser._BUDGET_DURATION_WORD_ALIASES).
var budgetDurationAliases = map[string]string{
	"hourly":  "1h",
	"daily":   "24h",
	"weekly":  "7d",
	"monthly": "30d",
}

var budgetDurationPattern = regexp.MustCompile(`^(\d+)(mo|[smhdw])$`)

// ParseBudgetDuration reads a budget_duration the way LiteLLM does and
// returns its calendar grid, as LiteLLM's get_next_standardized_reset_time
// lays it: "1d" from the reset time each day, "7d"/"1w" from Monday,
// "30d"/"1mo" from the 1st of the month, "Nh" and "Nm" in steps of N from the
// start of the day or hour ("24h" is a day from midnight). ok is false for a
// duration LiteLLM counts from the moment its reset job last ran ("2d",
// "14d", "2w", "5h", "48h"), for seconds and for anything it cannot parse:
// those windows cannot be placed without that job.
func ParseBudgetDuration(raw string) (BudgetDuration, bool) {
	duration := raw
	if alias, ok := budgetDurationAliases[strings.ToLower(strings.TrimSpace(raw))]; ok {
		duration = alias
	}
	match := budgetDurationPattern.FindStringSubmatch(duration)
	if match == nil {
		return BudgetDuration{}, false
	}
	value, err := strconv.Atoi(match[1])
	if err != nil || value <= 0 {
		return BudgetDuration{}, false
	}
	switch unit := match[2]; {
	case unit == "d" && value == 1:
		return BudgetDuration{unit: budgetUnitDay, value: 1}, true
	case unit == "d" && value == 7, unit == "w" && value == 1:
		return BudgetDuration{unit: budgetUnitWeek, value: 1}, true
	case unit == "d" && value == 30, unit == "mo" && value == 1:
		return BudgetDuration{unit: budgetUnitMonth, value: 1}, true
	case unit == "h" && 24%value == 0:
		return BudgetDuration{unit: budgetUnitHour, value: value}, true
	case unit == "m" && 60%value == 0:
		return BudgetDuration{unit: budgetUnitMinute, value: value}, true
	default:
		return BudgetDuration{}, false
	}
}

// Bounds returns the window of d that contains at, in UTC: start <= at < end.
//
// The windows are laid on the wall clock of the calendar, but chosen by
// instant: around a daylight saving change a wall-clock time can occur twice
// or not at all, so the window is the pair of grid points whose instants
// bracket at, not the one at's wall clock falls into.
func (d BudgetDuration) Bounds(at time.Time, calendar BudgetCalendar) (start, end time.Time) {
	loc := calendar.Location
	if loc == nil {
		loc = time.UTC
	}
	grid := d.grid(at.In(loc), calendar.ResetTime)
	point := func(step int) time.Time { return wallInstant(grid(step), loc) }
	step := 0
	for at.Before(point(step)) {
		step--
	}
	for !at.Before(point(step + 1)) {
		step++
	}
	return point(step).UTC(), point(step + 1).UTC()
}

// grid returns the grid points of d as wall-clock times (held in UTC), in
// steps from the window local's wall clock falls into (step 0); time.Date
// normalizes the fields, so a step is calendar arithmetic: a month step lands
// on the 1st, a day step on the reset time.
func (d BudgetDuration) grid(local time.Time, resetTime time.Duration) func(step int) time.Time {
	year, month, day := local.Date()
	switch d.unit {
	case budgetUnitMinute:
		minute := local.Minute() - local.Minute()%d.value
		return func(step int) time.Time {
			return time.Date(year, month, day, local.Hour(), minute+step*d.value, 0, 0, time.UTC)
		}
	case budgetUnitHour:
		hour := local.Hour() - local.Hour()%d.value
		return func(step int) time.Time {
			return time.Date(year, month, day, hour+step*d.value, 0, 0, 0, time.UTC)
		}
	}
	// The reset time of day moves the day boundary: before it, the local date
	// still belongs to the previous day, week or month.
	if wallClock(local) < resetTime {
		day--
	}
	reset := func(year int, month time.Month, day int) time.Time {
		return time.Date(year, month, day, 0, 0, 0, 0, time.UTC).Add(resetTime)
	}
	switch d.unit {
	case budgetUnitWeek:
		monday := day - (int(time.Date(year, month, day, 0, 0, 0, 0, time.UTC).Weekday())+6)%7
		return func(step int) time.Time { return reset(year, month, monday+7*step) }
	case budgetUnitMonth:
		year, month, _ = time.Date(year, month, day, 0, 0, 0, 0, time.UTC).Date()
		return func(step int) time.Time { return reset(year, month+time.Month(step), 1) }
	default:
		return func(step int) time.Time { return reset(year, month, day+step) }
	}
}

// wallInstant is the instant loc's clock first shows the wall-clock time
// wall (held in UTC). A time the clock skips maps to the change itself, where
// the clock passes it, so the grid points stay in order; time.Date would move
// it by the size of the change instead. A time the clock shows twice is
// placed where time.Date places it.
func wallInstant(wall time.Time, loc *time.Location) time.Time {
	t := time.Date(wall.Year(), wall.Month(), wall.Day(), wall.Hour(), wall.Minute(), wall.Second(), 0, loc)
	shown := time.Date(t.Year(), t.Month(), t.Day(), t.Hour(), t.Minute(), t.Second(), 0, time.UTC)
	if shown.Equal(wall) {
		return t
	}
	zoneStart, zoneEnd := t.ZoneBounds()
	if shown.After(wall) {
		return zoneStart // moved forward past the change
	}
	return zoneEnd // moved back before the change
}

// wallClock is the local time of day of t.
func wallClock(t time.Time) time.Duration {
	return time.Duration(t.Hour())*time.Hour + time.Duration(t.Minute())*time.Minute +
		time.Duration(t.Second())*time.Second + time.Duration(t.Nanosecond())
}

// BudgetWindow is one key budget window as of the moment the token was
// loaded: its limit, the window that was current then, and the key's spend
// within it. Start is zero for a limit AIR cannot enforce (a duration
// ParseBudgetDuration rejects or an unreadable budget_limits); Validate
// refuses such a key.
type BudgetWindow struct {
	BudgetLimit
	Start time.Time // UTC
	Spend float64
}

// checkBudgetWindows checks the key's budget windows (embedded, use >=), in
// the order budget_limits lists them.
func (t *TokenInfo) checkBudgetWindows() error {
	for _, window := range t.BudgetWindows {
		if window.Start.IsZero() {
			return ErrBudgetWindowUnsupported
		}
		if window.Spend >= window.MaxBudget {
			return &BudgetExceededError{Level: BudgetLevelKey, Window: window.Duration}
		}
	}
	return nil
}

// BudgetWindowsStale reports whether a window loaded with the token has ended
// by at: its spend belongs to a window that is no longer current, so a cached
// token must be reloaded rather than checked against it.
func (t *TokenInfo) BudgetWindowsStale(at time.Time, calendar BudgetCalendar) bool {
	for _, window := range t.BudgetWindows {
		if window.Start.IsZero() {
			continue
		}
		duration, ok := ParseBudgetDuration(window.Duration)
		if !ok {
			return true
		}
		if start, _ := duration.Bounds(at, calendar); !start.Equal(window.Start) {
			return true
		}
	}
	return false
}
