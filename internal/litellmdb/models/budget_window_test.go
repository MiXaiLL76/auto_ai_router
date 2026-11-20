package models

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func utcTime(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339, value)
	require.NoError(t, err)
	return parsed
}

func TestBudgetDurationBounds_MoscowCalendar(t *testing.T) {
	moscow, err := time.LoadLocation("Europe/Moscow")
	require.NoError(t, err)
	calendar := BudgetCalendar{Location: moscow}
	tests := []struct {
		name      string
		duration  string
		at        string
		wantStart string
		wantEnd   string
	}{
		{"day: last second of a Moscow day", "1d", "2026-03-10T20:59:59Z", "2026-03-09T21:00:00Z", "2026-03-10T21:00:00Z"},
		{"day: Moscow midnight starts the next", "1d", "2026-03-10T21:00:00Z", "2026-03-10T21:00:00Z", "2026-03-11T21:00:00Z"},
		{"day: 24h", "24h", "2026-03-10T21:00:00Z", "2026-03-10T21:00:00Z", "2026-03-11T21:00:00Z"},
		{"day: daily is 24h", "daily", "2026-03-10T20:59:59Z", "2026-03-09T21:00:00Z", "2026-03-10T21:00:00Z"},
		{"week: Sunday 23:59:59 Moscow", "7d", "2026-03-15T20:59:59Z", "2026-03-08T21:00:00Z", "2026-03-15T21:00:00Z"},
		{"week: Monday 00:00 Moscow", "7d", "2026-03-15T21:00:00Z", "2026-03-15T21:00:00Z", "2026-03-22T21:00:00Z"},
		{"week: 1w is 7d", "1w", "2026-03-18T12:00:00Z", "2026-03-15T21:00:00Z", "2026-03-22T21:00:00Z"},
		{"week: weekly is 7d", " Weekly ", "2026-03-18T12:00:00Z", "2026-03-15T21:00:00Z", "2026-03-22T21:00:00Z"},
		{"month: 31st", "1mo", "2026-01-31T20:59:59Z", "2025-12-31T21:00:00Z", "2026-01-31T21:00:00Z"},
		{"month: 1st", "1mo", "2026-01-31T21:00:00Z", "2026-01-31T21:00:00Z", "2026-02-28T21:00:00Z"},
		{"month: 28 February", "1mo", "2026-02-28T20:59:59Z", "2026-01-31T21:00:00Z", "2026-02-28T21:00:00Z"},
		{"month: 29 February of a leap year", "1mo", "2028-02-29T20:59:59Z", "2028-01-31T21:00:00Z", "2028-02-29T21:00:00Z"},
		{"month: 1 March of a leap year", "1mo", "2028-02-29T21:00:00Z", "2028-02-29T21:00:00Z", "2028-03-31T21:00:00Z"},
		{"month: 31 December", "30d", "2026-12-31T20:59:59Z", "2026-11-30T21:00:00Z", "2026-12-31T21:00:00Z"},
		{"month: 1 January", "30d", "2026-12-31T21:00:00Z", "2026-12-31T21:00:00Z", "2027-01-31T21:00:00Z"},
		{"month: monthly is 30d", "monthly", "2026-03-10T12:00:00Z", "2026-02-28T21:00:00Z", "2026-03-31T21:00:00Z"},
		{"hour: hourly", "hourly", "2026-03-10T09:59:59Z", "2026-03-10T09:00:00Z", "2026-03-10T10:00:00Z"},
		{"hour: 6h steps from Moscow midnight", "6h", "2026-03-10T08:30:00Z", "2026-03-10T03:00:00Z", "2026-03-10T09:00:00Z"},
		{"hour: the last 6h step ends at midnight", "6h", "2026-03-10T20:30:00Z", "2026-03-10T15:00:00Z", "2026-03-10T21:00:00Z"},
		{"minute: 15m steps from the hour", "15m", "2026-03-10T09:44:59Z", "2026-03-10T09:30:00Z", "2026-03-10T09:45:00Z"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			duration, ok := ParseBudgetDuration(tt.duration)
			require.True(t, ok)
			start, end := duration.Bounds(utcTime(t, tt.at), calendar)
			assert.Equal(t, utcTime(t, tt.wantStart), start)
			assert.Equal(t, utcTime(t, tt.wantEnd), end)
			assert.Equal(t, time.UTC, start.Location())
		})
	}
}

// LiteLLM's budget_reset_time moves the day, week and month boundaries to a
// time of day; hour windows and daily (24h) keep theirs.
func TestBudgetDurationBounds_ResetTime(t *testing.T) {
	calendar := BudgetCalendar{ResetTime: 12 * time.Hour}
	tests := []struct {
		name      string
		duration  string
		at        string
		wantStart string
		wantEnd   string
	}{
		{"day before the reset time", "1d", "2026-03-10T11:59:59Z", "2026-03-09T12:00:00Z", "2026-03-10T12:00:00Z"},
		{"day at the reset time", "1d", "2026-03-10T12:00:00Z", "2026-03-10T12:00:00Z", "2026-03-11T12:00:00Z"},
		{"Monday morning is still the previous week", "7d", "2026-03-16T08:00:00Z", "2026-03-09T12:00:00Z", "2026-03-16T12:00:00Z"},
		{"the 1st before noon is still the previous month", "1mo", "2026-04-01T08:00:00Z", "2026-03-01T12:00:00Z", "2026-04-01T12:00:00Z"},
		{"daily is 24h: from midnight", "daily", "2026-03-10T08:00:00Z", "2026-03-10T00:00:00Z", "2026-03-11T00:00:00Z"},
		{"hours ignore it", "1h", "2026-03-10T08:30:00Z", "2026-03-10T08:00:00Z", "2026-03-10T09:00:00Z"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			duration, ok := ParseBudgetDuration(tt.duration)
			require.True(t, ok)
			start, end := duration.Bounds(utcTime(t, tt.at), calendar)
			assert.Equal(t, utcTime(t, tt.wantStart), start)
			assert.Equal(t, utcTime(t, tt.wantEnd), end)
		})
	}
}

// Around a daylight saving change a wall-clock time occurs twice or not at
// all: the window is still the one whose bounds bracket the instant, and a
// window stretches or shrinks with the clock as a day does. Berlin repeats
// 02:00-03:00 on 2026-10-25 (00:00Z-01:00Z in summer time, 01:00Z-02:00Z in
// winter time; time.Date places a repeated wall-clock time in winter time, so
// the summer pass joins the window before it) and skips it on 2026-03-29
// (a skipped grid time is passed at the change, 01:00Z).
func TestBudgetDurationBounds_DaylightSaving(t *testing.T) {
	berlin, err := time.LoadLocation("Europe/Berlin")
	require.NoError(t, err)
	tests := []struct {
		name      string
		duration  string
		resetTime time.Duration
		at        string
		wantStart string
		wantEnd   string
	}{
		{"minute: 02:15 in summer time", "1m", 0, "2026-10-25T00:15:30Z", "2026-10-24T23:59:00Z", "2026-10-25T01:00:00Z"},
		{"minute: 02:15 in winter time", "1m", 0, "2026-10-25T01:15:30Z", "2026-10-25T01:15:00Z", "2026-10-25T01:16:00Z"},
		{"15m: 02:15 in summer time", "15m", 0, "2026-10-25T00:20:00Z", "2026-10-24T23:45:00Z", "2026-10-25T01:00:00Z"},
		{"15m: 02:15 in winter time", "15m", 0, "2026-10-25T01:20:00Z", "2026-10-25T01:15:00Z", "2026-10-25T01:30:00Z"},
		{"hour: 02:00 in summer time", "1h", 0, "2026-10-25T00:30:00Z", "2026-10-24T23:00:00Z", "2026-10-25T01:00:00Z"},
		{"hour: 02:00 in winter time", "1h", 0, "2026-10-25T01:30:00Z", "2026-10-25T01:00:00Z", "2026-10-25T02:00:00Z"},
		{"6h: a 7-hour window", "6h", 0, "2026-10-25T01:30:00Z", "2026-10-24T22:00:00Z", "2026-10-25T05:00:00Z"},
		{"day: a 25-hour day", "1d", 0, "2026-10-25T01:30:00Z", "2026-10-24T22:00:00Z", "2026-10-25T23:00:00Z"},
		{"day: 02:45 in summer time, before the reset in winter time", "1d", 150 * time.Minute, "2026-10-25T00:45:00Z", "2026-10-24T00:30:00Z", "2026-10-25T01:30:00Z"},
		{"hour: before the skipped hour", "1h", 0, "2026-03-29T00:59:59Z", "2026-03-29T00:00:00Z", "2026-03-29T01:00:00Z"},
		{"hour: after the skipped hour", "1h", 0, "2026-03-29T01:00:00Z", "2026-03-29T01:00:00Z", "2026-03-29T02:00:00Z"},
		{"day: a skipped reset time falls at the change", "1d", 150 * time.Minute, "2026-03-29T01:10:00Z", "2026-03-29T01:00:00Z", "2026-03-30T00:30:00Z"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			duration, ok := ParseBudgetDuration(tt.duration)
			require.True(t, ok)
			start, end := duration.Bounds(utcTime(t, tt.at), BudgetCalendar{Location: berlin, ResetTime: tt.resetTime})
			assert.Equal(t, utcTime(t, tt.wantStart), start)
			assert.Equal(t, utcTime(t, tt.wantEnd), end)
		})
	}
}

// Every instant falls into exactly one window, and consecutive windows meet:
// checked minute by minute across daylight saving changes, including zones
// that change at midnight, by half an hour or have a quarter-hour offset.
func TestBudgetDurationBounds_WindowsTileTime(t *testing.T) {
	zones := []string{"UTC", "Europe/Berlin", "America/New_York", "America/Santiago", "Australia/Lord_Howe", "America/St_Johns", "Asia/Kathmandu"}
	durations := []string{"1m", "15m", "20m", "1h", "6h", "24h", "1d", "7d", "1mo"}
	for _, zone := range zones {
		loc, err := time.LoadLocation(zone)
		require.NoError(t, err)
		for _, resetTime := range []time.Duration{0, 150 * time.Minute} {
			calendar := BudgetCalendar{Location: loc, ResetTime: resetTime}
			for _, raw := range durations {
				duration, ok := ParseBudgetDuration(raw)
				require.True(t, ok)
				for _, from := range []time.Time{
					time.Date(2026, 3, 6, 0, 0, 0, 0, time.UTC),
					time.Date(2026, 3, 27, 0, 0, 0, 0, time.UTC),
					time.Date(2026, 4, 3, 0, 0, 0, 0, time.UTC),
					time.Date(2026, 9, 4, 0, 0, 0, 0, time.UTC),
					time.Date(2026, 10, 2, 0, 0, 0, 0, time.UTC),
					time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC),
					time.Date(2026, 10, 30, 0, 0, 0, 0, time.UTC),
				} {
					var prevStart, prevEnd time.Time
					for at := from; at.Before(from.Add(4 * 24 * time.Hour)); at = at.Add(time.Minute) {
						start, end := duration.Bounds(at, calendar)
						if at.Before(start) || !at.Before(end) {
							t.Fatalf("%s %s reset %s: %s outside [%s, %s)", zone, raw, resetTime, at, start, end)
						}
						if !prevStart.IsZero() && !start.Equal(prevStart) && !start.Equal(prevEnd) {
							t.Fatalf("%s %s reset %s: window [%s, %s) does not follow [%s, %s)", zone, raw, resetTime, start, end, prevStart, prevEnd)
						}
						prevStart, prevEnd = start, end
					}
				}
			}
		}
	}
}

// Durations LiteLLM counts from the moment its reset job ran, seconds, and
// anything LiteLLM cannot parse have no calendar AIR could place them on.
func TestParseBudgetDuration_RejectsDurationsOffTheCalendar(t *testing.T) {
	for _, duration := range []string{
		"", "0d", "0h", "1", "d", "1D", "1 d", "1dx", "1y",
		"2d", "14d", "2w", "2mo", "5h", "7h", "48h", "25h", "7m", "90m", "1s", "30s",
	} {
		_, ok := ParseBudgetDuration(duration)
		assert.False(t, ok, duration)
	}
}

func TestParseBudgetLimits(t *testing.T) {
	limits, err := ParseBudgetLimits([]byte(`[{"budget_duration":"1d","max_budget":10,"reset_at":"2026-03-11T00:00:00+00:00"},{"budget_duration":"1mo","max_budget":0}]`))
	require.NoError(t, err)
	assert.Equal(t, []BudgetLimit{{Duration: "1d", MaxBudget: 10}, {Duration: "1mo", MaxBudget: 0}}, limits)

	for _, raw := range []string{"", "null", "[]"} {
		limits, err := ParseBudgetLimits([]byte(raw))
		require.NoError(t, err, raw)
		assert.Empty(t, limits, raw)
	}

	for _, raw := range []string{`{}`, `[{"budget_duration":"1d"}]`, `[{"max_budget":10}]`, `[{"budget_duration":"1d","max_budget":"10"}]`} {
		_, err := ParseBudgetLimits([]byte(raw))
		assert.Error(t, err, raw)
	}
}

// Two entries of one budget_duration share a window row and counter: they are
// one window with the smaller limit, where the first of them stands. Equal
// calendars under different spellings stay apart, as their rows do.
func TestParseBudgetLimits_MergesEntriesOfOneDuration(t *testing.T) {
	limits, err := ParseBudgetLimits([]byte(`[
		{"budget_duration":"1d","max_budget":10},
		{"budget_duration":"7d","max_budget":50},
		{"budget_duration":"1d","max_budget":4},
		{"budget_duration":"daily","max_budget":8}
	]`))
	require.NoError(t, err)
	assert.Equal(t, []BudgetLimit{
		{Duration: "1d", MaxBudget: 4},
		{Duration: "7d", MaxBudget: 50},
		{Duration: "daily", MaxBudget: 8},
	}, limits)
}

// A budget_limits entry AIR cannot enforce refuses the key: before budget
// windows were read the key passed, and passing it now would leave a limit
// someone configured unenforced.
func TestTokenInfo_Validate_UnsupportedBudgetWindow(t *testing.T) {
	token := TokenInfo{BudgetWindows: []BudgetWindow{{BudgetLimit: BudgetLimit{Duration: "2d", MaxBudget: 10}}}}
	assert.ErrorIs(t, token.Validate(""), ErrBudgetWindowUnsupported)

	under := TokenInfo{BudgetWindows: []BudgetWindow{{
		BudgetLimit: BudgetLimit{Duration: "1d", MaxBudget: 10},
		Start:       time.Date(2026, 3, 10, 0, 0, 0, 0, time.UTC),
		Spend:       9.99,
	}}}
	assert.NoError(t, under.Validate(""))
}

func TestTokenInfo_BudgetWindowsStale(t *testing.T) {
	loadedAt := time.Date(2026, 3, 10, 12, 0, 0, 0, time.UTC)
	day, ok := ParseBudgetDuration("1d")
	require.True(t, ok)
	start, _ := day.Bounds(loadedAt, BudgetCalendar{})
	token := TokenInfo{BudgetWindows: []BudgetWindow{
		{BudgetLimit: BudgetLimit{Duration: "1d", MaxBudget: 10}, Start: start},
		{BudgetLimit: BudgetLimit{Duration: "2d", MaxBudget: 10}},
	}}

	assert.False(t, token.BudgetWindowsStale(loadedAt.Add(time.Hour), BudgetCalendar{}))
	assert.True(t, token.BudgetWindowsStale(time.Date(2026, 3, 11, 0, 0, 0, 0, time.UTC), BudgetCalendar{}),
		"the day the spend was loaded for has ended")
	assert.False(t, (&TokenInfo{}).BudgetWindowsStale(loadedAt, BudgetCalendar{}))
}

func TestTokenInfo_CloneCopiesBudgetWindows(t *testing.T) {
	token := &TokenInfo{BudgetWindows: []BudgetWindow{{BudgetLimit: BudgetLimit{Duration: "1d", MaxBudget: 10}}}}
	clone := token.Clone()
	clone.BudgetWindows[0].Spend = 5
	assert.Zero(t, token.BudgetWindows[0].Spend)
}
