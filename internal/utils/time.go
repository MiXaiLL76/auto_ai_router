package utils

import "time"

// NowUTC returns current time in UTC timezone.
// Used throughout the codebase for consistent timestamp handling.
// This centralized function simplifies mocking in tests and ensures
// consistent UTC usage across all components.
func NowUTC() time.Time {
	return time.Now().UTC()
}
