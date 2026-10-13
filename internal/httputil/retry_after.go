package httputil

import "strconv"

// RetryAfterSeconds parses an upstream Retry-After header (integer seconds).
// HTTP dates and garbage return 0; the health worker then falls back to its
// own cooldown rules.
func RetryAfterSeconds(v string) int {
	if v == "" {
		return 0
	}
	secs, err := strconv.Atoi(v)
	if err != nil {
		return 0
	}
	if secs < 0 {
		return 0
	}
	return secs
}
