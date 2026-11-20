package security

import "strconv"

// RedactBodyForLog omits arbitrary content before it reaches a log sink.
// Credentials can appear in ordinary text, so truncation or matching known key
// formats is insufficient. Byte counts retain diagnostic context without text.
func RedactBodyForLog(body string) string {
	if body == "" {
		return ""
	}
	return "[REDACTED body bytes=" + strconv.Itoa(len(body)) + "]"
}
