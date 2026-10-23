package loopgate

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Trim cuts the looped tail off s: the text ends where the second round of the
// loop begins. N-grams of n words occurring at least minRepeat times are the
// loop, and the earliest second occurrence of any of them is where it repeats.
// One round is kept, it is part of the answer. Text without such a repeat is
// returned as is.
func Trim(s string, n, minRepeat int) string {
	var starts []int
	var words []string
	start := -1
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if unicode.IsSpace(r) {
			if start >= 0 {
				starts, words = append(starts, start), append(words, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
		i += size
	}
	if start >= 0 {
		starts, words = append(starts, start), append(words, s[start:])
	}
	if n < 2 || len(words) < n {
		return s
	}
	type seen struct{ count, second int }
	grams := make(map[string]*seen, len(words))
	for i := 0; i+n <= len(words); i++ {
		k := strings.Join(words[i:i+n], " ")
		g := grams[k]
		if g == nil {
			grams[k] = &seen{count: 1, second: -1}
			continue
		}
		g.count++
		if g.second < 0 {
			g.second = i
		}
	}
	cut := -1
	for _, g := range grams {
		if g.count >= minRepeat && (cut < 0 || g.second < cut) {
			cut = g.second
		}
	}
	if cut < 0 {
		return s
	}
	return strings.TrimRightFunc(s[:starts[cut]], unicode.IsSpace)
}
