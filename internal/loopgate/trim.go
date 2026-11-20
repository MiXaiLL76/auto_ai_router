package loopgate

import (
	"hash/maphash"
	"strings"
	"unicode"
)

// Trim cuts the looped tail off s: the text ends where the second round of the
// loop begins. N-grams of n words occurring at least minRepeat times are the
// loop, and the earliest second occurrence of any of them is where it repeats.
// One round is kept, it is part of the answer. Text without such a repeat is
// returned as is. Words are split and n-grams hashed as the detector does, in
// O(words) memory.
func Trim(s string, n, minRepeat int) string {
	if n < 2 {
		return s
	}
	type seen struct{ count, second int }
	var starts []int
	grams := make(map[uint64]seen)
	ng := newRoller(n, maphash.MakeSeed())
	emit := func(word []byte, at int) {
		starts = append(starts, at)
		gram, ok := ng.push(word)
		if !ok {
			return
		}
		i := len(starts) - n // index of the n-gram's first word
		g, found := grams[gram]
		if !found {
			grams[gram] = seen{count: 1, second: -1}
			return
		}
		g.count++
		if g.second < 0 {
			g.second = i
		}
		grams[gram] = g
	}
	b := []byte(s)
	if start := splitWords(b, -1, emit); start >= 0 {
		emit(b[start:], start)
	}
	cut := -1
	for _, g := range grams {
		if g.count >= minRepeat && g.second >= 0 && (cut < 0 || g.second < cut) {
			cut = g.second
		}
	}
	if cut < 0 {
		return s
	}
	return strings.TrimRightFunc(s[:starts[cut]], unicode.IsSpace)
}
