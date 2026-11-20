package loopgate

import (
	"hash/maphash"
	"unicode"
	"unicode/utf8"
)

// maxWordBytes bounds one word. A longer run without whitespace (a long
// token, base64, text in a language written without spaces) is split into
// words of this size, so the pending word of a stream never grows unbounded.
const maxWordBytes = 256

// splitWords calls emit with every complete word of b and its offset. Words
// are separated by whitespace; a Han, Hiragana or Katakana character is a
// word of its own (Chinese and Japanese are written without spaces). start is
// the offset of a word already begun before b's new bytes (-1: none); the
// offset of the unfinished word at the end of b is returned (-1: none).
func splitWords(b []byte, start int, emit func(word []byte, at int)) int {
	for i := 0; i < len(b); {
		r, size := utf8.DecodeRune(b[i:])
		switch {
		case unicode.IsSpace(r):
			if start >= 0 {
				emit(b[start:i], start)
				start = -1
			}
		case isIdeograph(r):
			if start >= 0 {
				emit(b[start:i], start)
				start = -1
			}
			emit(b[i:i+size], i)
		case start < 0:
			start = i
		case i-start >= maxWordBytes:
			emit(b[start:i], start)
			start = i
		}
		i += size
	}
	return start
}

func isIdeograph(r rune) bool {
	return r >= 0x2E80 && (unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r))
}

const rollBase = 1099511628211

// roller hashes the n-gram of the last n words in O(1) per word: a rolling
// polynomial over the hashes of its words.
type roller struct {
	seed   maphash.Seed
	pow    uint64   // rollBase^n, removes the oldest word from the window
	window []uint64 // hashes of the last n words, a ring
	head   int
	roll   uint64
	words  int
}

func newRoller(n int, seed maphash.Seed) roller {
	pow := uint64(1)
	for i := 0; i < n; i++ {
		pow *= rollBase
	}
	return roller{seed: seed, pow: pow, window: make([]uint64, n)}
}

// push adds a word and returns the hash of the n-gram it ends; ok is false
// until n words were pushed.
func (r *roller) push(word []byte) (hash uint64, ok bool) {
	h := maphash.Bytes(r.seed, word) | 1
	old := r.window[r.head]
	r.window[r.head] = h
	r.head = (r.head + 1) % len(r.window)
	r.roll = r.roll*rollBase + h - old*r.pow
	r.words++
	return r.roll, r.words >= len(r.window)
}
