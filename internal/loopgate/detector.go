// Package loopgate detects a model stuck in a generation loop while its output
// streams: the same long word sequence repeated over and over until max_tokens.
// It never touches the engine or the sampler — it only watches the text.
//
// Ported from the loop-gate reference detector by Valerii Kovalskii
// (https://github.com/vakovalskii/loop-gate-skill, MIT License).
package loopgate

import (
	"hash/maphash"
	"time"
	"unicode"
	"unicode/utf8"
)

// Config holds the detector thresholds.
type Config struct {
	// NGram is the length, in words, of the sequence counted for repeats. A long
	// window matches only literally identical text (values included), so arrays
	// of same-shaped JSON objects with different data never look like a loop.
	NGram int
	// ArmWords is the number of words a channel must produce before it can
	// fire: honest short answers with lists and tables repeat short runs.
	ArmWords int
	// MinRepeat is how many times one NGram must occur.
	MinRepeat int
	// MinRatio is the minimum share of n-grams that are repeats (0..1).
	MinRatio float64
}

// Fire describes the moment a channel was first seen looping.
type Fire struct {
	Words   int
	Elapsed time.Duration
	Repeat  int
	Ratio   float64
}

const rollBase = 1099511628211

// Channel counts repeated n-grams of one text stream (content or reasoning).
// Each new word costs O(1): an n-gram hash is a rolling polynomial over the
// hashes of its words. Not safe for concurrent use.
type Channel struct {
	cfg    Config
	seed   maphash.Seed
	pow    uint64   // rollBase^NGram, removes the oldest word from the window
	window []uint64 // hashes of the last NGram words, a ring
	head   int
	roll   uint64
	words  int
	tail   []byte // unfinished word at the end of the previous delta
	counts map[uint64]int32
	rep    int
	top    int
	fired  *Fire
}

// NewChannel returns an empty channel. cfg.NGram must be positive.
func NewChannel(cfg Config, seed maphash.Seed) *Channel {
	pow := uint64(1)
	for i := 0; i < cfg.NGram; i++ {
		pow *= rollBase
	}
	return &Channel{
		cfg:    cfg,
		seed:   seed,
		pow:    pow,
		window: make([]uint64, cfg.NGram),
		counts: make(map[uint64]int32),
	}
}

// Words returns the number of complete words fed so far.
func (c *Channel) Words() int { return c.words }

// Top returns the occurrence count of the most repeated n-gram.
func (c *Channel) Top() int { return c.top }

// Fired returns the first firing, or nil.
func (c *Channel) Fired() *Fire { return c.fired }

// Ratio returns the share of n-grams that are repeats. The second occurrence
// of an n-gram counts both copies, every later occurrence counts one.
func (c *Channel) Ratio() float64 {
	grams := c.words - c.cfg.NGram + 1
	if grams <= 0 {
		return 0
	}
	return float64(c.rep) / float64(grams)
}

func (c *Channel) push(word []byte) {
	h := maphash.Bytes(c.seed, word) | 1
	old := c.window[c.head]
	c.window[c.head] = h
	c.head = (c.head + 1) % c.cfg.NGram
	c.roll = c.roll*rollBase + h - old*c.pow
	c.words++
	if c.words < c.cfg.NGram {
		return
	}
	n := c.counts[c.roll] + 1
	c.counts[c.roll] = n
	switch {
	case n == 2:
		c.rep += 2
	case n > 2:
		c.rep++
	}
	if int(n) > c.top {
		c.top = int(n)
	}
}

// Feed splits a delta into words on whitespace. A word not yet followed by
// whitespace waits for the next delta.
func (c *Channel) Feed(s string) {
	if s == "" {
		return
	}
	start := -1
	if len(c.tail) > 0 {
		start = 0
	}
	// The pending word is extended in place; tail is re-copied below.
	buf := c.tail
	c.tail = nil
	buf = append(buf, s...)
	for i := 0; i < len(buf); {
		r, size := utf8.DecodeRune(buf[i:])
		if unicode.IsSpace(r) {
			if start >= 0 {
				c.push(buf[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
		i += size
	}
	if start >= 0 {
		c.tail = append([]byte(nil), buf[start:]...)
	}
}

// Flush counts the pending unfinished word, for the end of the stream.
func (c *Channel) Flush() {
	if len(c.tail) > 0 {
		c.push(c.tail)
		c.tail = nil
	}
}

// Looping reports whether the channel is armed and looping at ratio cut.
func (c *Channel) Looping(cut float64) bool {
	return c.words >= c.cfg.ArmWords && c.top >= c.cfg.MinRepeat && c.Ratio() >= cut
}

// Check records and returns the first firing at the configured MinRatio. It
// returns nil when the channel does not loop and on every call after the
// first firing.
func (c *Channel) Check(elapsed time.Duration) *Fire {
	if c.fired != nil || !c.Looping(c.cfg.MinRatio) {
		return nil
	}
	c.fired = &Fire{Words: c.words, Elapsed: elapsed, Repeat: c.top, Ratio: c.Ratio()}
	return c.fired
}
