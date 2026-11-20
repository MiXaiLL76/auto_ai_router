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
	// Window is how many of the latest n-grams are counted (0 = the
	// default, max(4*ArmWords, minWindow)). Older ones are forgotten, so the
	// memory of a channel is bounded however long the generation runs, and a
	// loop that starts late in a long answer is not diluted by its healthy
	// start.
	Window int
}

// minWindow is the smallest default window, for small ArmWords.
const minWindow = 4096

func (c Config) window() int {
	if c.Window > 0 {
		return c.Window
	}
	return max(4*c.ArmWords, minWindow)
}

// Fire describes the moment a channel was first seen looping.
type Fire struct {
	Words   int
	Elapsed time.Duration
	Repeat  int
	Ratio   float64
}

// Channel counts repeated n-grams of one text stream (content or reasoning)
// within a window of the latest cfg.Window n-grams. Each new word costs O(1)
// (see roller). Not safe for concurrent use.
type Channel struct {
	cfg    Config
	window int
	ng     roller
	tail   []byte   // unfinished word at the end of the previous delta
	grams  []uint64 // the n-grams in the window; a ring once full
	oldest int      // position of the oldest n-gram in a full ring
	counts map[uint64]int32
	freq   []int32 // freq[c]: how many distinct n-grams occur c times in the window
	rep    int
	top    int
	fired  *Fire
}

// NewChannel returns an empty channel. cfg.NGram must be positive.
func NewChannel(cfg Config, seed maphash.Seed) *Channel {
	return &Channel{
		cfg:    cfg,
		window: cfg.window(),
		ng:     newRoller(cfg.NGram, seed),
		counts: make(map[uint64]int32),
	}
}

// Words returns the number of complete words fed so far.
func (c *Channel) Words() int { return c.ng.words }

// Top returns the occurrence count of the most repeated n-gram in the window.
func (c *Channel) Top() int { return c.top }

// Fired returns the first firing, or nil.
func (c *Channel) Fired() *Fire { return c.fired }

// Ratio returns the share of the window's n-grams that are repeats: every
// occurrence of an n-gram that occurs more than once in the window.
func (c *Channel) Ratio() float64 {
	if len(c.grams) == 0 {
		return 0
	}
	return float64(c.rep) / float64(len(c.grams))
}

func (c *Channel) push(word []byte, _ int) {
	gram, ok := c.ng.push(word)
	if !ok {
		return
	}
	if len(c.grams) < c.window {
		c.grams = append(c.grams, gram)
	} else {
		c.forget(c.grams[c.oldest])
		c.grams[c.oldest] = gram
		c.oldest = (c.oldest + 1) % c.window
	}
	n := c.counts[gram] + 1
	c.counts[gram] = n
	switch {
	case n == 2:
		c.rep += 2
	case n > 2:
		c.rep++
	}
	c.recount(int(n)-1, int(n))
	if int(n) > c.top {
		c.top = int(n)
	}
}

// forget removes one occurrence of gram, the oldest n-gram of the window.
func (c *Channel) forget(gram uint64) {
	n := c.counts[gram]
	switch {
	case n == 2:
		c.rep -= 2
	case n > 2:
		c.rep--
	}
	if n == 1 {
		delete(c.counts, gram)
	} else {
		c.counts[gram] = n - 1
	}
	c.recount(int(n), int(n)-1)
	// The forgotten n-gram now occurs n-1 times, so the top drops by one at
	// most.
	if int(n) == c.top && c.freq[n] == 0 {
		c.top--
	}
}

// recount moves one distinct n-gram from occurring from times to to times.
func (c *Channel) recount(from, to int) {
	if from > 0 {
		c.freq[from]--
	}
	if to > 0 {
		for len(c.freq) <= to {
			c.freq = append(c.freq, 0)
		}
		c.freq[to]++
	}
}

// Feed splits a delta into words (see splitWords). A word not yet followed by
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
	if start = splitWords(buf, start, c.push); start >= 0 {
		c.tail = append([]byte(nil), buf[start:]...)
	}
}

// Flush counts the pending unfinished word, for the end of the stream.
func (c *Channel) Flush() {
	if len(c.tail) > 0 {
		c.push(c.tail, 0)
		c.tail = nil
	}
}

// Looping reports whether the channel is armed and looping at ratio cut.
func (c *Channel) Looping(cut float64) bool {
	return c.ng.words >= c.cfg.ArmWords && c.top >= c.cfg.MinRepeat && c.Ratio() >= cut
}

// Check records and returns the first firing at the configured MinRatio. It
// returns nil when the channel does not loop and on every call after the
// first firing.
func (c *Channel) Check(elapsed time.Duration) *Fire {
	if c.fired != nil || !c.Looping(c.cfg.MinRatio) {
		return nil
	}
	c.fired = &Fire{Words: c.ng.words, Elapsed: elapsed, Repeat: c.top, Ratio: c.Ratio()}
	return c.fired
}
