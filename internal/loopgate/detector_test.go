package loopgate

import (
	stdjson "encoding/json"
	"fmt"
	"hash/maphash"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var testCfg = Config{NGram: 32, ArmWords: 3000, MinRepeat: 5, MinRatio: 0.35}

// uniqueText returns n distinct words.
func uniqueText(n int) string {
	words := make([]string, n)
	for i := range words {
		words[i] = fmt.Sprintf("w%d", i)
	}
	return strings.Join(words, " ")
}

// loopText returns a phrase of phraseWords distinct words repeated times times.
func loopText(phraseWords, times int) string {
	phrase := make([]string, phraseWords)
	for i := range phrase {
		phrase[i] = fmt.Sprintf("loop%d", i)
	}
	p := strings.Join(phrase, " ")
	return strings.TrimSpace(strings.Repeat(p+" ", times))
}

func feedInDeltas(c *Channel, s string, size int) {
	for len(s) > 0 {
		n := min(size, len(s))
		c.Feed(s[:n])
		s = s[n:]
	}
	c.Flush()
}

func TestChannel_UniqueTextNeverFires(t *testing.T) {
	c := NewChannel(testCfg, maphash.MakeSeed())
	feedInDeltas(c, uniqueText(10000), 7)
	assert.Equal(t, 10000, c.Words())
	assert.Equal(t, 1, c.Top())
	assert.Zero(t, c.Ratio())
	assert.Nil(t, c.Check(time.Second))
}

func TestChannel_LoopFiresOnceArmed(t *testing.T) {
	c := NewChannel(testCfg, maphash.MakeSeed())
	text := uniqueText(500) + " " + loopText(40, 100)
	feedInDeltas(c, text, 13)
	assert.Equal(t, 4500, c.Words())
	assert.GreaterOrEqual(t, c.Top(), 90)
	assert.Greater(t, c.Ratio(), 0.8)

	f := c.Check(80 * time.Second)
	require.NotNil(t, f)
	assert.Equal(t, 4500, f.Words)
	assert.Equal(t, 80*time.Second, f.Elapsed)
	assert.Nil(t, c.Check(81*time.Second), "fires only once")
	assert.Same(t, f, c.Fired())
}

func TestChannel_NotArmedBeforeArmWords(t *testing.T) {
	c := NewChannel(testCfg, maphash.MakeSeed())
	feedInDeltas(c, loopText(40, 50), 64) // 2000 words, all looping
	assert.Greater(t, c.Ratio(), 0.9)
	assert.False(t, c.Looping(testCfg.MinRatio))
	assert.Nil(t, c.Check(time.Second))
}

func TestChannel_FeedSplitsWordsAcrossDeltas(t *testing.T) {
	c := NewChannel(Config{NGram: 2, ArmWords: 0, MinRepeat: 2, MinRatio: 0.1}, maphash.MakeSeed())
	for _, d := range []string{"he", "llo wor", "ld\n\t", "привет", " мир"} {
		c.Feed(d)
	}
	assert.Equal(t, 3, c.Words(), "the trailing word waits for whitespace")
	c.Flush()
	assert.Equal(t, 4, c.Words())
}

// Same-shaped JSON objects with different values do not match a 32-word window.
func TestChannel_JSONArrayOfSimilarObjectsDoesNotFire(t *testing.T) {
	var b strings.Builder
	b.WriteString("[\n")
	for i := 0; i < 600; i++ {
		fmt.Fprintf(&b, "  { \"id\": %d, \"name\": \"item %d\", \"price\": %d.5, \"tags\": [ \"a\", \"b\" ] },\n", i, i, i*3)
	}
	b.WriteString("]")
	c := NewChannel(testCfg, maphash.MakeSeed())
	feedInDeltas(c, b.String(), 11)
	require.GreaterOrEqual(t, c.Words(), testCfg.ArmWords)
	assert.Nil(t, c.Check(time.Second))
}

func TestTrim(t *testing.T) {
	head := uniqueText(100)
	loop := loopText(40, 20)
	phrase := loopText(40, 1)

	got := Trim(head+"\n"+loop+"  ", 32, 5)
	assert.Equal(t, head+"\n"+phrase, got, "one round of the loop is kept")

	clean := uniqueText(5000)
	assert.Equal(t, clean, Trim(clean, 32, 5))
	assert.Equal(t, "a b", Trim("a b", 32, 5))
	assert.Equal(t, loop, Trim(loop, 1, 5), "n < 2 is a no-op")
}

func TestDetector_ObserveChatChunk(t *testing.T) {
	start := time.Unix(1000, 0)
	d := NewDetector(testCfg, start)

	chunk := func(field, text string) []byte {
		b, err := jsonMarshalString(text)
		require.NoError(t, err)
		return []byte(`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"` + field + `":` + b + `}}]}`)
	}

	// Tool call arguments are never counted.
	toolArgs := loopText(40, 200)
	argsJSON, err := jsonMarshalString(toolArgs)
	require.NoError(t, err)
	assert.Empty(t, d.ObserveChatChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":`+argsJSON+`}}]}}]}`), start))
	assert.Zero(t, d.Content().Words())

	// Another choice is ignored.
	assert.Empty(t, d.ObserveChatChunk([]byte(`{"choices":[{"index":1,"delta":{"content":"a b c "}}]}`), start))
	assert.Zero(t, d.Content().Words())

	// Non-chunk payloads are ignored.
	assert.Empty(t, d.ObserveChatChunk([]byte(`[DONE]`), start))
	assert.Empty(t, d.ObserveChatChunk([]byte(`{"delta": not json`), start))

	// Reasoning loops (both vLLM field spellings); content stays clean.
	var firings []Firing
	words := strings.Fields(uniqueText(200) + " " + loopText(40, 100))
	for i, w := range words {
		field := "reasoning_content"
		if i%2 == 1 {
			field = "reasoning"
		}
		firings = append(firings, d.ObserveChatChunk(chunk(field, w+" "), start.Add(time.Duration(i)*time.Millisecond))...)
		if i%3 == 0 {
			firings = append(firings, d.ObserveChatChunk(chunk("content", fmt.Sprintf("c%d ", i)), start)...)
		}
	}
	assert.Empty(t, d.FinishReason())
	firings = append(firings, d.ObserveChatChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`), start)...)
	assert.Equal(t, "length", d.FinishReason())
	firings = append(firings, d.Finish(start.Add(time.Minute))...)
	require.Len(t, firings, 1)
	assert.Equal(t, ChannelReasoning, firings[0].Channel)
	assert.Equal(t, testCfg.ArmWords, firings[0].Words, "fires on the first armed word")
	assert.Equal(t, time.Duration(testCfg.ArmWords-1)*time.Millisecond, firings[0].Elapsed)
	assert.Nil(t, d.Content().Fired())
}

func jsonMarshalString(s string) (string, error) {
	b, err := stdjson.Marshal(s)
	return string(b), err
}
