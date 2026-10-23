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

func TestDetector_LoopingChannelAndMeta(t *testing.T) {
	cfg := Config{NGram: 4, ArmWords: 20, MinRepeat: 3, MinRatio: 0.3}
	d := NewDetector(cfg, time.Unix(0, 0))
	assert.Empty(t, d.LoopingChannel(0.3))

	d.ObserveChatChunk([]byte(`{"id":"c1","created":7,"model":"qwen","choices":[{"index":0,"delta":{"content":"`+loopText(5, 10)+` "}}]}`), time.Unix(1, 0))
	assert.Equal(t, ChannelContent, d.LoopingChannel(0.3))
	assert.Empty(t, d.LoopingChannel(1.01), "cut ratio above the reached ratio")
	assert.Equal(t, ChunkMeta{ID: "c1", Created: 7, Model: "qwen"}, d.Meta())

	d.ObserveChatChunk([]byte(`{"id":"","choices":[{"index":0,"delta":{"reasoning":"`+loopText(5, 10)+` "}}]}`), time.Unix(2, 0))
	assert.Equal(t, ChannelReasoning, d.LoopingChannel(0.3), "reasoning is reported first")
	assert.Equal(t, "c1", d.Meta().ID, "an empty id does not overwrite")
}

func TestDetector_AnswerStarted(t *testing.T) {
	d := NewDetector(testCfg, time.Unix(0, 0))
	for _, payload := range []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"thinking hard"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"\n\n "}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":null}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[]}}]}`,
		`{"choices":[{"index":1,"delta":{"content":"other choice"}}]}`,
	} {
		d.ObserveChatChunk([]byte(payload), time.Unix(0, 0))
	}
	assert.False(t, d.AnswerStarted(), "role, reasoning, whitespace and empty tool_calls are not an answer")

	d.ObserveChatChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"Hi"}}]}`), time.Unix(0, 0))
	assert.True(t, d.AnswerStarted())

	d = NewDetector(testCfg, time.Unix(0, 0))
	d.ObserveChatChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":""}}]}}]}`), time.Unix(0, 0))
	assert.True(t, d.AnswerStarted(), "a tool call is an answer")
}

func TestDetector_ObserveResponsesEvent(t *testing.T) {
	cfg := Config{NGram: 4, ArmWords: 20, MinRepeat: 3, MinRatio: 0.3}
	d := NewDetector(cfg, time.Unix(0, 0))
	now := time.Unix(1, 0)
	ev := func(s string) []Firing { return d.Observe(FormatResponses, []byte(s), now) }

	ev(`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","created_at":5,"model":"qwen","status":"in_progress"}}`)
	ev(`{"type":"response.output_item.added","sequence_number":1,"item":{"type":"reasoning"}}`)
	assert.False(t, d.AnswerStarted(), "a reasoning item is not an answer")
	firings := ev(`{"type":"response.reasoning_text.delta","sequence_number":2,"delta":"` + loopText(5, 10) + ` "}`)
	require.Len(t, firings, 1)
	assert.Equal(t, ChannelReasoning, firings[0].Channel)
	ev(`{"type":"response.reasoning_summary_text.delta","sequence_number":3,"delta":"sum "}`)
	assert.Equal(t, 51, d.Reasoning().Words())

	ev(`{"type":"response.output_item.added","sequence_number":4,"item":{"type":"message"}}`)
	assert.False(t, d.AnswerStarted())
	ev(`{"type":"response.output_text.delta","sequence_number":5,"delta":"Hello "}`)
	assert.True(t, d.AnswerStarted())
	assert.Equal(t, 1, d.Content().Words())

	ev(`{"type":"response.incomplete","sequence_number":9,"response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`)
	assert.Equal(t, "max_output_tokens", d.FinishReason())
	assert.Equal(t, 9, d.Sequence())
	assert.Equal(t, ChunkMeta{ID: "resp_1", Created: 5, Model: "qwen"}, d.Meta())

	d = NewDetector(cfg, time.Unix(0, 0))
	d.Observe(FormatResponses, []byte(`{"type":"response.output_item.added","item":{"type":"function_call"}}`), now)
	assert.True(t, d.AnswerStarted(), "a function call is an answer")
	d.Observe(FormatResponses, []byte(`{"type":"response.function_call_arguments.delta","delta":"`+loopText(5, 10)+`"}`), now)
	assert.Zero(t, d.Content().Words(), "function arguments are not counted")
}

func TestDetector_ObserveMessagesEvent(t *testing.T) {
	cfg := Config{NGram: 4, ArmWords: 20, MinRepeat: 3, MinRatio: 0.3}
	d := NewDetector(cfg, time.Unix(0, 0))
	now := time.Unix(1, 0)
	ev := func(s string) []Firing { return d.Observe(FormatMessages, []byte(s), now) }

	ev(`{"type":"message_start","message":{"id":"msg_1","model":"qwen","role":"assistant","content":[]}}`)
	_, open := d.OpenBlock()
	assert.False(t, open)
	ev(`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`)
	firings := ev(`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"` + loopText(5, 10) + ` "}}`)
	require.Len(t, firings, 1)
	assert.Equal(t, ChannelReasoning, firings[0].Channel)
	idx, open := d.OpenBlock()
	assert.True(t, open)
	assert.Equal(t, 0, idx)
	ev(`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}`)
	ev(`{"type":"content_block_stop","index":0}`)
	_, open = d.OpenBlock()
	assert.False(t, open)
	assert.False(t, d.AnswerStarted())

	ev(`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`)
	ev(`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi there "}}`)
	assert.True(t, d.AnswerStarted())
	assert.Equal(t, 2, d.Content().Words())
	ev(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`)
	assert.Equal(t, "end_turn", d.FinishReason())
	assert.Equal(t, ChunkMeta{ID: "msg_1", Model: "qwen"}, d.Meta())

	d = NewDetector(cfg, time.Unix(0, 0))
	d.Observe(FormatMessages, []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"f","input":{}}}`), now)
	assert.True(t, d.AnswerStarted(), "a tool call is an answer")
	d.Observe(FormatMessages, []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"`+loopText(5, 10)+`"}}`), now)
	assert.Zero(t, d.Content().Words())
}
