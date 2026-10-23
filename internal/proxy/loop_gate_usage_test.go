package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

func TestStripRunningUsage(t *testing.T) {
	for name, tc := range map[string]struct{ in, want string }{
		"running usage on a delta": {
			`data: {"id":"c","choices":[{"index":0,"delta":{"content":"a \"usage\":{}"}}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,"prompt_tokens_details":{"cached_tokens":0}}}` + "\n",
			`data: {"id":"c","choices":[{"index":0,"delta":{"content":"a \"usage\":{}"}}],"usage":null}` + "\n",
		},
		"spaces around": {
			`data: {"choices": [ {"index":0}], "usage" : {"total_tokens":4}}` + "\n",
			`data: {"choices": [ {"index":0}], "usage" : null}` + "\n",
		},
		"final usage chunk is kept": {
			`data: {"id":"c","choices":[],"usage":{"total_tokens":4}}` + "\n",
			`data: {"id":"c","choices":[],"usage":{"total_tokens":4}}` + "\n",
		},
		"no usage":        {`data: {"choices":[{"index":0}]}` + "\n", `data: {"choices":[{"index":0}]}` + "\n"},
		"usage null":      {`data: {"choices":[{"index":0}],"usage":null}` + "\n", `data: {"choices":[{"index":0}],"usage":null}` + "\n"},
		"comment":         {": keepalive\n", ": keepalive\n"},
		"done":            {"data: [DONE]\n", "data: [DONE]\n"},
		"unfinished JSON": {`data: {"choices":[{"index":0}],"usage":{"total_tokens":4` + "\n", `data: {"choices":[{"index":0}],"usage":{"total_tokens":4` + "\n"},
	} {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, string(stripRunningUsage([]byte(tc.in))))
		})
	}
}

func TestLoopGateUsageStripReader_LongAndSplitLines(t *testing.T) {
	long := strings.Repeat("x", 100*1024) // longer than the reader's buffer
	in := `data: {"choices":[{"index":0,"delta":{"content":"` + long + `"}}],"usage":{"total_tokens":1}}` + "\n\n" +
		`data: {"choices":[],"usage":{"total_tokens":2}}` + "\n\ndata: [DONE]\n\n"

	out, err := io.ReadAll(newLoopGateUsageStripReader(iotest.OneByteReader(strings.NewReader(in))))

	require.NoError(t, err)
	assert.Equal(t, strings.Replace(in, `"usage":{"total_tokens":1}`, `"usage":null`, 1), string(out))
}

func TestLoopGateRunningUsageBody(t *testing.T) {
	got, clientAsked, err := loopGateRunningUsageBody([]byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[]}`))
	require.NoError(t, err)
	assert.False(t, clientAsked)
	assert.JSONEq(t, `{"model":"m","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},"messages":[]}`, string(got))

	in := []byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}`)
	got, clientAsked, err = loopGateRunningUsageBody(in)
	require.NoError(t, err)
	assert.True(t, clientAsked)
	assert.Equal(t, string(in), string(got))

	_, _, err = loopGateRunningUsageBody([]byte(`[]`))
	assert.Error(t, err)
}

// writeLoopGateUsageStream is writeLoopGateEndless with vLLM's running usage
// (continuous_usage_stats) on every chunk, when the request asked for it.
func writeLoopGateUsageStream(w http.ResponseWriter, r *http.Request, body map[string]any, id string, endless bool) bool {
	continuous := false
	if so, ok := body["stream_options"].(map[string]any); ok {
		continuous, _ = so["continuous_usage_stats"].(bool)
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher := w.(http.Flusher)
	words := loopGateWords(500)
	n := 300 / 10
	if endless {
		n = 1_000_000
	}
	usage := func(i int) string {
		return fmt.Sprintf(`{"prompt_tokens":1000,"completion_tokens":%d,"total_tokens":%d}`, 12*i, 1000+12*i)
	}
	for i := 1; i <= n; i++ {
		if r.Context().Err() != nil {
			return true
		}
		var chunk []string
		if !endless || i*10 <= len(words) {
			chunk = words[(i-1)*10%len(words) : (i-1)*10%len(words)+10]
		} else {
			chunk = loopGateLoop(1)[(i*10)%40 : (i*10)%40+10]
		}
		field := "reasoning_content"
		if !endless {
			field = "content"
		}
		text, _ := json.Marshal(strings.Join(chunk, " ") + " ")
		running := ""
		if continuous {
			running = `,"usage":` + usage(i)
		}
		if _, err := fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[{\"index\":0,\"delta\":{%q:%s},\"finish_reason\":null}]%s}\n\n", id, field, text, running); err != nil {
			return true
		}
		flusher.Flush()
	}
	_, _ = fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"model\":\"Qwen/Qwen3.6-35B\",\"choices\":[],\"usage\":%s}\n\ndata: [DONE]\n\n", id, usage(n))
	return false
}

func TestLoopGate_EnforceStreamAsksForRunningUsageAndHidesIt(t *testing.T) {
	resetLoopGateEnforceMetrics()
	// The script reads the request body back from the snapshot.
	var script *scriptedVLLM
	var url string
	script, url = newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		bodies, _ := script.snapshot()
		if n == 0 {
			return writeLoopGateUsageStream(w, r, bodies[n], "c1", true)
		}
		return writeLoopGateUsageStream(w, r, bodies[n], "c2", false)
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))
	tpmBefore := prx.rateLimiter.GetCurrentTPM("lg-cred")

	w := postChat(t, prx, `{"model":"qwen","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`)

	require.Equal(t, http.StatusOK, w.Code)
	bodies, _ := script.snapshot()
	require.Len(t, bodies, 2, "restarted once")
	for i, b := range bodies {
		assert.Equal(t, map[string]any{"include_usage": true, "continuous_usage_stats": true}, b["stream_options"], "attempt %d", i+1)
	}
	chunks, done := loopGateSSEChunks(t, w.Body.String())
	require.True(t, done)
	withUsage := 0
	for _, c := range chunks {
		if u, ok := c["usage"].(map[string]any); ok && u != nil {
			withUsage++
			assert.Empty(t, c["choices"], "usage only on the closing usage chunk")
			assert.Equal(t, float64(1000+12*30), u["total_tokens"], "the final attempt's usage")
		}
	}
	assert.Equal(t, 1, withUsage)
	// The restarted attempt ran the prefill (1000 prompt tokens) and generated
	// up to the loop: the engine's running usage, not an estimate.
	aborted := 1000 + 12*config.DefaultLoopGateArmWords/10
	assert.GreaterOrEqual(t, prx.rateLimiter.GetCurrentTPM("lg-cred")-tpmBefore, 1000+12*30+aborted,
		"the restarted attempt's prompt and generated tokens count against the TPM")
}

func TestLoopGate_ClientAskedRunningUsageKeepsIt(t *testing.T) {
	resetLoopGateEnforceMetrics()
	var script *scriptedVLLM
	var url string
	script, url = newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
		bodies, _ := script.snapshot()
		return writeLoopGateUsageStream(w, r, bodies[n], "c1", false)
	})
	prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

	w := postChat(t, prx, `{"model":"qwen","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true},"messages":[{"role":"user","content":"hi"}]}`)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 31, strings.Count(w.Body.String(), `"usage":{`), "every chunk keeps the running usage the client asked for")
}

func TestLoopGateObserver_KeepaliveWhileRestartIsOnItsWay(t *testing.T) {
	saved := loopGateKeepaliveInterval
	loopGateKeepaliveInterval = 10 * time.Millisecond
	t.Cleanup(func() { loopGateKeepaliveInterval = saved })

	o := newTestLoopGateObserver(t, config.LoopGateEnforce)
	var reasoning []string
	reasoning = append(reasoning, loopGateWords(100)...)
	reasoning = append(reasoning, loopGateLoop(20)...)
	input := strings.ReplaceAll(loopGateSSE(reasoning), `"content"`, `"reasoning_content"`)
	release := make(chan struct{})
	o.respawn = func(int, float64) (io.ReadCloser, error) {
		<-release // the new attempt's prefill
		return io.NopCloser(strings.NewReader(loopGateSSE(loopGateWords(30)))), nil
	}
	r := o.wrap(io.NopCloser(strings.NewReader(input)))

	var out strings.Builder
	buf := make([]byte, 256)
	for strings.Count(out.String(), string(loopGateKeepalive)) < 3 {
		n, err := r.Read(buf)
		require.NoError(t, err)
		out.Write(buf[:n])
	}
	close(release)
	rest, err := io.ReadAll(r)
	require.NoError(t, err)
	out.Write(rest)
	o.finish()

	assert.Equal(t, 1, o.restarts)
	assert.Nil(t, o.cut, "the restart went through")
	chunks, done := loopGateSSEChunks(t, out.String())
	assert.True(t, done)
	assert.NotEmpty(t, chunks)
	assert.Contains(t, out.String(), `w29 "`, "the new attempt follows the keepalives")
}

func TestLoopGateObserver_FinishClosesRestartStillOnItsWay(t *testing.T) {
	o := newTestLoopGateObserver(t, config.LoopGateEnforce)
	var reasoning []string
	reasoning = append(reasoning, loopGateWords(100)...)
	reasoning = append(reasoning, loopGateLoop(20)...)
	input := strings.ReplaceAll(loopGateSSE(reasoning), `"content"`, `"reasoning_content"`)
	release := make(chan struct{})
	restarted := &closeTrackingBody{Reader: strings.NewReader(loopGateSSE(loopGateWords(30)))}
	closed := make(chan struct{})
	o.respawn = func(int, float64) (io.ReadCloser, error) {
		<-release
		return &closeNotifyBody{closeTrackingBody: restarted, closed: closed}, nil
	}
	r := o.wrap(io.NopCloser(strings.NewReader(input)))
	buf := make([]byte, 256)
	for o.restarting == nil {
		_, err := r.Read(buf)
		require.NoError(t, err)
	}

	o.finish() // the client went away while the restart was on its way
	close(release)
	assert.Zero(t, o.restarts, "the restart never continued the stream")

	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("the restart that arrived after the handler returned is not closed")
	}
}

type closeNotifyBody struct {
	*closeTrackingBody
	closed chan struct{}
}

func (b *closeNotifyBody) Close() error {
	_ = b.closeTrackingBody.Close()
	close(b.closed)
	return nil
}

func TestLoopGate_RestartKeepaliveReachesTheClient(t *testing.T) {
	saved := loopGateKeepaliveInterval
	loopGateKeepaliveInterval = 10 * time.Millisecond
	t.Cleanup(func() { loopGateKeepaliveInterval = saved })

	for name, tc := range map[string]struct{ path, body, end string }{
		"chat":               {"/v1/chat/completions", loopGateStreamRequest, "data: [DONE]"},
		"converted messages": {"/v1/messages", `{"model":"qwen","max_tokens":100000,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, "message_stop"},
	} {
		t.Run(name, func(t *testing.T) {
			resetLoopGateEnforceMetrics()
			_, url := newScriptedVLLM(t, func(n int, w http.ResponseWriter, r *http.Request) bool {
				if n == 0 {
					return writeLoopGateEndless(w, r, "c1")
				}
				time.Sleep(100 * time.Millisecond) // the restart's prefill
				return writeLoopGateStream(w, r, "c2", loopGateWords(20), []string{"final", "answer"}, "stop")
			})
			prx := newLoopGateProxyWithGate(t, url, enforceGate(0, 3))

			w := postLoopGate(t, prx, tc.path, tc.body)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			body := w.Body.String()
			keepalive := strings.Index(body, string(loopGateKeepalive))
			require.GreaterOrEqual(t, keepalive, 0, "keepalives while the restart is on its way")
			assert.Greater(t, strings.Index(body, "final answer"), keepalive, "then the new attempt")
			assert.Contains(t, body, tc.end)
		})
	}
}
