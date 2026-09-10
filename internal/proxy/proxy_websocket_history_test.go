package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWebSocketResponsesPreservesInputHistory(t *testing.T) {
	for _, tc := range []struct {
		name        string
		firstInput  string
		firstOutput string
		secondInput string
		wantHistory string
	}{
		{
			name:        "string input",
			firstInput:  `"Remember 48219. Reply only OK."`,
			firstOutput: `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"OK"}]}`,
			secondInput: `"What was the code?"`,
			wantHistory: `[{"role":"user","content":"Remember 48219. Reply only OK."},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"OK"}]},{"role":"user","content":"What was the code?"}]`,
		},
		{
			name:        "array input and tool result",
			firstInput:  `[{"role":"user","content":[{"type":"input_text","text":"Add 17 and 25."}]}]`,
			firstOutput: `{"type":"function_call","call_id":"call_sum","name":"sum_numbers","arguments":"{\"a\":17,\"b\":25}"}`,
			secondInput: `[{"type":"function_call_output","call_id":"call_sum","output":"42"}]`,
			wantHistory: `[{"role":"user","content":[{"type":"input_text","text":"Add 17 and 25."}]},{"type":"function_call","call_id":"call_sum","name":"sum_numbers","arguments":"{\"a\":17,\"b\":25}"},{"type":"function_call_output","call_id":"call_sum","output":"42"}]`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			outbound := make(chan map[string]json.RawMessage, 3)
			turn := 0
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				outbound <- body
				turn++
				output := `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"42"}]}`
				if turn == 1 {
					output = tc.firstOutput
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprintf(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_%d\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-6-astra\",\"output\":[%s],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\ndata: [DONE]\n\n", turn, output)
			}))
			defer upstream.Close()
			prx := NewTestProxyBuilder().WithSingleCredential("upstream", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").Build()
			server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
			defer server.Close()
			conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/v1/responses", http.Header{"Authorization": {"Bearer master-key"}})
			require.NoError(t, err)
			defer func() { assert.NoError(t, conn.Close()) }()

			for i, input := range []string{tc.firstInput, tc.secondInput, `"Repeat the result."`} {
				previous := ""
				if i > 0 {
					previous = fmt.Sprintf(`,"previous_response_id":"resp_%d"`, i)
				}
				wsWrite(t, conn, fmt.Sprintf(`{"type":"response.create","model":"gpt-6-astra","store":false,"input":%s%s}`, input, previous))
				wsEvent(t, conn, "response.completed")
				select {
				case body := <-outbound:
					assert.NotContains(t, body, "previous_response_id")
					switch i {
					case 1:
						assert.JSONEq(t, tc.wantHistory, string(body["input"]))
					case 2:
						want := strings.TrimSuffix(tc.wantHistory, "]") + `,{"type":"message","role":"assistant","content":[{"type":"output_text","text":"42"}]},{"role":"user","content":"Repeat the result."}]`
						assert.JSONEq(t, want, string(body["input"]))
					}
				case <-time.After(5 * time.Second):
					t.Fatal("upstream did not receive the request")
				}
			}
		})
	}
}
