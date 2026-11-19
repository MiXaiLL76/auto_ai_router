package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHandlePassthroughResponsesStreaming_ErrorStatusIsMasked covers a
// production incident: a passthrough-responses credential (providerPassthroughDefaults
// defaults every OpenAI-typed credential to true for /v1/responses) answered
// a 400 with its own raw body -- a nonstandard "type":"permission_error" and
// a provider-specific code ("session_blocked_by_cyber_policy") -- and
// handlePassthroughResponsesStreaming forwarded it to the client byte-for-byte
// via streamToClient, because nothing in this function special-cased a
// non-2xx status before reaching it. The masking every other response path
// applies never ran.
//
// The client must instead get this package's own classified body, the same
// as the non-streaming and native-streaming error paths produce for the
// identical provider signal -- and that signal is specifically
// session_blocked, not the generic content_policy_violation: a different
// credential answers the same request fine (this credential's session got
// blocked, not the prompt), so it must stay distinguishable from a genuine
// content refusal even though the message also happens to say "...content
// policy" in plain English.
func TestHandlePassthroughResponsesStreaming_ErrorStatusIsMasked(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	rawProviderBody := `{"error":{"code":"session_blocked_by_cyber_policy","message":"This session is blocked by content policy","type":"permission_error"}}`

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{}`))
	logCtx := &RequestLogContext{Request: req, StartTime: time.Now()}
	resp := &http.Response{
		StatusCode: http.StatusBadRequest,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(rawProviderBody)),
	}
	rec := httptest.NewRecorder()

	err := prx.handlePassthroughResponsesStreaming(rec, resp, "r_inference.cheap_global", "openai/gpt-6.1-sol", logCtx, nil)
	require.NoError(t, err)

	assert.Equal(t, http.StatusBadRequest, rec.Code, "the real status (400) must reach the client, not be swallowed")
	body := rec.Body.String()
	assert.NotContains(t, body, "session_blocked_by_cyber_policy", "the provider's own code must never reach the client")
	assert.NotContains(t, body, "permission_error", "the provider's own nonstandard type must never reach the client")
	assert.Contains(t, body, `"code":"session_blocked"`, "a credential-scoped block must stay distinct from a generic content-policy refusal")
	assert.Equal(t, "failure", logCtx.Status)
	assert.Equal(t, http.StatusBadRequest, logCtx.HTTPStatus)
}
