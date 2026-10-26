package proxy

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
)

// geminiContinuationRetryDelay is the pause before the one retry of a follow-up
// request that failed in transit or with a 429/5xx.
var geminiContinuationRetryDelay = time.Second

// geminiContinuation sends the follow-up requests of a Gemini long decode (see
// vertex/continuation.go) for one client request: the converted body of the first
// request again, now carrying the last continuation token, to its URL and through
// its credential -- a continuation token is only good on the account that issued
// it. Follow-ups stop as soon as the client is gone, and a generation that cannot
// be continued ends where it stopped: the client gets the output so far as cut
// short, and every finished request's usage is billed.
type geminiContinuation struct {
	p         *Proxy
	clientCtx context.Context
	cred      *config.CredentialConfig
	modelID   string // the model the credential's limits and bans are kept for
	requestID string
	// template is the first upstream request: method, URL, headers, and a context
	// carrying the upstream deadline and the credential's egress proxy.
	template *http.Request
	body     []byte
	requests int // follow-up requests sent
}

// newGeminiContinuation returns the follow-up sender for a request to realModelID
// through cred, or nil when that model does not decode past one request.
func (p *Proxy) newGeminiContinuation(r *http.Request, cred *config.CredentialConfig, modelID, realModelID string,
	template *http.Request, body []byte, logCtx *RequestLogContext) *geminiContinuation {
	switch cred.EffectiveProviderType() {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
	default:
		return nil
	}
	if !vertex.SupportsLongDecode(realModelID) {
		return nil
	}
	c := &geminiContinuation{
		p:         p,
		clientCtx: r.Context(),
		cred:      cred,
		modelID:   modelID,
		template:  template,
		body:      body,
	}
	if logCtx != nil {
		c.requestID = logCtx.RequestID
	}
	return c
}

// errContinuationLimit stops a generation after vertex.MaxContinuations follow-ups.
var errContinuationLimit = errors.New("continuation request limit reached")

// send issues the next follow-up request with token and returns its successful
// response; any other outcome is an error, after one retry of a transport error
// or a 429/5xx.
func (c *geminiContinuation) send(token string) (*http.Response, error) {
	if c.requests >= vertex.MaxContinuations {
		return nil, errContinuationLimit
	}
	c.requests++
	body, err := vertex.ContinuationRequest(c.body, token)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := range 2 {
		if attempt > 0 {
			select {
			case <-c.clientCtx.Done():
				return nil, c.clientCtx.Err()
			case <-time.After(geminiContinuationRetryDelay):
			}
		}
		resp, retryable, err := c.sendOnce(body)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !retryable {
			break
		}
	}
	return nil, lastErr
}

// sendOnce sends one follow-up request with resendToCredential; retryable
// reports a transport error or a 429/5xx while the client is still there.
func (c *geminiContinuation) sendOnce(body []byte) (*http.Response, bool, error) {
	if err := c.clientCtx.Err(); err != nil {
		return nil, false, err
	}
	c.p.logger.InfoContext(c.template.Context(), "Continuing Gemini long decode",
		"credential", c.cred.Name, "model", c.modelID, "continuation", c.requests,
		"request_id", c.requestID)
	resp, err := c.p.resendToCredential(c.clientCtx, c.template, body, c.cred, c.modelID)
	if err == nil {
		return resp, false, nil
	}
	if c.clientCtx.Err() != nil {
		return nil, false, err
	}
	var statusErr *credentialStatusError
	if errors.As(err, &statusErr) {
		return nil, statusErr.retryable(), fmt.Errorf("continuation request: %w", err)
	}
	return nil, true, err
}

// complete continues a non-streamed generation whose first response is first,
// and returns the whole generation as one response. A follow-up that fails ends it
// there, as a response cut short at the output cap.
func (c *geminiContinuation) complete(first []byte) []byte {
	bodies := [][]byte{first}
	for last := first; ; {
		finishReason, token, ok := vertex.ReadContinuation(last)
		if !ok || !vertex.IsContinuation(finishReason) || token == "" {
			break
		}
		next, err := c.fetch(token)
		if err != nil {
			c.stopped(err)
			break
		}
		bodies = append(bodies, next)
		last = next
	}
	if len(bodies) == 1 {
		return first
	}
	merged, err := vertex.MergeContinuationResponses(bodies)
	if err != nil {
		c.p.logger.ErrorContext(c.template.Context(), "Failed to merge Gemini long decode responses",
			"credential", c.cred.Name, "model", c.modelID, "error", err, "request_id", c.requestID)
		return first
	}
	return merged
}

func (c *geminiContinuation) fetch(token string) ([]byte, error) {
	resp, err := c.send(token)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	return c.p.readLimitedResponseBody(resp.Body)
}

func (c *geminiContinuation) stopped(err error) {
	if c.clientCtx.Err() != nil {
		return
	}
	c.p.logger.WarnContext(c.template.Context(), "Gemini long decode not continued, returning the output so far",
		"credential", c.cred.Name, "model", c.modelID, "continuation", c.requests, "error", err,
		"request_id", c.requestID)
}

// stream returns the body of a streamed generation whose first response is first:
// the SSE of every request one after the other. At the end of a request that
// stopped with CONTINUATION it sends the next one, carrying the last token the
// stream gave. The converters read the chained requests as one generation.
func (c *geminiContinuation) stream(first io.ReadCloser) io.ReadCloser {
	return &continuationStream{c: c, body: first, reader: bufio.NewReader(first)}
}

type continuationStream struct {
	c       *geminiContinuation
	reader  *bufio.Reader
	pending []byte
	err     error
	// finishReason and token of the running request, from its chunks.
	finishReason string
	token        string

	// body is the running request's; Close may come from another goroutine
	// (an abort, a timeout) while Read switches to the next request.
	mu     sync.Mutex
	body   io.ReadCloser
	closed bool
}

func (s *continuationStream) Read(p []byte) (int, error) {
	for len(s.pending) == 0 {
		if s.err != nil {
			return 0, s.err
		}
		line, err := s.reader.ReadBytes('\n')
		if len(line) > 0 {
			s.observe(line)
			s.pending = line
		}
		if err == nil {
			continue
		}
		if !errors.Is(err, io.EOF) || !s.next() {
			s.err = err
			continue
		}
		// End the request's last event before the next request's first one.
		if len(line) > 0 && line[len(line)-1] != '\n' {
			s.pending = append(s.pending, '\n')
		}
		s.pending = append(s.pending, '\n')
	}
	n := copy(p, s.pending)
	s.pending = s.pending[n:]
	return n, nil
}

// observe keeps the finish reason and the latest continuation token of the running
// request. A token comes with the CONTINUATION finish and, at checkpoints, in chunks
// without one; a final finish of another kind clears it.
func (s *continuationStream) observe(line []byte) {
	payload, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok || (!bytes.Contains(payload, []byte(`"finishReason"`)) && !bytes.Contains(payload, []byte(`"continuationToken"`))) {
		return
	}
	finishReason, token, ok := vertex.ReadContinuation(bytes.TrimSpace(payload))
	if !ok {
		return
	}
	if token != "" {
		s.token = token
	}
	if finishReason != "" {
		s.finishReason = finishReason
		if !vertex.IsContinuation(finishReason) && token == "" {
			s.token = ""
		}
	}
}

// next starts the follow-up request when the request that just ended asked for one.
func (s *continuationStream) next() bool {
	if !vertex.IsContinuation(s.finishReason) || s.token == "" {
		return false
	}
	resp, err := s.c.send(s.token)
	if err == nil && !IsStreamingResponse(resp) {
		_ = resp.Body.Close()
		err = fmt.Errorf("continuation request answered %q, not an event stream", resp.Header.Get("Content-Type"))
	}
	if err != nil {
		s.c.stopped(err)
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		_ = resp.Body.Close()
		return false
	}
	_ = s.body.Close()
	s.body = resp.Body
	s.reader = bufio.NewReader(resp.Body)
	s.finishReason, s.token = "", ""
	return true
}

func (s *continuationStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return s.body.Close()
}
