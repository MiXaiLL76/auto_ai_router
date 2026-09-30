package proxy

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
)

// embedContentFanOutConcurrency caps the parallel upstream calls one
// fanned-out embeddings request makes (Vertex AI embedContent embeds a single
// content per call).
const embedContentFanOutConcurrency = 8

// embeddingFanOutBodies returns conv's per-input embedContent bodies, or nil
// when the request goes upstream as one call (conv is nil on the native
// Responses path).
func embeddingFanOutBodies(conv *converter.ProviderConverter) [][]byte {
	if conv == nil {
		return nil
	}
	return conv.EmbeddingFanOutBodies()
}

// embedContentFanOutReplies keeps the embedContent replies a fanned-out
// request has already received, across its credential attempts. Each reply is
// a call the provider has billed, so a retry on the next credential sends only
// the inputs still missing. Replies are keyed by the exact upstream body and
// reused only for the same provider model: vectors of different models must
// never be mixed in one response.
type embedContentFanOutReplies struct {
	model   string
	byInput map[[sha256.Size]byte][]byte
}

// forModel drops the kept replies when model differs from the one they came from.
func (c *embedContentFanOutReplies) forModel(model string) {
	if c.byInput == nil || c.model != model {
		c.model = model
		c.byInput = make(map[[sha256.Size]byte][]byte)
	}
}

type embedContentFanOutResult struct {
	resp *http.Response
	body []byte
	err  error
}

// doEmbedContentFanOut sends one Vertex AI embedContent call per body that
// replies has no answer for yet, cloning template (URL, headers, context — and
// with it the credential's egress proxy) for each, and answers as if it were
// a single upstream call:
//
//   - every input has a reply: a synthesized 200 whose body is the replies
//     joined in input order by vertex.MergeEmbedContentResponses;
//   - some call got a non-2xx reply: that reply (the lowest input index), so
//     retry, fail2ban and error mapping see the provider's own status and body;
//   - otherwise the first transport error.
//
// The first failure stops launching further calls, since a partial set of
// vectors is useless to the client. Calls already in flight are left to finish:
// the provider bills them either way, and their replies stay in replies for
// the next credential attempt.
//
// inputFault reports a refusal that faults the input rather than the
// credential: the provider rejected one input with a 400/413/422 while
// embedding another on this same credential in this attempt. Every other
// credential would reject it the same way, so the caller must not retry.
func (p *Proxy) doEmbedContentFanOut(template *http.Request, model string, bodies [][]byte, replies *embedContentFanOutReplies) (resp *http.Response, inputFault bool, err error) {
	replies.forModel(model)
	keys := make([][sha256.Size]byte, len(bodies))
	for i, body := range bodies {
		keys[i] = sha256.Sum256(body)
	}

	ctx := template.Context()
	stop := make(chan struct{})
	var stopOnce sync.Once
	halt := func() { stopOnce.Do(func() { close(stop) }) }

	results := make([]embedContentFanOutResult, len(bodies))
	sem := make(chan struct{}, embedContentFanOutConcurrency)
	var wg sync.WaitGroup
	for i, body := range bodies {
		if _, ok := replies.byInput[keys[i]]; ok {
			continue
		}
		wg.Add(1)
		go func(i int, body []byte) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-stop:
				return
			case <-ctx.Done():
				results[i].err = ctx.Err()
				return
			}
			defer func() { <-sem }()
			// A free slot and a stop can be ready at once; stop wins.
			select {
			case <-stop:
				return
			default:
			}

			req := template.Clone(ctx)
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
			req.ContentLength = int64(len(body))
			req.Header.Set("Content-Length", strconv.Itoa(len(body)))

			resp, err := p.client.Do(req) //nolint:gosec // G704: template's URL, host comes from the matched credential
			if err != nil {
				results[i].err = err
				halt()
				return
			}
			data, readErr := p.readLimitedResponseBody(resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				results[i].err = readErr
				halt()
				return
			}
			results[i] = embedContentFanOutResult{resp: resp, body: data}
			if !isSuccessStatus(resp.StatusCode) {
				halt()
			}
		}(i, body)
	}
	wg.Wait()

	var (
		refused      *embedContentFanOutResult
		transportErr error
		succeeded    *http.Response
	)
	for i := range results {
		result := &results[i]
		switch {
		case result.resp != nil && isSuccessStatus(result.resp.StatusCode):
			replies.byInput[keys[i]] = []byte(decodeResponseBody(result.body, result.resp.Header.Get("Content-Encoding")))
			if succeeded == nil {
				succeeded = result.resp
			}
		case result.resp != nil:
			if refused == nil {
				refused = result
			}
		case result.err != nil:
			if transportErr == nil {
				transportErr = result.err
			}
		}
	}
	if refused != nil {
		inputFault = succeeded != nil && isEmbedContentInputFault(refused.resp.StatusCode)
		return responseWithBody(refused.resp, refused.body), inputFault, nil
	}
	if transportErr != nil {
		return nil, false, transportErr
	}

	ordered := make([][]byte, len(bodies))
	for i := range bodies {
		ordered[i] = replies.byInput[keys[i]]
	}
	merged, err := vertex.MergeEmbedContentResponses(ordered)
	if err != nil {
		// An unusable reply must not be replayed into the next attempt.
		replies.byInput = nil
		return nil, false, fmt.Errorf("merge embedContent fan-out responses: %w", err)
	}
	out := &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Request:    template,
	}
	if succeeded != nil {
		out.Status, out.StatusCode = succeeded.Status, succeeded.StatusCode
		out.Proto, out.ProtoMajor, out.ProtoMinor = succeeded.Proto, succeeded.ProtoMajor, succeeded.ProtoMinor
		out.Header = succeeded.Header.Clone()
	}
	out.Header.Del("Content-Encoding")
	out.Header.Del("Content-Length")
	out.Header.Set("Content-Type", "application/json")
	return responseWithBody(out, merged), false, nil
}

func isSuccessStatus(statusCode int) bool {
	return statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices
}

// isEmbedContentInputFault reports the statuses a provider answers when it
// cannot embed the content it was given (malformed, oversized or unsupported
// media). Auth, missing-model, quota and server errors are about the
// credential and stay retryable.
func isEmbedContentInputFault(statusCode int) bool {
	switch statusCode {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		return true
	}
	return false
}

// responseWithBody returns resp with body as its already-read, replayable Body.
func responseWithBody(resp *http.Response, body []byte) *http.Response {
	resp.Body = io.NopCloser(bytes.NewReader(body))
	resp.ContentLength = int64(len(body))
	return resp
}
