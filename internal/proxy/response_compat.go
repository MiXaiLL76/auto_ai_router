package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"

	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
)

type responseCompatContextKey struct{}

type responseCompatRequest struct {
	RequestID      string
	RequestedModel string
	IncludeUsage   bool
}

func withResponseCompatRequest(r *http.Request) (*http.Request, *responseCompatRequest) {
	info := &responseCompatRequest{}
	return r.WithContext(context.WithValue(r.Context(), responseCompatContextKey{}, info)), info
}

func responseCompatRequestFromContext(ctx context.Context) *responseCompatRequest {
	info, _ := ctx.Value(responseCompatContextKey{}).(*responseCompatRequest)
	return info
}

func clientRequestedStreamUsage(body []byte) bool {
	var request map[string]any
	if json.Unmarshal(body, &request) != nil {
		return false
	}
	options, _ := request["stream_options"].(map[string]any)
	includeUsage, _ := options["include_usage"].(bool)
	return includeUsage
}

type responseCompatibilityWriter struct {
	target      http.ResponseWriter
	transformer *compatlitellm.Transformer
	request     *http.Request
	header      http.Header
	body        bytes.Buffer
	statusCode  int
	stream      bool
	streamPipe  *io.PipeWriter
	streamDone  chan error
	closeOnce   sync.Once
	closeErr    error
}

func newResponseCompatibilityWriter(
	target http.ResponseWriter,
	transformer *compatlitellm.Transformer,
	request *http.Request,
) *responseCompatibilityWriter {
	return &responseCompatibilityWriter{
		target:      target,
		transformer: transformer,
		request:     request,
		header:      target.Header().Clone(),
	}
}

func (w *responseCompatibilityWriter) Header() http.Header {
	return w.header
}

func (w *responseCompatibilityWriter) WriteHeader(statusCode int) {
	if w.statusCode != 0 {
		return
	}
	w.statusCode = statusCode
	contentType := strings.ToLower(w.header.Get("Content-Type"))
	if statusCode < http.StatusBadRequest && strings.Contains(contentType, "text/event-stream") {
		w.startStream()
	}
}

func (w *responseCompatibilityWriter) Write(body []byte) (int, error) {
	if w.statusCode == 0 {
		w.WriteHeader(http.StatusOK)
	}
	if w.stream {
		return w.streamPipe.Write(body)
	}
	return w.body.Write(body)
}

func (w *responseCompatibilityWriter) Flush() {
	_ = w.FlushError()
}

func (w *responseCompatibilityWriter) FlushError() error {
	return nil
}

func (w *responseCompatibilityWriter) Unwrap() http.ResponseWriter {
	return w.target
}

func (w *responseCompatibilityWriter) Close() error {
	w.closeOnce.Do(func() {
		if w.statusCode == 0 {
			w.statusCode = http.StatusOK
		}
		if w.stream {
			w.closeErr = w.streamPipe.Close()
			if streamErr := <-w.streamDone; w.closeErr == nil {
				w.closeErr = streamErr
			}
			return
		}

		result := w.transformer.Transform(w.compatContext(), compatlitellm.Response{
			StatusCode: w.statusCode,
			Headers:    w.header,
			Body:       w.body.Bytes(),
		})
		copyHeaders(w.target.Header(), result.Headers)
		w.target.Header().Set("Content-Length", itoa(len(result.Body)))
		w.target.WriteHeader(result.StatusCode)
		if len(result.Body) > 0 {
			_, w.closeErr = w.target.Write(result.Body)
		}
	})
	return w.closeErr
}

func (w *responseCompatibilityWriter) startStream() {
	w.stream = true
	headers := w.transformer.TransformHeaders(w.compatContext(), w.header)
	copyHeaders(w.target.Header(), headers)
	w.target.WriteHeader(w.statusCode)

	reader, writer := io.Pipe()
	w.streamPipe = writer
	w.streamDone = make(chan error, 1)
	go func() {
		stream := w.transformer.Stream(w.compatContext(), reader)
		_, err := io.Copy(flushingWriter{target: w.target}, stream)
		_ = reader.CloseWithError(err)
		w.streamDone <- err
	}()
}

func (w *responseCompatibilityWriter) compatContext() compatlitellm.Context {
	ctx := compatlitellm.Context{Endpoint: w.request.URL.Path}
	if info := responseCompatRequestFromContext(w.request.Context()); info != nil {
		ctx.RequestID = info.RequestID
		ctx.RequestedModel = info.RequestedModel
		ctx.IncludeUsage = info.IncludeUsage
	}
	return ctx
}

func copyHeaders(target, source http.Header) {
	for key := range target {
		target.Del(key)
	}
	for key, values := range source {
		target[key] = append([]string(nil), values...)
	}
}

type flushingWriter struct {
	target http.ResponseWriter
}

func (w flushingWriter) Write(body []byte) (int, error) {
	written, err := w.target.Write(body)
	if err == nil {
		if flusher, ok := w.target.(http.Flusher); ok {
			flusher.Flush()
		}
	}
	return written, err
}
