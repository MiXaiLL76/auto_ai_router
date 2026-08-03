package litellm

import (
	"io"
	"net/http"
)

type Context struct {
	Endpoint       string
	RequestedModel string
	RequestID      string
	IncludeUsage   bool
}

type Response struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

type Transformer struct{}

func New() *Transformer {
	return &Transformer{}
}

func (t *Transformer) Stream(ctx Context, reader io.Reader) io.Reader {
	return newStreamReader(ctx, reader)
}
