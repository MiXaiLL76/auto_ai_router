package httputil

import (
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// lifetimeTransport bounds how long a provider connection takes requests
// (server.max_connection_lifetime).
//
// net/http keeps a pooled connection for as long as requests keep reusing it:
// IdleConnTimeout only closes a connection nobody used for that long, and the
// pool hands out the most recently used connection first, so under steady load
// a connection lives until something cuts it. A proxy on the way can then never
// drain its tunnels, so restarting it cuts whatever request is on them.
// net/http has no maximum connection lifetime and no hook to keep one
// connection out of the pool, so lifetimeTransport replaces the whole transport.
type lifetimeTransport struct {
	lifetime time.Duration
	build    func() *http.Transport
	now      func() time.Time

	mu      sync.Mutex
	current *generation
}

type generation struct {
	transport *http.Transport
	retireAt  time.Time
	retired   atomic.Bool
}

func newLifetimeTransport(lifetime time.Duration, build func() *http.Transport) *lifetimeTransport {
	return &lifetimeTransport{lifetime: lifetime, build: build, now: time.Now}
}

func (t *lifetimeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	g := t.generation()
	resp, err := g.transport.RoundTrip(req)
	if err != nil {
		g.requestEnded()
		return nil, err
	}
	// An upgraded connection (101) never returns to the pool, and its body must
	// stay the io.ReadWriteCloser it is.
	if resp.StatusCode != http.StatusSwitchingProtocols {
		resp.Body = &generationBody{ReadCloser: resp.Body, g: g}
	}
	return resp, nil
}

// generation returns the current generation, starting a new one when there is
// none or the current one has lived its lifetime.
func (t *lifetimeTransport) generation() *generation {
	now := t.now()
	t.mu.Lock()
	old := t.current
	if old != nil && now.Before(old.retireAt) {
		t.mu.Unlock()
		return old
	}
	g := &generation{transport: t.build(), retireAt: now.Add(t.lifetime)}
	t.current = g
	t.mu.Unlock()

	if old != nil {
		old.retired.Store(true)
		old.closeIdle()
	}
	return g
}

// CloseIdleConnections closes the current generation's idle connections; a
// retired generation closes its own.
func (t *lifetimeTransport) CloseIdleConnections() {
	t.mu.Lock()
	g := t.current
	t.mu.Unlock()
	if g != nil {
		g.transport.CloseIdleConnections()
	}
}

func (g *generation) requestEnded() {
	if g.retired.Load() {
		g.closeIdle()
	}
}

func (g *generation) closeIdle() {
	g.transport.CloseIdleConnections()
	// an HTTP/2 stream leaves its connection a moment after its body ends.
	time.AfterFunc(time.Second, g.transport.CloseIdleConnections)
}

// generationBody is a response body that tells its generation when the request
// ends: at EOF or a read error, or at Close, whichever comes first
type generationBody struct {
	io.ReadCloser
	g     *generation
	ended sync.Once
}

func (b *generationBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil {
		b.ended.Do(b.g.requestEnded)
	}
	return n, err
}

func (b *generationBody) Close() error {
	err := b.ReadCloser.Close()
	b.ended.Do(b.g.requestEnded)
	return err
}
