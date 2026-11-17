// Package toolhttp caches the HTTP clients the router's own tool calls go
// out through.
//
// The router calls a provider's search API on the model's behalf, and those
// calls are made again and again: once per request that asks for a search.
// A client built per call would open a fresh TCP and TLS connection every
// time and throw the connection away afterwards, so every tool-using request
// would pay a handshake the previous request had already paid.
//
// The cache keeps one client per distinct destination (the base URL, the
// proxy and the timeout): the connections survive between the requests, and
// the credential's key -- which differs per credential -- never enters the
// cache key, because the key is a header, not a connection.
package toolhttp

import (
	"net/http"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
)

// Cache hands out one *http.Client per distinct destination.
type Cache struct {
	mu      sync.Mutex
	clients map[string]*http.Client
}

// NewCache builds an empty cache.
func NewCache() *Cache {
	return &Cache{clients: make(map[string]*http.Client)}
}

// Client returns the client for the given destination, building it on the
// first request.
//
// The timeout bounds one call, so it is part of the key: two destinations
// that differ only in their timeout need different clients.
func (c *Cache) Client(baseURL, proxyURL string, timeout time.Duration) *http.Client {
	key := baseURL + "\x00" + proxyURL + "\x00" + timeout.String()

	c.mu.Lock()
	defer c.mu.Unlock()
	if client, ok := c.clients[key]; ok {
		return client
	}
	httpCfg := httputil.DefaultHTTPClientConfig()
	httpCfg.Timeout = timeout
	client := httputil.NewHTTPClient(httpCfg)
	if proxyURL != "" {
		client = httputil.WithClientProxy(client, proxyURL)
	}
	c.clients[key] = client
	return client
}
