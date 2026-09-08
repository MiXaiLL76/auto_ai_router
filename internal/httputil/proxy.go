package httputil

import (
	"context"
	"net/http"
	"net/url"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

type proxyURLContextKey struct{}

type proxyTransport struct {
	base     http.RoundTripper
	proxyURL string
}

func (t proxyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.base.RoundTrip(req.Clone(WithProxyURL(req.Context(), t.proxyURL)))
}

func WithClientProxy(client *http.Client, proxyURL string) *http.Client {
	cloned := *client
	cloned.Transport = proxyTransport{base: client.Transport, proxyURL: proxyURL}
	return &cloned
}

func WithProxyURL(ctx context.Context, proxyURL string) context.Context {
	return context.WithValue(ctx, proxyURLContextKey{}, proxyURL)
}

func proxyFromRequest(req *http.Request) (*url.URL, error) {
	if raw, _ := req.Context().Value(proxyURLContextKey{}).(string); raw != "" {
		return config.ParseProxyURL(raw)
	}
	return http.ProxyFromEnvironment(req)
}
