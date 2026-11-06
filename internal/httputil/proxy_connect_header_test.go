package httputil

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/baggage"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// The CONNECT that opens a tunnel carries the trace context of the client
// span of the request it is opened for, and nothing else of the request's
// context.
func TestProxyConnectHeader_CarriesTheClientSpan(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()
	proxy := newConnectProxy(t)
	transport := tlsBuild(srv.Certificate(), false)()
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: otelhttp.NewTransport(transport)}

	ctx, root := provider.Tracer("test").Start(context.Background(), "request")
	member, err := baggage.NewMember("tenant", "secret")
	require.NoError(t, err)
	bag, err := baggage.New(member)
	require.NoError(t, err)
	ctx = baggage.ContextWithBaggage(ctx, bag)
	ctx = WithProxyURL(ctx, "http://"+proxy.listener.Addr().String())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, http.NoBody)
	require.NoError(t, err)
	resp, err := client.Do(req)
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	root.End()

	var clientSpan sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.SpanKind() == trace.SpanKindClient {
			clientSpan = span
		}
	}
	require.NotNil(t, clientSpan, "otelhttp's client span")
	require.Equal(t, root.SpanContext().TraceID(), clientSpan.SpanContext().TraceID())

	connects := proxy.connectHeaders()
	require.Len(t, connects, 1)
	sc := clientSpan.SpanContext()
	assert.Equal(t, "00-"+sc.TraceID().String()+"-"+sc.SpanID().String()+"-01", connects[0].Get("Traceparent"),
		"the tunnel's span would be a child of the client span that opened it")
	assert.Empty(t, connects[0].Get("Baggage"), "baggage stays inside the request")
}

func TestProxyConnectHeader_NothingWithoutASpan(t *testing.T) {
	header, err := proxyConnectHeader(context.Background(), nil, "")
	require.NoError(t, err)
	assert.Empty(t, header)
}
