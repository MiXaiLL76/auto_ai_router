package router

import (
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
)

// inFlightLimit caps the API requests served at once (server.max_in_flight_requests); 0 is no cap.
type inFlightLimit struct {
	max     int64
	current atomic.Int64
}

// acquire takes a slot, or reports false at the cap.
func (l *inFlightLimit) acquire() bool {
	if l.max <= 0 {
		return true
	}
	if l.current.Add(1) > l.max {
		l.current.Add(-1)
		return false
	}
	return true
}

func (l *inFlightLimit) release() {
	if l.max > 0 {
		l.current.Add(-1)
	}
}

// isAPIPath reports whether the cap counts the request: metrics and internal endpoints are outside /v1/.
func isAPIPath(path string) bool {
	return strings.HasPrefix(path, "/v1/")
}

// writeShed answers a request over the cap without reading its body.
func writeShed(w http.ResponseWriter) {
	monitoring.RequestsShedTotal.Inc()
	w.Header().Set("Retry-After", "1")
	proxy.WriteErrorOverloaded(w, "Too many requests in flight, retry later")
}
