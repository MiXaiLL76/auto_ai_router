package proxy

import (
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// requestScope is the per-request identity and logging state established
// before any credential is chosen: the ids, the AIR marker handling and the
// RequestLogContext the whole request logs through.
type requestScope struct {
	Start     time.Time
	RequestID string
	EventID   string
	LogCtx    *RequestLogContext
}

// newRequestScope mints the ids, strips the internal AIR markers and builds
// the RequestLogContext. It returns the request it modified: the markers are
// removed from its headers and the denylist/attempts contexts are attached.
func (p *Proxy) newRequestScope(w http.ResponseWriter, r *http.Request) (*requestScope, *http.Request) {
	start := utils.NowUTC()
	// Reuse the ID assigned by requestid.Middleware (also used as the OTel
	// trace_id, see internal/telemetry's IDGenerator) so request_id, trace_id,
	// and the X-Request-Id header returned to the client are all identical.
	// Falls back to minting one when the middleware didn't run (e.g. tests
	// calling this handler directly).
	requestID := requestid.FromContext(r.Context())
	if requestID == "" {
		requestID = requestid.New()
	}
	// Normally requestid.Middleware already set this header to the same value;
	// re-set it so the fallback-minted ID (middleware didn't run) still reaches
	// the client and stays consistent with what we log below.
	w.Header().Set(requestid.Header, requestID)
	if info := responseCompatRequestFromContext(r.Context()); info != nil {
		info.RequestID = requestID
	}
	// Minted fresh regardless of requestID's origin: when otel trusts an
	// inbound traceparent, requestID can be client-supplied and shared across
	// distinct requests (retry replay, or a malicious caller), which would
	// otherwise defeat AirEventID's job as the spend-log collision fallback.
	eventID := requestid.New()

	// Save and strip internal proxy markers before normal request handling. Their
	// value is trusted only after authentication proves this is a master-key AIR peer.
	proxyMarkerPresent := r.Header.Get(HeaderAIRProxyClient) == "1" ||
		r.Header.Get(HeaderLegacyAIRProxyClient) == "1"
	r.Header.Del(HeaderAIRProxyClient)
	r.Header.Del(HeaderLegacyAIRProxyClient)
	r = captureCredentialDenylist(r, proxyMarkerPresent)
	// The request's upstream attempts, for its trace and its spend event.
	r = r.WithContext(httputil.WithAttempts(r.Context()))

	// Create logging context that will be filled throughout request processing
	// and logged at the end via defer to ensure all requests are logged
	logCtx := &RequestLogContext{
		RequestID:      requestID,
		EventID:        eventID,
		StartTime:      start,
		Request:        r,
		Status:         "unknown",
		IsProxyRequest: false,
	}
	// Arms w, if it is the LiteLLM response-compatibility writer, to re-add
	// the router's own cost after that writer's transform has already run --
	// see wireCostInjectionForCompatWriter. Safe to call unconditionally and
	// this early: whether to actually act is re-checked lazily against
	// logCtx.OrganizationPolicy, which isn't resolved yet at this point.
	wireCostInjectionForCompatWriter(w, p, logCtx)
	return &requestScope{
		Start:     start,
		RequestID: requestID,
		EventID:   eventID,
		LogCtx:    logCtx,
	}, r
}
