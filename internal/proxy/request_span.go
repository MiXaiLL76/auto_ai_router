package proxy

import (
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const tracerName = "github.com/mixaill76/auto_ai_router/internal/proxy"

// innerRequestSpan gives a request made inside another (a vision describe, a
// WebSocket turn) a span of its own. The function returned ends it.
func innerRequestSpan(r *http.Request) (*http.Request, func(status int)) {
	outer := trace.SpanFromContext(r.Context())
	ctx, span := outer.TracerProvider().Tracer(tracerName).Start(r.Context(), r.Method+" "+r.URL.Path)
	return r.WithContext(ctx), func(status int) {
		if status > 0 {
			span.SetAttributes(attribute.Int("http.response.status_code", status))
		}
		if status >= http.StatusInternalServerError {
			span.SetStatus(codes.Error, "")
		}
		span.End()
	}
}

// The moments of a request before its first upstream send, as span events.
const (
	markAuthenticated   = "authenticated"       // the key looked up and checked
	markBodyRead        = "body_read"           // the client has sent the request's body
	markRouted          = "credential_selected" // the body parsed, the model resolved, a credential chosen
	markRequestPrepared = "request_prepared"    // the body converted for the credential's provider
	markAdmitted        = "admitted"            // the budget and the rate limits checked
	markFirstTokenEvent = "first_token"
)

// mark names a moment of the request on its span.
func (logCtx *RequestLogContext) mark(name string) {
	if logCtx == nil {
		return
	}
	if span := trace.SpanFromContext(logCtx.Context()); span.IsRecording() {
		span.AddEvent(name)
	}
}

// markFirstToken stamps the first token's arrival and marks it on the span.
func (logCtx *RequestLogContext) markFirstToken() {
	logCtx.CompletionStartTime = time.Now()
	logCtx.mark(markFirstTokenEvent)
}

// finalStatus is how the request ended: the handlers' status, else the
// response code's.
func (logCtx *RequestLogContext) finalStatus() string {
	if logCtx.Status != "" {
		return logCtx.Status
	}
	if logCtx.HTTPStatus >= 400 {
		return "failure"
	}
	return "success"
}

// annotateRequestSpan completes the request's span with what its spend row
// says, under the same names: whose it was, what served it, how it ended.
func annotateRequestSpan(logCtx *RequestLogContext) {
	span := trace.SpanFromContext(logCtx.Context())
	if !span.IsRecording() {
		return
	}
	attrs := make([]attribute.KeyValue, 0, 13)
	add := func(key, value string) {
		if value != "" {
			attrs = append(attrs, attribute.String(key, value))
		}
	}

	if logCtx.Token != "" {
		add("aar.api_key_hash", litellmdb.HashToken(logCtx.Token))
	}
	if info := logCtx.TokenInfo; info != nil {
		add("aar.key_alias", info.KeyAlias)
		add("aar.user_id", info.UserID)
		add("aar.team_id", info.TeamID)
		add("aar.team_alias", info.TeamAlias)
		add("aar.organization_id", info.OrganizationID)
		add("aar.organization_alias", info.OrganizationAlias)
	}

	// The spend row's stand-in for "no credential" is not a credential.
	attempts := httputil.AttemptsFrom(logCtx.Context()).Count()
	attrs = append(attrs, attribute.Int("aar.upstream_attempts", attempts))
	if cred := logCtx.Credential; cred != nil && attempts > 0 {
		add("aar.credential", cred.Name)
		add("aar.provider", string(cred.Type))
	}
	add("aar.real_model", logCtx.RealModelID)

	// "unknown" says nothing: the span's HTTP status does.
	switch status := logCtx.finalStatus(); status {
	case "success":
		add("aar.status", status)
	case "failure":
		add("aar.status", status)
		add("error.type", mapHTTPStatusToErrorClass(logCtx.HTTPStatus))
	}
	span.SetAttributes(attrs...)
}
