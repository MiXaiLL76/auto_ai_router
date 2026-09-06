package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
)

const nativeWSMaxResponses = 128
const nativeWSWriteTimeout = 10 * time.Second

type nativeWSRoutingKey struct{}
type nativeWSRouting struct {
	credential    string
	historyTokens int
}

func nativeWSRoutingFromContext(ctx context.Context) *nativeWSRouting {
	routing, _ := ctx.Value(nativeWSRoutingKey{}).(*nativeWSRouting)
	return routing
}

type nativeWSMessage struct {
	body []byte
	err  error
}
type nativeWSTurn struct {
	log         *RequestLogContext
	id          string
	body        []byte
	accumulator *completionTokenAccumulator
}

type nativeWSSession struct {
	proxy            *Proxy
	client, upstream *websocket.Conn
	request          *http.Request
	credential       *config.CredentialConfig
	model            string
	targetURL        string
	actualCredential string
	denylist         []string
	realModel        string
	active, pending  *nativeWSTurn
	waitingForTools  bool
	clientAborted    bool
	template         map[string]json.RawMessage
	completed        map[string]int
	turns            int
}

func readNativeWS(ctx context.Context, conn *websocket.Conn, messages chan<- nativeWSMessage) {
	for {
		kind, body, err := conn.ReadMessage()
		if err == nil && kind != websocket.TextMessage {
			err = errors.New("expected a text frame")
		}
		select {
		case messages <- nativeWSMessage{body, err}:
		case <-ctx.Done():
			return
		}
		if err != nil {
			return
		}
	}
}

func writeNativeWS(conn *websocket.Conn, body []byte) error {
	if err := conn.SetWriteDeadline(time.Now().Add(nativeWSWriteTimeout)); err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, body)
}

func (p *Proxy) handleNativeResponsesWebSocket(client *websocket.Conn, r *http.Request, first []byte) {
	ctx, cancel := context.WithTimeout(r.Context(), time.Hour)
	defer cancel()
	s := &nativeWSSession{proxy: p, client: client, request: r.WithContext(ctx), completed: make(map[string]int)}
	defer s.close()
	if !s.clientEvent(first) {
		return
	}
	if s.upstream == nil {
		return
	}
	clientMessages := make(chan nativeWSMessage, 1)
	upstreamMessages := make(chan nativeWSMessage, 1)
	go readNativeWS(ctx, client, clientMessages)
	go readNativeWS(ctx, s.upstream, upstreamMessages)
	var drainTimer *time.Timer
	var drainDone <-chan time.Time
	defer func() {
		if drainTimer != nil {
			drainTimer.Stop()
		}
	}()
	startDrain := func() {
		s.clientAborted = true
		clientMessages = nil
		if drainTimer == nil {
			drainTimer = time.NewTimer(streamDrainTimeout)
			drainDone = drainTimer.C
		}
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-drainDone:
			return
		case msg := <-clientMessages:
			if msg.err != nil {
				s.clientAborted = true
				if p.drainUpstreamOnAbort && (s.active != nil || s.pending != nil) {
					startDrain()
					continue
				}
				return
			}
			if !s.clientEvent(msg.body) {
				return
			}
		case msg := <-upstreamMessages:
			if msg.err != nil {
				s.sendError("upstream_disconnected", "Upstream connection closed; reconnect with full input history")
				return
			}
			if !s.upstreamEvent(msg.body) {
				return
			}
			if s.clientAborted {
				if !p.drainUpstreamOnAbort || (s.active == nil && s.pending == nil) {
					return
				}
				startDrain()
			}
		}
	}
}

func (s *nativeWSSession) sendError(code, message string) {
	_ = s.client.SetWriteDeadline(time.Now().Add(nativeWSWriteTimeout))
	sendWSError(s.client, code, message)
}

func (s *nativeWSSession) close() {
	if s.upstream != nil {
		_ = s.upstream.Close()
	}
	_ = s.client.Close()
	if s.active != nil {
		outcome := "stream_error"
		if s.clientAborted {
			outcome = "client_aborted"
		}
		s.finish(s.active, nil, outcome)
		s.active = nil
	}
	s.releasePending()
}

func (s *nativeWSSession) releasePending() {
	if s.pending != nil {
		s.proxy.reconcileBudgetAndRateLimits(s.pending.log, 0)
		s.pending = nil
	}
	s.waitingForTools = false
}

func (s *nativeWSSession) clientEvent(body []byte) bool {
	var event map[string]json.RawMessage
	if json.Unmarshal(body, &event) != nil || event == nil {
		s.sendError("invalid_request", "Invalid JSON object")
		return true
	}
	var eventType string
	_ = json.Unmarshal(event["type"], &eventType)
	switch eventType {
	case "response.create":
		return s.create(event)
	case "response.steer":
		return s.steer(event)
	default:
		s.sendError("invalid_request", "Unsupported WebSocket event")
		return true
	}
}

func (s *nativeWSSession) create(event map[string]json.RawMessage) bool {
	if s.active != nil || (s.pending != nil && !s.waitingForTools) {
		s.sendError("response_in_progress", "Wait for the current response or send response.steer")
		return true
	}
	var model, previous string
	_ = json.Unmarshal(event["model"], &model)
	_ = json.Unmarshal(event["previous_response_id"], &previous)
	if model == "" && s.model != "" {
		model = s.model
		event["model"], _ = json.Marshal(model)
	}
	if s.model != "" && model != s.model {
		s.sendError("invalid_request", "Reconnect to change models")
		return true
	}
	if previous != "" && !s.hasCompleted(previous) {
		s.sendError("previous_response_not_found", "Send full input history on a new connection")
		return true
	}
	historyTokens := s.completed[previous]
	resuming := s.pending != nil
	if resuming {
		if previous != s.pending.id {
			s.sendError("invalid_request", "Return tool results for the steered response")
			return true
		}
		historyTokens = max(historyTokens, s.pending.log.promptTokensEstimate())
		s.releasePending()
	}
	turn, wire, ok := s.prepare(event, historyTokens)
	if !ok {
		return s.upstream != nil && !resuming
	}
	if s.upstream == nil {
		if err := s.connect(turn.log); err != nil {
			turn.log.ErrorMsg = "Upstream WebSocket handshake failed"
			turn.log.promptTokensEstimateFn = nil
			s.finish(turn, nil, "stream_error")
			s.sendError("upstream_websocket_unavailable", "Upstream WebSocket handshake failed")
			return false
		}
	}
	s.model = model
	s.realModel = turn.log.RealModelID
	s.template = event
	s.active = turn
	wire["type"] = json.RawMessage(`"response.create"`)
	encoded, err := json.Marshal(wire)
	if err != nil {
		return false
	}
	return writeNativeWS(s.upstream, encoded) == nil
}

func (s *nativeWSSession) steer(event map[string]json.RawMessage) bool {
	if s.active == nil || s.active.id == "" || s.pending != nil {
		s.sendError("invalid_request", "Steering requires an active response and no queued steer")
		return true
	}
	var previous string
	_ = json.Unmarshal(event["previous_response_id"], &previous)
	if previous != s.active.id {
		s.sendError("invalid_request", "previous_response_id must identify the active response")
		return true
	}
	for key := range event {
		if key != "type" && key != "previous_response_id" && key != "input" {
			s.sendError("invalid_request", "Steering accepts only type, previous_response_id, and input")
			return true
		}
	}
	if len(event["input"]) == 0 {
		s.sendError("invalid_request", "Steering input is required")
		return true
	}
	request := make(map[string]json.RawMessage, len(s.template))
	for key, value := range s.template {
		request[key] = value
	}
	request["input"] = event["input"]
	delete(request, "previous_response_id")
	historyTokens := s.active.log.promptTokensEstimate() + s.proxy.estimateCompletionTokens(s.active.body)
	pending, _, ok := s.prepare(request, historyTokens)
	if !ok {
		return true
	}
	pending.id = previous
	s.pending = pending
	wire, err := json.Marshal(event)
	if err != nil {
		return false
	}
	return writeNativeWS(s.upstream, wire) == nil
}

func (s *nativeWSSession) prepare(event map[string]json.RawMessage, historyTokens int) (*nativeWSTurn, map[string]json.RawMessage, bool) {
	if s.turns >= nativeWSMaxResponses {
		s.sendError("websocket_limit", "Reconnect after 128 responses")
		return nil, nil, false
	}
	requestBody := make(map[string]json.RawMessage, len(event))
	for key, value := range event {
		requestBody[key] = value
	}
	delete(requestBody, "type")
	requestBody["stream"] = json.RawMessage(`true`)
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, nil, false
	}
	routing := &nativeWSRouting{historyTokens: historyTokens}
	if s.credential != nil {
		routing.credential = s.credential.Name
	}
	ctx := context.WithValue(s.request.Context(), nativeWSRoutingKey{}, routing)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "/v1/responses", bytes.NewReader(body))
	if err != nil {
		return nil, nil, false
	}
	req.Header = s.request.Header.Clone()
	req.Header.Set("Content-Type", "application/json")
	req.RemoteAddr = s.request.RemoteAddr
	marker := req.Header.Get(HeaderAIRProxyClient) == "1" || req.Header.Get(HeaderLegacyAIRProxyClient) == "1"
	req.Header.Del(HeaderAIRProxyClient)
	req.Header.Del(HeaderLegacyAIRProxyClient)
	req = captureCredentialDenylist(req, marker)
	logCtx := &RequestLogContext{RequestID: requestid.New(), EventID: requestid.New(), StartTime: time.Now(), Request: req, Status: "unknown", IsResponsesAPI: true}
	recorder := newCaptureResponseWriter()
	prepared, ok := s.proxy.orchestrateRequest(recorder, req, logCtx)
	if ok {
		logCtx.Request = prepared.request
		logCtx.RealModelID = prepared.realModelID
		if !prepared.passthroughResponses || (prepared.cred.Type != config.ProviderTypeOpenAI && !prepared.cred.IsProxyLike()) {
			WriteErrorBadRequest(recorder, "Native WebSocket requires an OpenAI-compatible Responses provider")
			ok = false
		} else if s.credential != nil && (prepared.cred.Name != s.credential.Name || prepared.cred.APIKey != s.credential.APIKey || prepared.cred.BaseURL != s.credential.BaseURL || prepared.cred.Type != s.credential.Type || prepared.realModelID != s.realModel || !slices.Equal(effectiveCredentialDenylist(prepared.request.Context()), s.denylist)) {
			WriteErrorBadRequest(recorder, "Routing policy changed; reconnect")
			ok = false
		} else {
			ok = s.proxy.enforceBudgetAndRateLimits(recorder, prepared.request, logCtx, prepared.modelID, prepared.realModelID, prepared.body)
		}
	}
	if !ok {
		s.proxy.reconcileBudgetAndRateLimits(logCtx, 0)
		_ = s.client.SetWriteDeadline(time.Now().Add(nativeWSWriteTimeout))
		sendWSHTTPError(s.client, recorder.body.Bytes(), recorder.statusCode)
		return nil, nil, false
	}
	wireBody := prepared.body
	if prepared.cred.IsProxyLike() {
		wireBody = prepared.proxyBody
	}
	var wire map[string]json.RawMessage
	if json.Unmarshal(wireBody, &wire) != nil {
		s.proxy.reconcileBudgetAndRateLimits(logCtx, 0)
		return nil, nil, false
	}
	delete(wire, "stream")
	delete(wire, "background")
	wire["store"] = json.RawMessage(`false`)
	logCtx.TargetURL = s.targetURL
	logCtx.WebSearchRequested, logCtx.WebSearchContextSize = extractWebSearchRequestUsage(prepared.body, "application/json")
	s.proxy.setPromptTokensEstimate(logCtx, prepared.body, prepared.realModelID)
	logCtx.ActualCredentialName = s.actualCredential
	s.turns++
	return &nativeWSTurn{log: logCtx, body: prepared.body, accumulator: s.proxy.newCompletionTokenAccumulator(prepared.realModelID)}, wire, true
}

func nativeWebSocketURL(baseURL string) (string, error) {
	target, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}
	switch target.Scheme {
	case "https":
		target.Scheme = "wss"
	case "http":
		target.Scheme = "ws"
	default:
		return "", errors.New("expected http or https upstream URL")
	}
	if target.Host == "" || target.User != nil {
		return "", errors.New("invalid upstream URL")
	}
	basePath := strings.TrimRight(target.Path, "/")
	if extractVersionSuffix(basePath) != "" {
		target.Path = basePath + "/responses"
	} else {
		target.Path = basePath + "/v1/responses"
	}
	return target.String(), nil
}

func (s *nativeWSSession) connect(logCtx *RequestLogContext) error {
	cred := logCtx.Credential
	target, err := nativeWebSocketURL(cred.BaseURL)
	if err != nil {
		return err
	}
	headers := make(http.Header)
	// Handshake headers are rebuilt: client authentication and WebSocket keys never reach the provider.
	for _, key := range []string{"OpenAI-Beta", "OpenAI-Organization", "OpenAI-Project"} {
		if value := s.request.Header.Get(key); value != "" {
			headers.Set(key, value)
		}
	}
	headers.Set("Authorization", "Bearer "+cred.APIKey)
	if cred.IsProxyLike() {
		headers.Set(HeaderAIRProxyClient, "1")
		headers.Set(HeaderLegacyAIRProxyClient, "1")
		if carriesCredentialDenylist(cred) {
			if err := setCredentialDenylistHeader(headers, effectiveCredentialDenylist(logCtx.Request.Context())); err != nil {
				return err
			}
		}
	}
	dialer := websocket.Dialer{HandshakeTimeout: 30 * time.Second, Proxy: http.ProxyFromEnvironment}
	conn, response, err := dialer.DialContext(s.request.Context(), target, headers)
	if response != nil && response.Body != nil {
		_ = response.Body.Close()
	}
	if err != nil {
		return err
	}
	limit := int64(s.proxy.maxBodySizeMB) * 1024 * 1024
	if limit <= 0 {
		limit = 1024 * 1024
	}
	conn.SetReadLimit(limit)
	s.upstream = conn
	snapshot := *cred
	s.credential = &snapshot
	s.denylist = slices.Clone(effectiveCredentialDenylist(logCtx.Request.Context()))
	if cred.IsProxyLike() && response != nil {
		s.actualCredential = response.Header.Get("X-Credential-Name")
		logCtx.ActualCredentialName = s.actualCredential
	}
	s.targetURL = target
	return nil
}

func (s *nativeWSSession) upstreamEvent(body []byte) bool {
	var event struct {
		Type       string `json:"type"`
		ResponseID string `json:"response_id"`
		Response   struct {
			ID     string          `json:"id"`
			Model  string          `json:"model"`
			Output json.RawMessage `json:"output"`
		} `json:"response"`
	}
	if json.Unmarshal(body, &event) != nil {
		s.sendError("upstream_protocol_error", "Invalid upstream event")
		return false
	}
	id := event.Response.ID
	if id == "" {
		id = event.ResponseID
	}
	if id != "" && s.hasCompleted(id) {
		return true
	}
	if event.Type == "response.created" {
		if s.active == nil && s.pending != nil {
			s.active = s.pending
			s.active.id = ""
			s.pending = nil
			s.waitingForTools = false
		}
		if s.active == nil || s.active.id != "" || id == "" {
			s.sendError("upstream_protocol_error", "Unexpected response.created")
			return false
		}
		s.active.id = id
		s.active.log.ClientResponseID = id
	}
	if event.Type == "response.steer.pending" {
		s.waitingForTools = true
	}
	if event.Type == "response.steer.failed" {
		s.releasePending()
	}
	if s.active != nil {
		if id != "" && s.active.id != "" && id != s.active.id {
			s.sendError("upstream_protocol_error", "Unexpected response ID")
			return false
		}
		chunk := append(append([]byte("data: "), body...), '\n', '\n')
		s.active.accumulator.AddChunk(chunk)
		if strings.HasSuffix(event.Type, ".delta") && s.active.log.CompletionStartTime.IsZero() {
			s.active.log.CompletionStartTime = time.Now()
		}
		if event.Response.Model != "" {
			body = openai.ReplaceModelInBody(body, event.Response.Model, s.active.log.PublicModelID)
		}
	}
	original := body
	body, _ = promanutils.SanitizeUpstreamJSONBody(body, s.model)
	delivered := !s.clientAborted && writeNativeWS(s.client, body) == nil
	if !delivered {
		s.clientAborted = true
	}
	switch event.Type {
	case "response.completed", "response.done", "response.incomplete", "response.failed":
		if s.active == nil || id == "" {
			return false
		}
		s.active.log.RequestCompleted = delivered
		s.finish(s.active, original, "completed")
		s.completed[id] = s.active.log.TokenUsage.PromptTokens + s.active.log.TokenUsage.CompletionTokens
		if s.pending != nil {
			var output []struct {
				Type  string `json:"type"`
				Async bool   `json:"async"`
			}
			_ = json.Unmarshal(event.Response.Output, &output)
			for _, item := range output {
				if !item.Async && (item.Type == "function_call" || item.Type == "custom_tool_call" || item.Type == "mcp_approval_request") {
					s.waitingForTools = true
				}
			}
		}
		s.active = nil
	case "error", "response.error":
		if s.active != nil {
			s.finish(s.active, original, "stream_error")
			s.active = nil
		}
		return false
	}
	return delivered || (s.clientAborted && s.proxy.drainUpstreamOnAbort)
}

func (s *nativeWSSession) finish(turn *nativeWSTurn, event []byte, outcome string) {
	if turn.log.Logged {
		return
	}
	if s.clientAborted && outcome == "completed" {
		outcome = "client_aborted"
	}
	turn.log.StreamOutcome = outcome
	turn.log.TargetURL = s.targetURL
	status := http.StatusOK
	if outcome == "stream_error" {
		status = http.StatusBadGateway
	}
	chunk := append(append([]byte("data: "), event...), '\n', '\n')
	s.proxy.finalizeStreamingLog(turn.log, turn.accumulator.TokenCount(), chunk, "openai", status, false)
	if turn.log.Credential != nil && turn.log.TokenUsage != nil {
		tokens := turn.log.TokenUsage.PromptTokens + turn.log.TokenUsage.CompletionTokens
		s.proxy.rateLimiter.ConsumeTokens(turn.log.Credential.Name, tokens)
		s.proxy.rateLimiter.ConsumeModelTokens(turn.log.Credential.Name, turn.log.ModelID, tokens)
	}
	s.proxy.reconcileBudgetAndRateLimits(turn.log, 0)
}

func (s *nativeWSSession) hasCompleted(id string) bool {
	_, ok := s.completed[id]
	return ok
}
