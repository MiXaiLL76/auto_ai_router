package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"sync"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

var wsUpgrader = websocket.Upgrader{
	CheckOrigin:     func(r *http.Request) bool { return true },
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
}

// wsSSEWriter implements http.ResponseWriter + http.Flusher.
// It intercepts SSE-formatted output from ProxyRequest and forwards each event
// as a plain JSON WebSocket text message. When the SSE stream sends "[DONE]",
// the writer forwards it and signals completion via the done channel.
type wsSSEWriter struct {
	conn    *websocket.Conn
	writeMu sync.Mutex // guards WebSocket writes
	header  http.Header
	status  int
	bufMu   sync.Mutex // guards buf
	buf     []byte

	done     chan struct{}
	doneOnce sync.Once

	finalResp *responses.Response
	isFailed  bool
}

func newWSSSEWriter(conn *websocket.Conn) *wsSSEWriter {
	return &wsSSEWriter{
		conn:   conn,
		header: make(http.Header),
		done:   make(chan struct{}),
	}
}

func (w *wsSSEWriter) Header() http.Header  { return w.header }
func (w *wsSSEWriter) WriteHeader(code int) { w.status = code }
func (w *wsSSEWriter) Flush()               {} // data is pushed on every Write

func (w *wsSSEWriter) Write(data []byte) (int, error) {
	w.bufMu.Lock()
	w.buf = append(w.buf, data...)
	w.parseSSEEvents()
	w.bufMu.Unlock()
	return len(data), nil
}

func (w *wsSSEWriter) closeDone() {
	w.doneOnce.Do(func() { close(w.done) })
}

// parseSSEEvents must be called with w.bufMu held.
func (w *wsSSEWriter) parseSSEEvents() {
	for {
		idx := bytes.Index(w.buf, []byte("\n\n"))
		if idx < 0 {
			return
		}
		chunk := w.buf[:idx]
		w.buf = w.buf[idx+2:]

		var eventType, eventData string
		for _, line := range bytes.Split(chunk, []byte("\n")) {
			if bytes.HasPrefix(line, []byte("event: ")) {
				eventType = string(bytes.TrimPrefix(line, []byte("event: ")))
			} else if bytes.HasPrefix(line, []byte("data: ")) {
				eventData = string(bytes.TrimPrefix(line, []byte("data: ")))
			}
		}

		// SSE end sentinel — close turn WITHOUT sending [DONE] to WS.
		// The terminal event (response.completed etc.) already advanced the turn.
		if eventData == "[DONE]" {
			w.closeDone()
			return
		}

		if eventData == "" {
			continue
		}

		// Fall back to JSON "type" field when event: prefix is absent
		effectiveType := eventType
		if effectiveType == "" {
			var typeOnly struct {
				Type string `json:"type"`
			}
			if json.Unmarshal([]byte(eventData), &typeOnly) == nil {
				effectiveType = typeOnly.Type
			}
		}

		// Track final response from terminal response events
		if effectiveType == "response.completed" || effectiveType == "response.done" {
			var doneEvt struct {
				Response json.RawMessage `json:"response"`
			}
			if json.Unmarshal([]byte(eventData), &doneEvt) == nil && len(doneEvt.Response) > 0 {
				var resp responses.Response
				if json.Unmarshal(doneEvt.Response, &resp) == nil {
					w.finalResp = &resp
					if resp.Status == "failed" {
						w.isFailed = true
					}
				}
			}
		}

		if effectiveType == "error" || effectiveType == "response.error" {
			w.isFailed = true
		}
		if effectiveType == "response.failed" {
			w.isFailed = true
		}

		w.writeMu.Lock()
		_ = w.conn.WriteMessage(websocket.TextMessage, []byte(eventData))
		w.writeMu.Unlock()

		// Terminal events close the turn (no separate [DONE] needed)
		switch effectiveType {
		case "response.completed", "response.done",
			"response.failed", "response.incomplete",
			"error", "response.error":
			w.closeDone()
			return
		}
	}
}

// sendWSError sends a structured error event to the WebSocket client.
func sendWSError(conn *websocket.Conn, code, message string) {
	evt := map[string]interface{}{
		"type":            "error",
		"sequence_number": 0,
		"error": map[string]interface{}{
			"code":    code,
			"message": message,
			"type":    "server_error",
			"param":   nil,
		},
	}
	if b, err := json.Marshal(evt); err == nil {
		_ = conn.WriteMessage(websocket.TextMessage, b)
	}
	// Do NOT send [DONE] — the error event itself signals turn completion
}

func sendWSHTTPError(conn *websocket.Conn, raw []byte, fallbackStatus int) {
	code := "api_error"
	message := http.StatusText(fallbackStatus)
	if message == "" {
		message = "Request failed"
	}
	errType := "server_error"
	var param interface{}

	var body struct {
		Error struct {
			Message string      `json:"message"`
			Type    string      `json:"type"`
			Param   interface{} `json:"param"`
			Code    string      `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &body) == nil {
		if body.Error.Message != "" {
			message = body.Error.Message
		}
		if body.Error.Type != "" {
			errType = body.Error.Type
		}
		if body.Error.Param != nil {
			param = body.Error.Param
		}
		if body.Error.Code != "" {
			code = body.Error.Code
		}
	}

	evt := map[string]interface{}{
		"type":            "error",
		"sequence_number": 0,
		"error": map[string]interface{}{
			"code":    code,
			"message": message,
			"type":    errType,
			"param":   param,
		},
	}
	if b, err := json.Marshal(evt); err == nil {
		_ = conn.WriteMessage(websocket.TextMessage, b)
	}
}

// HandleWebSocketResponses handles WebSocket connections on /v1/responses.
//
// Protocol:
//  1. Client sends {"type":"response.create", model, input, ...}
//  2. Server streams each SSE event as a plain JSON text message.
//  3. Server sends "[DONE]" when the turn is complete.
//  4. Client may send another response.create on the same connection.
//
// For store:false responses, completed responses are cached in connection-local
// memory so that previous_response_id continuations work within the same session.
// Reconnecting on a new WebSocket clears the cache, triggering previous_response_not_found.
func (p *Proxy) HandleWebSocketResponses(w http.ResponseWriter, r *http.Request) {
	// Authenticate the upgrade request BEFORE establishing the socket. An invalid
	// key / exhausted budget gets an ordinary HTTP error (401/402/403) and the
	// connection is never upgraded. The per-message response.create path still runs
	// its own full auth via ProxyRequest (Authorization is cloned into internalReq),
	// so this pre-check is purely a gate on connection establishment.
	authLogCtx := &RequestLogContext{
		RequestID: uuid.New().String(),
		StartTime: utils.NowUTC(),
		Request:   r,
		Status:    "unknown",
	}
	if !p.authenticateRequest(w, r, authLogCtx, p.isLiteLLMHealthy()) {
		return
	}

	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		p.logger.DebugContext(r.Context(), "ws: upgrade failed", "error", err)
		return
	}
	defer func() {
		if closeErr := conn.Close(); closeErr != nil && p.logger != nil {
			p.logger.DebugContext(r.Context(), "ws: close failed", "error", closeErr)
		}
	}()

	// Connection-local cache: response ID → completed Response
	localCache := make(map[string]*responses.Response)
	var cacheMu sync.Mutex

outerLoop:
	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			if !websocket.IsCloseError(err,
				websocket.CloseNormalClosure,
				websocket.CloseGoingAway,
				websocket.CloseNoStatusReceived,
			) {
				p.logger.DebugContext(r.Context(), "ws: read error", "error", err)
			}
			return
		}

		var reqMap map[string]interface{}
		if err := json.Unmarshal(msg, &reqMap); err != nil {
			p.logger.WarnContext(r.Context(), "ws: invalid JSON in client message", "error", err)
			sendWSError(conn, "invalid_request", "Invalid JSON")
			continue
		}

		msgType, _ := reqMap["type"].(string)
		if msgType != "response.create" {
			p.logger.WarnContext(r.Context(), "ws: unexpected message type from client", "type", msgType)
			sendWSError(conn, "invalid_request", "Expected type: response.create")
			continue
		}

		// Remove the protocol-level "type" field before forwarding.
		delete(reqMap, "type")

		// isStoreFalse is true only when the client explicitly sent store:false.
		// An absent "store" field defaults to true (persistent store), not false.
		isStoreFalse := false
		if storeRaw, exists := reqMap["store"]; exists {
			if b, ok := storeRaw.(bool); ok && !b {
				isStoreFalse = true
			}
		}
		prevRespID, _ := reqMap["previous_response_id"].(string)

		// Handle connection-local cache for store:false continuations.
		// Only when store:false was explicit — for store:true the orchestrator
		// handles previous_response_id via the persistent response store.
		if prevRespID != "" && isStoreFalse {
			cacheMu.Lock()
			prevResp, found := localCache[prevRespID]
			cacheMu.Unlock()

			if !found {
				sendWSError(conn, "previous_response_not_found",
					"The previous response was not found in this WebSocket session")
				continue
			}

			// Validate that every function_call_output in the input has a matching
			// function_call in the previous response. Mismatched call_ids indicate
			// a broken continuation; reject early and evict the stale cache entry so
			// a retry correctly returns previous_response_not_found.
			if inputRaw, ok := reqMap["input"]; ok {
				if inputArr, ok := inputRaw.([]interface{}); ok {
					allowedCallIDs := make(map[string]bool)
					for _, outItem := range prevResp.Output {
						if outItem.Type == "function_call" && outItem.CallID != "" {
							allowedCallIDs[outItem.CallID] = true
						}
					}
					for _, item := range inputArr {
						itemMap, ok := item.(map[string]interface{})
						if !ok {
							continue
						}
						if itemType, _ := itemMap["type"].(string); itemType == "function_call_output" {
							callID, _ := itemMap["call_id"].(string)
							if callID != "" && !allowedCallIDs[callID] {
								cacheMu.Lock()
								delete(localCache, prevRespID)
								cacheMu.Unlock()
								sendWSError(conn, "invalid_request_error",
									"No matching function call for call_id: "+callID)
								continue outerLoop
							}
						}
					}
				}
			}

			// Prepend the previous response output to current input and strip
			// previous_response_id so the proxy doesn't attempt its own store lookup.
			delete(reqMap, "previous_response_id")
			if bodyTmp, err := json.Marshal(reqMap); err == nil {
				if merged, err := responses.PrependOutputToInput(bodyTmp, prevResp.Output); err == nil {
					var mergedMap map[string]interface{}
					if json.Unmarshal(merged, &mergedMap) == nil {
						reqMap = mergedMap
					}
				}
			}
		}

		// Always stream over WebSocket.
		reqMap["stream"] = true

		bodyBytes, err := json.Marshal(reqMap)
		if err != nil {
			p.logger.ErrorContext(r.Context(), "ws: failed to marshal request",
				"error_code", http.StatusInternalServerError, "error", err)
			sendWSError(conn, "internal_error", "Failed to marshal request")
			continue
		}

		internalReq, err := http.NewRequestWithContext(
			r.Context(), "POST",
			"http://internal/v1/responses",
			bytes.NewReader(bodyBytes),
		)
		if err != nil {
			p.logger.ErrorContext(r.Context(), "ws: failed to create internal request",
				"error_code", http.StatusInternalServerError, "error", err)
			sendWSError(conn, "internal_error", "Failed to create request")
			continue
		}
		internalReq.URL.Path = "/v1/responses"
		internalReq.Header = r.Header.Clone()
		internalReq.Header.Set("Content-Type", "application/json")

		wsWriter := newWSSSEWriter(conn)

		// Run ProxyRequest in a goroutine; wait for the turn to finish.
		turnDone := make(chan struct{})
		go func() {
			defer close(turnDone)
			p.ProxyRequest(wsWriter, internalReq)

			// ProxyRequest returned. If it produced a non-SSE error body (e.g. HTTP 4xx
			// before streaming started), forward it and close the turn.
			select {
			case <-wsWriter.done:
				// Already signalled by SSE [DONE] — nothing to do.
			default:
				wsWriter.bufMu.Lock()
				wsWriter.parseSSEEvents()
				rawBuf := make([]byte, len(wsWriter.buf))
				copy(rawBuf, wsWriter.buf)
				wsWriter.buf = wsWriter.buf[:0]
				wsWriter.bufMu.Unlock()

				if len(rawBuf) > 0 && wsWriter.status >= 400 {
					// Non-streaming error: normalize raw JSON error body into a WS error event.
					wsWriter.writeMu.Lock()
					sendWSHTTPError(conn, rawBuf, wsWriter.status)
					wsWriter.writeMu.Unlock()
					wsWriter.isFailed = true
				}
				wsWriter.closeDone()
			}
		}()

		// Wait for the turn to complete or the client to disconnect.
		select {
		case <-wsWriter.done:
		case <-r.Context().Done():
			return
		}
		<-turnDone

		// Update connection-local cache (only for explicit store:false responses).
		if isStoreFalse {
			cacheMu.Lock()
			if wsWriter.finalResp != nil {
				localCache[wsWriter.finalResp.ID] = wsWriter.finalResp
			}
			// Evict the previous response when a continuation fails so that
			// a subsequent retry correctly returns previous_response_not_found.
			if wsWriter.isFailed && prevRespID != "" {
				delete(localCache, prevRespID)
			}
			cacheMu.Unlock()
		}
	}
}
