package router

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"
)

type messagesErrorWriter struct {
	http.ResponseWriter
	body      bytes.Buffer
	status    int
	buffering bool
}

func (w *messagesErrorWriter) WriteHeader(status int) {
	if status >= http.StatusBadRequest {
		w.status = status
		w.buffering = true
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *messagesErrorWriter) Write(body []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	if w.buffering {
		return w.body.Write(body)
	}
	return w.ResponseWriter.Write(body)
}

func (w *messagesErrorWriter) Flush() {
	if w.buffering {
		return
	}
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *messagesErrorWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *messagesErrorWriter) finalize() {
	if !w.buffering {
		return
	}
	message := http.StatusText(w.status)
	var response struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Detail string `json:"detail"`
	}
	if json.Unmarshal(w.body.Bytes(), &response) == nil {
		if response.Error.Message != "" {
			message = response.Error.Message
		} else if response.Detail != "" {
			message = response.Detail
		}
	} else if w.body.Len() > 0 {
		message = w.body.String()
	}
	body, _ := json.Marshal(map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    messagesErrorType(w.status),
			"message": message,
		},
	})
	header := w.Header()
	header.Del("Content-Encoding")
	header.Set("Content-Type", "application/json")
	header.Set("Content-Length", strconv.Itoa(len(body)))
	w.ResponseWriter.WriteHeader(w.status)
	_, _ = w.ResponseWriter.Write(body)
}

func messagesErrorType(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

func (r *Router) proxyPublicRequest(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path != "/v1/messages" {
		r.proxy.ProxyRequest(w, req)
		return
	}
	writer := &messagesErrorWriter{ResponseWriter: w}
	req.Header.Set("Accept-Encoding", "identity")
	r.proxy.ProxyRequest(writer, req)
	writer.finalize()
}
