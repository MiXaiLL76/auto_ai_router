package proxy

import (
	"bytes"
	"encoding/json"
)

const clientResponseIDScanLimit = 256 << 10

type clientResponseIDEnvelope struct {
	ID       string `json:"id"`
	Response *struct {
		ID string `json:"id"`
	} `json:"response"`
	Message *struct {
		ID string `json:"id"`
	} `json:"message"`
}

func (logCtx *RequestLogContext) captureClientResponseID(payload []byte) {
	if logCtx == nil || logCtx.ClientResponseID != "" {
		return
	}

	var envelope clientResponseIDEnvelope
	if json.Unmarshal(payload, &envelope) != nil {
		return
	}

	switch {
	case envelope.ID != "":
		logCtx.ClientResponseID = envelope.ID
	case envelope.Response != nil && envelope.Response.ID != "":
		logCtx.ClientResponseID = envelope.Response.ID
	case envelope.Message != nil && envelope.Message.ID != "":
		logCtx.ClientResponseID = envelope.Message.ID
	}
}

func (logCtx *RequestLogContext) spendRequestID() string {
	if logCtx != nil && logCtx.ClientResponseID != "" {
		return logCtx.ClientResponseID
	}
	if logCtx == nil {
		return ""
	}
	return logCtx.RequestID
}

type clientResponseIDScanner struct {
	buffer []byte
	total  int
}

type clientResponseIDObserver struct {
	scanner *clientResponseIDScanner
	logCtx  *RequestLogContext
}

func (o clientResponseIDObserver) Write(chunk []byte) (int, error) {
	o.scanner.observe(o.logCtx, chunk)
	return len(chunk), nil
}

func (s *clientResponseIDScanner) observe(logCtx *RequestLogContext, chunk []byte) {
	if logCtx == nil || logCtx.ClientResponseID != "" || s.total >= clientResponseIDScanLimit {
		return
	}

	remaining := clientResponseIDScanLimit - s.total
	if len(chunk) > remaining {
		chunk = chunk[:remaining]
	}
	s.total += len(chunk)
	s.buffer = append(s.buffer, chunk...)

	for {
		newline := bytes.IndexByte(s.buffer, '\n')
		if newline < 0 {
			return
		}
		line := bytes.TrimSpace(s.buffer[:newline])
		s.buffer = s.buffer[newline+1:]
		if bytes.HasPrefix(line, []byte("data:")) {
			line = bytes.TrimSpace(line[len("data:"):])
		}
		logCtx.captureClientResponseID(line)
		if logCtx.ClientResponseID != "" {
			return
		}
	}
}
