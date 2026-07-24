package proxy

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// captureResponseWriter is a simple http.ResponseWriter that buffers the response.
type captureResponseWriter struct {
	header     http.Header
	body       bytes.Buffer
	statusCode int
}

func newCaptureResponseWriter() *captureResponseWriter {
	return &captureResponseWriter{header: make(http.Header), statusCode: 200}
}

func (cw *captureResponseWriter) Header() http.Header         { return cw.header }
func (cw *captureResponseWriter) WriteHeader(code int)        { cw.statusCode = code }
func (cw *captureResponseWriter) Write(b []byte) (int, error) { return cw.body.Write(b) }

// HandleCompactResponse handles POST /v1/responses/compact.
// It summarizes the provided conversation via the LLM and returns a CompactResource
// containing a single compaction output item that can be reused as input context.
func (p *Proxy) HandleCompactResponse(w http.ResponseWriter, r *http.Request) {
	// Authorize the public compact operation before reading or transforming its
	// body. The inner chat request inherits this in-process identity rather than
	// re-checking allowed_routes against the implementation-detail chat path.
	tokenInfo, ok := p.AuthenticateClientRequest(w, r)
	if !ok {
		return
	}
	rawToken, _ := extractClientToken(r)

	body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20))
	if err != nil {
		WriteErrorBadRequest(w, "Failed to read request body")
		return
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		WriteErrorBadRequest(w, "Invalid JSON")
		return
	}

	model, _ := raw["model"].(string)
	if model == "" {
		WriteErrorBadRequest(w, "model is required for compaction")
		return
	}

	// Build a chat completions request that asks the model to compact the conversation.
	chatReq := map[string]interface{}{
		"model":    model,
		"messages": buildCompactionMessages(raw["input"]),
	}
	chatBody, err := json.Marshal(chatReq)
	if err != nil {
		WriteErrorInternal(w, "Failed to build compaction request")
		return
	}

	internalReq, err := http.NewRequestWithContext(r.Context(), "POST", "http://internal/v1/chat/completions", bytes.NewReader(chatBody))
	if err != nil {
		WriteErrorInternal(w, "Failed to create internal request")
		return
	}
	internalReq.URL.Path = "/v1/chat/completions"
	internalReq.Header = r.Header.Clone()
	internalReq.Header.Set("Content-Type", "application/json")
	internalReq = withTrustedClientAuth(internalReq, rawToken, tokenInfo)

	capture := newCaptureResponseWriter()
	p.ProxyRequest(capture, internalReq)

	if capture.statusCode >= 400 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(capture.statusCode)
		_, _ = w.Write(capture.body.Bytes())
		return
	}

	var ccResp struct {
		ID      string `json:"id"`
		Created int64  `json:"created"`
		Usage   *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	respBody := capture.body.Bytes()
	if capture.header.Get("Content-Encoding") == "gzip" {
		gr, err := gzip.NewReader(bytes.NewReader(respBody))
		if err != nil {
			WriteErrorInternal(w, "Failed to decompress LLM response")
			return
		}
		decompressed, err := io.ReadAll(gr)
		if closeErr := gr.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
		if err != nil {
			WriteErrorInternal(w, "Failed to read decompressed LLM response")
			return
		}
		respBody = decompressed
	}

	if err := json.Unmarshal(respBody, &ccResp); err != nil {
		WriteErrorInternal(w, "Failed to parse LLM response")
		return
	}

	summary := ""
	if len(ccResp.Choices) > 0 {
		summary = ccResp.Choices[0].Message.Content
	}

	createdAt := ccResp.Created
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}

	var usage *responses.Usage
	if ccResp.Usage != nil {
		usage = &responses.Usage{
			InputTokens:         ccResp.Usage.PromptTokens,
			OutputTokens:        ccResp.Usage.CompletionTokens,
			TotalTokens:         ccResp.Usage.TotalTokens,
			InputTokensDetails:  responses.InputDetails{},
			OutputTokensDetails: responses.OutputDetails{},
		}
	}

	result := responses.CompactResource{
		ID:     responses.GenerateResponseID(),
		Object: "response.compaction",
		Output: []responses.OutputItem{{
			Type:             "compaction",
			ID:               responses.GenerateItemID("compact_"),
			EncryptedContent: summary,
		}},
		CreatedAt: createdAt,
		Usage:     usage,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(result); err != nil {
		p.logger.ErrorContext(r.Context(), "compact: encode error", "error", err)
	}
}

// buildCompactionMessages converts a Responses API input value into a pair of chat
// messages (system + user) that ask the model to produce a compact conversation summary.
func buildCompactionMessages(input interface{}) []map[string]interface{} {
	system := map[string]interface{}{
		"role":    "system",
		"content": "You are a conversation memory compactor. Summarize the following conversation into a dense, lossless context that preserves all key facts, decisions, and context needed to continue the conversation accurately. Output only the summary, no meta-commentary.",
	}

	conversationText := extractConversationText(input)
	if conversationText == "" {
		conversationText = "(empty conversation)"
	}

	user := map[string]interface{}{
		"role":    "user",
		"content": "Produce a compact memory summary of this conversation:\n\n" + conversationText,
	}

	return []map[string]interface{}{system, user}
}

func extractConversationText(input interface{}) string {
	switch v := input.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, item := range v {
			switch it := item.(type) {
			case map[string]interface{}:
				role, _ := it["role"].(string)
				content := extractContentText(it["content"])
				if role != "" && content != "" {
					parts = append(parts, role+": "+content)
				}
			case string:
				parts = append(parts, it)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func extractContentText(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var parts []string
		for _, p := range c {
			if pm, ok := p.(map[string]interface{}); ok {
				if t, ok := pm["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, " ")
	}
	return ""
}
