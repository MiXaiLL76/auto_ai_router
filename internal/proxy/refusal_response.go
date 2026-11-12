package proxy

import (
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/publicrefusal"
)

func isAnthropicResponse(cred *config.CredentialConfig, model, endpoint string) bool {
	if endpoint == "/v1/messages" {
		return true
	}
	switch cred.EffectiveProviderType() {
	case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI, config.ProviderTypeProMan:
		return true
	case config.ProviderTypeBedrock:
		return strings.HasPrefix(model, "anthropic.") || strings.Contains(model, ".anthropic.")
	default:
		return false
	}
}

// Invalid native JSON must fail closed, just like an invalid native SSE event.
// This is a malformed provider response, not a model refusal: valid refusals
// keep their status, usage and retry behavior.
func sanitizeRefusalResponse(status int, body []byte, logCtx *RequestLogContext) (int, []byte) {
	if status < 200 || status >= 300 {
		return status, body
	}
	clean, _, err := publicrefusal.SanitizeJSON(body)
	if err == nil {
		return status, clean
	}
	if logCtx != nil {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadGateway
		logCtx.ErrorMsg = "Invalid provider response"
	}
	return http.StatusBadGateway, maskedUpstreamErrorBody(http.StatusBadGateway, logContextRequestID(logCtx))
}

type messagesSanitizingReadCloser struct {
	*io.PipeReader
	source io.Closer
}

func (r *messagesSanitizingReadCloser) Close() error {
	return errors.Join(r.PipeReader.Close(), r.source.Close())
}

func newMessagesSanitizingReadCloser(source io.ReadCloser, model string) io.ReadCloser {
	reader, writer := io.Pipe()
	go func() {
		_ = writer.CloseWithError(anthropicconv.PassthroughMessagesStream(source, writer, model))
	}()
	return &messagesSanitizingReadCloser{PipeReader: reader, source: source}
}
