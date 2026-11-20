package proxy

import (
	"context"
	"fmt"
	"net/http"

	goccyjson "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// systemOnePath is the native System One API endpoint. A request carries the
// model, a state (string, object or array) and questions with nested
// instructions and criteria; the reply carries answers, probabilities and
// usage.input_tokens / output_tokens. There is no chat shape and no streaming:
// the body goes upstream as is, only the model id is swapped.
const systemOnePath = "/v1/systemone"

func isSystemOnePath(path string) bool {
	return path == systemOnePath
}

// systemOneExclusions returns the credentials a request may not use because of
// the API it calls: /v1/systemone goes only to system_one credentials, every
// other endpoint never does. Proxy-like credentials are never excluded: their
// peer router applies the same rule to the request it receives.
func (p *Proxy) systemOneExclusions(systemOne bool) map[string]bool {
	names := p.balancer.CredentialNamesMatching(func(cred *config.CredentialConfig) bool {
		return !cred.IsProxyLike() && cred.SystemOne != systemOne
	})
	if len(names) == 0 {
		return nil
	}
	excluded := make(map[string]bool, len(names))
	for _, name := range names {
		excluded[name] = true
	}
	return excluded
}

// admitSystemOneRequest rejects, before a credential is picked, a request the
// model cannot serve on the API it calls: stream:true on /v1/systemone, a model
// without a system_one credential on /v1/systemone, and a System One model on
// any other endpoint. A model with no credentials at all falls through to the
// usual 404 in selectCredentialForModel.
func (p *Proxy) admitSystemOneRequest(
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	modelID string,
	streaming bool,
	systemOne bool,
	excluded map[string]bool,
) bool {
	reject := func(message string) bool {
		p.logger.WarnContext(logCtx.Context(), "Request rejected: endpoint not supported by the model",
			"error_code", http.StatusBadRequest,
			"model", modelID,
			"endpoint", logCtx.RequestEndpoint,
			"error", message,
			"request_id", logCtx.RequestID,
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = message
		WriteErrorBadRequest(w, message)
		return false
	}
	if systemOne && streaming {
		return reject("stream is not supported by the System One API")
	}
	if len(excluded) == 0 || p.modelManager == nil || !p.modelManager.IsEnabled() {
		return true
	}
	creds := p.modelManager.GetCredentialsForModel(modelID)
	if len(creds) == 0 {
		return true
	}
	for _, name := range creds {
		if !excluded[name] {
			return true
		}
	}
	if systemOne {
		return reject(fmt.Sprintf("model %s does not support the System One API", modelID))
	}
	return reject(fmt.Sprintf("model %s is only available via %s", modelID, systemOnePath))
}

// systemOneUsage settles the token count of a non-streaming System One reply.
// The API reports usage.input_tokens / output_tokens and need not send
// total_tokens, the only field the rate limiter otherwise reads, so their sum
// feeds the credential and model TPM instead. A successful reply without any
// usage is billed on the request's prompt estimate instead of passing as a
// free call, and logged; with no requestBody at hand the request's armed
// estimate is used. Every other request is returned unchanged.
func (p *Proxy) systemOneUsage(
	ctx context.Context,
	logCtx *RequestLogContext,
	credName string,
	statusCode int,
	requestBody []byte,
	realModelID string,
	tokens int,
	usage *converter.TokenUsage,
) (int, *converter.TokenUsage) {
	if logCtx == nil || !logCtx.IsSystemOne || statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return tokens, usage
	}
	if usage.Total() == 0 {
		estimate := 0
		switch {
		case requestBody == nil:
			estimate = logCtx.promptTokensEstimate()
		case p.tiktokenEnabled:
			estimate = estimatePromptTokensForModel(requestBody, realModelID)
		}
		p.logger.WarnContext(ctx, "System One reply carries no usage, billing the request's prompt estimate",
			"credential", credName,
			"model", realModelID,
			"estimated_prompt_tokens", estimate,
			"request_id", logCtx.RequestID,
		)
		if estimate > 0 {
			usage = &converter.TokenUsage{PromptTokens: estimate}
			logCtx.UsageSource = "estimated"
		}
	}
	if tokens <= 0 {
		tokens = usage.Total()
	}
	return tokens, usage
}

// observeSystemOneReplyModel logs a System One reply that names another model
// version than the one requested. The client is billed at the public model's
// price and sees the public id, so without this a provider answering from
// another version would go unnoticed.
func (p *Proxy) observeSystemOneReplyModel(ctx context.Context, logCtx *RequestLogContext, credName, requested string, body []byte) {
	if logCtx == nil || !logCtx.IsSystemOne {
		return
	}
	var reply struct {
		Model string `json:"model"`
	}
	if goccyjson.Unmarshal(body, &reply) != nil || reply.Model == "" || reply.Model == requested {
		return
	}
	p.logger.WarnContext(ctx, "System One reply names another model version than requested",
		"credential", credName,
		"requested_model", requested,
		"reply_model", reply.Model,
		"request_id", logCtx.RequestID,
	)
}
