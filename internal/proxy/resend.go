package proxy

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// credentialStatusError is a resent request the credential answered with a
// status other than 200.
type credentialStatusError struct {
	status int
	body   []byte
}

func (e *credentialStatusError) Error() string {
	return fmt.Sprintf("upstream answered %d: %s", e.status, e.body)
}

// retryable reports a status worth one more try: 429 or 5xx.
func (e *credentialStatusError) retryable() bool {
	return e.status == http.StatusTooManyRequests || e.status >= http.StatusInternalServerError
}

// resendToCredential sends body to the credential of an upstream attempt
// already made, as a clone of template (method, URL, headers, and a context
// carrying the upstream deadline and the credential's proxy_url). It is
// refused once clientCtx is done, so a generation nobody reads is never
// started. Used by the loop gate's restarts and the Gemini long-decode
// follow-ups, it counts as a request to the credential like any other:
//
//   - toward its RPM (and the model's), while under the limit: the generation
//     it belongs to is already running, so it is never refused or held;
//   - a Vertex AI access token is fetched again: a long generation can
//     outlive the first one;
//   - its outcome goes to fail2ban and the attempt error metric, a transport
//     error as 502 / 408 unless the client is gone.
//
// A status other than 200 is a *credentialStatusError, its body read (up to
// maxRetryBodyScan) and closed.
func (p *Proxy) resendToCredential(clientCtx context.Context, template *http.Request, body []byte, cred *config.CredentialConfig, modelID string) (*http.Response, error) {
	if err := clientCtx.Err(); err != nil {
		return nil, err
	}
	if p.rateLimiter != nil {
		p.rateLimiter.Allow(cred.Name)
		if modelID != "" {
			p.rateLimiter.AllowModel(cred.Name, modelID)
		}
	}
	req := template.Clone(template.Context())
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	req.ContentLength = int64(len(body))
	if cred.EffectiveProviderType() == config.ProviderTypeVertexAI {
		token, err := p.tokenManager.GetToken(cred.Name, cred.CredentialsFile, cred.CredentialsJSON, cred.ProxyURL)
		if err != nil {
			return nil, fmt.Errorf("vertex token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := p.client.Do(req) //nolint:gosec // G704: the URL of the attempt being resent, from the credential's configuration
	if err != nil {
		if clientCtx.Err() != nil {
			// The client left: not the credential's failure.
			return nil, err
		}
		status := http.StatusBadGateway
		if isTimeoutError(err) {
			status = http.StatusRequestTimeout
		}
		p.balancer.RecordResponse(cred.Name, modelID, status)
		p.metrics.RecordCredentialAttemptError(cred.Name)
		return nil, err
	}
	p.balancer.RecordResponse(cred.Name, modelID, resp.StatusCode)
	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxRetryBodyScan))
		_ = resp.Body.Close()
		p.metrics.RecordCredentialAttemptError(cred.Name)
		return nil, &credentialStatusError{status: resp.StatusCode, body: bytes.TrimSpace(errBody)}
	}
	return resp, nil
}
