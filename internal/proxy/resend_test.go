package proxy

import (
	"context"
	"io"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

func TestResendToCredential(t *testing.T) {
	monitoring.CredentialErrorsTotal.Reset()
	var requests atomic.Int32
	status := atomic.Int32{}
	srv := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		body, _ := io.ReadAll(r.Body)
		w.WriteHeader(int(status.Load()))
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	cred := config.CredentialConfig{Name: "rs-cred", Type: config.ProviderTypeVLLM, BaseURL: srv.URL, RPM: 100, TPM: -1}
	builder := NewTestProxyBuilder().WithCredentials(cred)
	builder.config.Metrics = monitoring.New(true)
	prx := builder.Build()
	template, err := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", nil)
	require.NoError(t, err)
	rpmBefore := prx.rateLimiter.GetCurrentRPM("rs-cred")

	status.Store(http.StatusOK)
	resp, err := prx.resendToCredential(context.Background(), template, []byte(`{"n":1}`), &cred, "")
	require.NoError(t, err)
	got, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	assert.Equal(t, `{"n":1}`, string(got), "the body is sent")

	status.Store(http.StatusServiceUnavailable)
	_, err = prx.resendToCredential(context.Background(), template, []byte(`overloaded`), &cred, "")
	var statusErr *credentialStatusError
	require.ErrorAs(t, err, &statusErr)
	assert.Equal(t, http.StatusServiceUnavailable, statusErr.status)
	assert.True(t, statusErr.retryable())
	assert.Equal(t, "overloaded", string(statusErr.body))

	assert.Equal(t, 2, prx.rateLimiter.GetCurrentRPM("rs-cred")-rpmBefore, "every resend counts toward the credential RPM")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.CredentialErrorsTotal.WithLabelValues("rs-cred")), "the 503 is a failed attempt")

	gone, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = prx.resendToCredential(gone, template, []byte(`{}`), &cred, "")
	assert.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, int32(2), requests.Load(), "nothing is sent once the client is gone")
	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.CredentialErrorsTotal.WithLabelValues("rs-cred")), "the client leaving is not the credential's failure")
}
