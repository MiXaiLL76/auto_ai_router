package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeResponseStore struct {
	entry *responsestore.StoredEntry
}

func (s *fakeResponseStore) SaveResponse(context.Context, string, *responses.Response, map[string]string, int, json.RawMessage, string) error {
	return nil
}

func (s *fakeResponseStore) GetResponse(context.Context, string, string) (*responses.Response, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *fakeResponseStore) GetEntry(context.Context, string, string) (*responsestore.StoredEntry, error) {
	if s.entry == nil {
		return nil, fmt.Errorf("not found")
	}
	return s.entry, nil
}

func (s *fakeResponseStore) GetResponseByID(context.Context, string) (*responses.Response, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *fakeResponseStore) CleanupExpired(context.Context) error { return nil }
func (s *fakeResponseStore) Close() error                         { return nil }

func TestProxyRequest_ResponsesAPIPreviousResponseIDUsesStoredCredential(t *testing.T) {
	server1 := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp-1",
			"object":"response",
			"model":"gpt-4",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"from-cred-1"}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server1.Close()

	server2 := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp-2",
			"object":"response",
			"model":"gpt-4",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"from-cred-2"}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server2.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			testProxyCredential("cred1", server1.URL, false),
			testProxyCredential("cred2", server2.URL, false),
		).
		Build()
	prx.responseStore = &fakeResponseStore{
		entry: &responsestore.StoredEntry{
			ResponseID:     "resp-prev",
			APIKeyHash:     "hash",
			CredentialName: "cred2",
			ResponseJSON: &responses.Response{
				ID:    "resp-prev",
				Model: "gpt-4",
			},
		},
	}

	reqBody := `{"model":"gpt-4","input":"hello","previous_response_id":"resp-prev"}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "from-cred-2")
	assert.NotContains(t, w.Body.String(), "from-cred-1")
}

func testProxyCredential(name, baseURL string, lastResort bool) config.CredentialConfig {
	priority := 0
	if lastResort {
		priority = config.FallbackPriorityGroup
	}
	return config.CredentialConfig{
		Name:     name,
		Type:     config.ProviderTypeProxy,
		BaseURL:  baseURL,
		APIKey:   "upstream-key-" + name,
		RPM:      100,
		TPM:      10000,
		Priority: priority,
	}
}
