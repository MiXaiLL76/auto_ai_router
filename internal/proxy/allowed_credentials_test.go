package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/require"
)

func requestCredentialAllowlist(p *Proxy, token, model, session string) *httptest.ResponseRecorder {
	body, _ := json.Marshal(map[string]any{"model": model, "messages": []any{}})
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", "application/json")
	if session != "" {
		request.Header.Set("Session-Id", session)
	}
	recorder := httptest.NewRecorder()
	p.ProxyRequest(recorder, request)
	return recorder
}

func TestCredentialAllowlistAppliesAcrossModelsAndReplacesSticky(t *testing.T) {
	var normalCalls, allowedCalls, peerCalls atomic.Int32
	var unused atomic.Bool
	normalServer := denylistProvider(t, &normalCalls, &unused)
	allowedServer := denylistProvider(t, &allowedCalls, &unused)
	peerServer := denylistProvider(t, &peerCalls, &unused)
	defer normalServer.Close()
	defer allowedServer.Close()
	defer peerServer.Close()

	normal := directDenylistCredential("private-normal", normalServer.URL)
	normal.Weight = 100
	allowed := directDenylistCredential("private-account", allowedServer.URL)
	allowed.RoutingID = "route-02"
	peer := proxyCred("private-peer", peerServer.URL, 100)
	peer.RoutingID = "route-02" // Guard even programmatically constructed peers.
	p := NewTestProxyBuilder().WithCredentials(normal, allowed, peer).
		WithSessionSticky(5 * time.Minute).Build()
	info := &dbmodels.TokenInfo{Token: "restricted-hash", UserID: "test-user"}
	p.LiteLLMDB = &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": info}}
	registerTestModel(p, normal.Name, "route-a")
	registerTestModel(p, allowed.Name, "route-a")
	registerTestModel(p, allowed.Name, "route-b")
	setTestModelPrice(p, "route-a", &routermodels.ModelPrice{})
	p.priceRegistry.Update(map[string]*routermodels.ModelPrice{
		"route-a": {},
		"route-b": {},
	})
	initial := requestCredentialAllowlist(p, "token", "route-a", "session-1")
	require.Equal(t, http.StatusOK, initial.Code, initial.Body.String())
	require.EqualValues(t, 1, normalCalls.Load())
	sessionKey := p.stickySessionKey(&RequestLogContext{Token: "token", SessionID: "session-1"})
	require.Equal(t, normal.Name, boundTo(t, p, sessionKey))

	// Restrict the same authenticated key after it acquired the sticky binding.
	info.Metadata = map[string]any{"allowed_cred": []any{"route-02"}}
	registerTestModel(p, peer.Name, "route-a")
	require.Equal(t, http.StatusOK, requestCredentialAllowlist(p, "token", "route-a", "session-1").Code)
	require.Equal(t, allowed.Name, boundTo(t, p, sessionKey))
	require.Equal(t, http.StatusOK, requestCredentialAllowlist(p, "token", "route-b", "").Code)
	require.EqualValues(t, 1, normalCalls.Load())
	require.EqualValues(t, 2, allowedCalls.Load())
	require.Zero(t, peerCalls.Load())

	// A disjoint parent policy and an invalid value must both stop before sending.
	info.OrganizationMetadata = map[string]any{"allowed_cred": []string{"route-03"}}
	failure := requestCredentialAllowlist(p, "token", "route-a", "session-1")
	require.Equal(t, http.StatusServiceUnavailable, failure.Code)
	require.NotContains(t, failure.Body.String(), "private-account")
	require.NotContains(t, failure.Body.String(), "route-02")
	info.OrganizationMetadata = nil
	info.Metadata = map[string]any{"allowed_cred": "private-account"}
	require.Equal(t, http.StatusServiceUnavailable, requestCredentialAllowlist(p, "token", "route-a", "").Code)
	require.EqualValues(t, 1, normalCalls.Load())
	require.EqualValues(t, 2, allowedCalls.Load())
	require.Zero(t, peerCalls.Load())
}

func TestCredentialAllowlistConstrainsRetryAndOrganizationDenylist(t *testing.T) {
	var failingCalls, otherCalls, reserveCalls atomic.Int32
	var unused atomic.Bool
	failingServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		failingCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"unavailable"}}`))
	}))
	otherServer := denylistProvider(t, &otherCalls, &unused)
	reserveServer := denylistProvider(t, &reserveCalls, &unused)
	defer failingServer.Close()
	defer otherServer.Close()
	defer reserveServer.Close()

	main := directDenylistCredential("private-main", failingServer.URL)
	main.RoutingID, main.Priority, main.Weight = "route-01", 1, 100
	other := directDenylistCredential("private-other", otherServer.URL)
	other.RoutingID, other.Priority, other.Weight = "route-03", 2, 100
	reserve := directDenylistCredential("private-reserve", reserveServer.URL)
	reserve.RoutingID, reserve.Priority = "route-02", 3
	info := &dbmodels.TokenInfo{
		Token: "restricted-hash", UserID: "test-user", OrganizationID: "org-1", DirectOrganizationID: "org-1",
		Metadata: map[string]any{"allowed_cred": []string{"route-01", "route-02"}},
	}
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{"token": info}}
	credentials := []config.CredentialConfig{main, other, reserve}
	p := newLocalDenylistPolicyProxy(t, credentials, db, nil, true)
	p.maxProviderRetries = 2
	setTestModelPrice(p, "route-a", &routermodels.ModelPrice{})
	response := requestCredentialAllowlist(p, "token", "route-a", "")
	require.Equal(t, http.StatusOK, response.Code, response.Body.String())
	require.EqualValues(t, 1, failingCalls.Load())
	require.EqualValues(t, 1, reserveCalls.Load())
	require.Zero(t, otherCalls.Load())

	p = newLocalDenylistPolicyProxy(t, credentials, db, []string{reserve.Name}, true)
	p.maxProviderRetries = 2
	setTestModelPrice(p, "route-a", &routermodels.ModelPrice{})
	failure := requestCredentialAllowlist(p, "token", "route-a", "")
	require.Equal(t, http.StatusInternalServerError, failure.Code)
	require.EqualValues(t, 2, failingCalls.Load())
	require.EqualValues(t, 1, reserveCalls.Load())
	require.Zero(t, otherCalls.Load())
}
