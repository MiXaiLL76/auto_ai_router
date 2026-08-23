package proxy

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	aimodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProtectedWebSocketTurnReusesHandshakeRouteAndRevalidatesKey(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		assert.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_cloud\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"model\":\"public/chat\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\ndata: [DONE]\n\n"))
	}))
	defer upstream.Close()

	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"cloud-key": {
			Token:          "cloud-key-hash",
			OrganizationID: "org-cloud",
			Models:         []string{"public/chat"},
			AllowedRoutes:  []string{"GET /v1/responses"},
			Metadata:       map[string]interface{}{"air_scopes": []string{"cloud-ru"}},
		},
	}}
	passthrough := true
	logger := testhelpers.NewTestLogger()
	credential := config.CredentialConfig{
		Name: "provider", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "provider-key", RPM: 100, TPM: 10000,
	}
	manager := aimodels.New(logger, 100, []config.ModelRPMConfig{{
		Name: "public/chat", Credential: credential.Name, RPM: 100, TPM: 10000, PassthroughResponses: &passthrough,
	}})
	manager.SetClientModelIDs([]string{"public/chat"})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})
	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	prx := builder.Build()
	prx.LiteLLMDB = db
	prx.protectedTenants = []config.ProtectedTenantConfig{{
		Name: "cloud-ru", OrganizationIDs: []string{"org-cloud"}, RequiredScopes: []string{"cloud-ru"},
		RequireModelACL: true, RequireRouteACL: true,
	}}

	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	defer server.Close()
	serverHost, _, splitErr := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	require.NoError(t, splitErr)
	prx.protectedTenants[0].Hostnames = []string{serverHost}
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/v1/responses"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, http.Header{"Authorization": []string{"Bearer cloud-key"}})
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	require.NoError(t, conn.WriteJSON(map[string]interface{}{
		"type": "response.create", "model": "public/chat", "input": "hello",
	}))
	require.Eventually(t, func() bool {
		return upstreamCalls.Load() == 1 && len(db.seenTokens()) >= 2
	}, 5*time.Second, 10*time.Millisecond, "inner POST must reuse the authorized WS route and revalidate the key")

	db.mu.Lock()
	db.tokens["cloud-key"].OrganizationID = "org-other"
	db.tokens["cloud-key"].Metadata = map[string]interface{}{"air_scopes": []string{"other"}}
	db.mu.Unlock()
	require.NoError(t, conn.WriteJSON(map[string]interface{}{
		"type": "response.create", "model": "public/chat", "input": "second",
	}))
	require.Eventually(t, func() bool { return len(db.seenTokens()) >= 3 }, 5*time.Second, 10*time.Millisecond)
	assert.Never(t, func() bool { return upstreamCalls.Load() > 1 }, 250*time.Millisecond, 10*time.Millisecond,
		"changed organization on the protected Host must block the second turn")
}

func TestWebSocketResponsesAuthenticatesBeforeUpgrade(t *testing.T) {
	db := &clientAuthTestDB{
		tokens: map[string]*dbmodels.TokenInfo{
			"llm-key": {
				Token: "llm-key-hash",
			},
			"cloud-key": {
				Token:          "cloud-key-hash",
				OrganizationID: "org-cloud",
				Models:         []string{"public/chat"},
				AllowedRoutes:  []string{"GET /v1/responses"},
				Metadata:       map[string]interface{}{"air_scopes": []string{"cloud-ru"}},
			},
		},
		errors: map[string]error{"invalid-key": litellmdb.ErrTokenNotFound},
	}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	prx.protectedTenants = []config.ProtectedTenantConfig{{
		Name: "cloud-ru", OrganizationIDs: []string{"org-cloud"}, RequiredScopes: []string{"cloud-ru"},
		RequireModelACL: true, RequireRouteACL: true,
	}}
	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	t.Cleanup(server.Close)
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/v1/responses"

	tests := []struct {
		name       string
		headers    http.Header
		wantStatus int
	}{
		{name: "missing", headers: http.Header{}, wantStatus: http.StatusUnauthorized},
		{name: "invalid bearer", headers: http.Header{"Authorization": []string{"Bearer invalid-key"}}, wantStatus: http.StatusUnauthorized},
		{name: "valid bearer", headers: http.Header{"Authorization": []string{"Bearer llm-key"}}, wantStatus: http.StatusSwitchingProtocols},
		{name: "valid x api key", headers: http.Header{"x-api-key": []string{"llm-key"}}, wantStatus: http.StatusSwitchingProtocols},
		{name: "protected websocket route", headers: http.Header{"Authorization": []string{"Bearer cloud-key"}}, wantStatus: http.StatusSwitchingProtocols},
		{name: "master", headers: http.Header{"Authorization": []string{"Bearer master-key"}}, wantStatus: http.StatusSwitchingProtocols},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conn, response, err := websocket.DefaultDialer.Dial(wsURL, tt.headers)
			if tt.wantStatus == http.StatusSwitchingProtocols {
				require.NoError(t, err)
				require.NotNil(t, conn)
				require.NoError(t, conn.Close())
				return
			}

			require.Error(t, err)
			if conn != nil {
				_ = conn.Close()
			}
			require.NotNil(t, response)
			assert.Equal(t, tt.wantStatus, response.StatusCode)
			require.NoError(t, response.Body.Close())
		})
	}
}

func TestWaitForWSTurnWaitsForWorkerAfterRequestCancellation(t *testing.T) {
	turnDone := make(chan struct{})
	streamDone := make(chan struct{})
	requestDone := make(chan struct{})
	close(requestDone)

	result := make(chan bool, 1)
	go func() {
		result <- waitForWSTurn(turnDone, streamDone, requestDone)
	}()

	select {
	case <-result:
		t.Fatal("waitForWSTurn returned while the worker was still running")
	case <-time.After(25 * time.Millisecond):
	}

	close(turnDone)
	assert.False(t, <-result)
}
