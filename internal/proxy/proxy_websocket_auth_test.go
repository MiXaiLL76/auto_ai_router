package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWebSocketResponsesAuthenticatesBeforeUpgrade(t *testing.T) {
	db := &clientAuthTestDB{
		tokens: map[string]*dbmodels.TokenInfo{
			"llm-key": {
				Token: "llm-key-hash",
			},
		},
		errors: map[string]error{"invalid-key": litellmdb.ErrTokenNotFound},
	}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
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
