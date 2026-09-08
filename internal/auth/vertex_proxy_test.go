package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVertexTokenUsesCredentialProxy(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		assert.Equal(t, "http://oauth.invalid/token", r.URL.String())
		assert.NoError(t, r.ParseForm())
		assert.NotEmpty(t, r.Form.Get("assertion"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"access_token":"proxied-token","token_type":"Bearer","expires_in":3600}`)
	}))
	defer server.Close()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	data, err := json.Marshal(map[string]string{
		"type": "service_account", "client_email": "test@example.com", "token_uri": "http://oauth.invalid/token",
		"private_key": string(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})),
	})
	require.NoError(t, err)
	tm := NewVertexTokenManager(testhelpers.NewTestLogger())
	defer tm.Stop()
	token, err := tm.GetToken("vertex", "", string(data), server.URL)
	require.NoError(t, err)
	require.Equal(t, "proxied-token", token)
	tm.mu.Lock()
	tm.tokens["vertex"].expiresAt = time.Now()
	tm.tokens["vertex"].token.Expiry = time.Now()
	tm.mu.Unlock()
	token, err = tm.GetToken("vertex", "", string(data), server.URL)
	require.NoError(t, err)
	require.Equal(t, "proxied-token", token)
	require.EqualValues(t, 2, requests.Load())
}
