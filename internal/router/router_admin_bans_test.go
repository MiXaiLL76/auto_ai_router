package router

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const adminTestMasterKey = "admin-master-key"

type adminBanEnv struct {
	router *Router
	f2b    *fail2ban.Fail2Ban
	bal    *balancer.RoundRobin
}

func newAdminBanEnv(t *testing.T) *adminBanEnv {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, time.Minute, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "vllm-1", APIKey: "k1", BaseURL: "http://vllm-1.test", RPM: 100},
		{Name: "vllm-2", APIKey: "k2", BaseURL: "http://vllm-2.test", RPM: 100},
	}
	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
		rl.AddModel(cred.Name, "llama", 100)
		rl.AddModel(cred.Name, "qwen", 100)
	}

	bal := balancer.New(credentials, f2b, rl)
	prx := proxy.New(&proxy.Config{
		Balancer:            bal,
		Logger:              logger,
		MaxBodySizeMB:       10,
		RequestTimeout:      30 * time.Second,
		MaxIdleConns:        10,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout:     time.Minute,
		Metrics:             monitoring.New(false),
		MasterKey:           adminTestMasterKey,
		RateLimiter:         rl,
		TokenManager:        auth.NewVertexTokenManager(logger),
		ModelManager:        createTestModelManager(),
		Version:             "test",
		Commit:              "test",
	})
	return &adminBanEnv{
		router: New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil),
		f2b:    f2b,
		bal:    bal,
	}
}

func (e *adminBanEnv) do(method, path, body, token string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	e.router.ServeHTTP(w, req)
	return w
}

func (e *adminBanEnv) master(method, path, body string) *httptest.ResponseRecorder {
	return e.do(method, path, body, adminTestMasterKey)
}

func decodeBody(t *testing.T, w *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var out map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out), w.Body.String())
	return out
}

func TestAdminBans_AuthMasterKeyOnly(t *testing.T) {
	env := newAdminBanEnv(t)
	body := `{"credential":"vllm-1","ttl":"1m"}`

	cases := []struct {
		name   string
		method string
		path   string
		token  string
		want   int
	}{
		{"ban without token", "POST", "/api/ban", "", http.StatusUnauthorized},
		{"ban with non-master token", "POST", "/api/ban", "sk-some-litellm-key", http.StatusForbidden},
		{"unban without token", "POST", "/api/unban", "", http.StatusUnauthorized},
		{"unban with non-master token", "POST", "/api/unban", "sk-some-litellm-key", http.StatusForbidden},
		{"list without token", "GET", "/api/bans", "", http.StatusUnauthorized},
		{"list with non-master token", "GET", "/api/bans", "sk-some-litellm-key", http.StatusForbidden},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := env.do(tc.method, tc.path, body, tc.token)
			assert.Equal(t, tc.want, w.Code)
		})
	}
	assert.False(t, env.f2b.IsBanned("vllm-1", "llama"), "unauthorized calls must not ban anything")
}

func TestAdminBans_MalformedAuthorization(t *testing.T) {
	env := newAdminBanEnv(t)
	req := httptest.NewRequest("POST", "/api/ban", strings.NewReader(`{"credential":"vllm-1"}`))
	req.Header.Set("Authorization", "Basic abc")
	w := httptest.NewRecorder()
	env.router.ServeHTTP(w, req)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.False(t, env.f2b.IsBanned("vllm-1", "llama"))
}

func TestAdminBans_MethodNotAllowed(t *testing.T) {
	env := newAdminBanEnv(t)

	w := env.master("GET", "/api/ban", "")
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	assert.Equal(t, "POST", w.Header().Get("Allow"))

	w = env.master("GET", "/api/unban", "")
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)

	w = env.master("POST", "/api/bans", "{}")
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	assert.Equal(t, "GET", w.Header().Get("Allow"))
}

func TestAdminBan_WholeCredentialWithTTL(t *testing.T) {
	env := newAdminBanEnv(t)

	w := env.master("POST", "/api/ban", `{"credential":"vllm-1","ttl":"30m","reason":"air-scaler drain"}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	out := decodeBody(t, w)
	assert.Equal(t, "vllm-1", out["credential"])
	assert.Equal(t, "*", out["model"])
	assert.Equal(t, "admin", out["origin"])
	assert.Equal(t, "admin: air-scaler drain", out["reason"])
	assert.Equal(t, false, out["permanent"])
	until, err := time.Parse(time.RFC3339Nano, out["until"].(string))
	require.NoError(t, err)
	assert.WithinDuration(t, time.Now().Add(30*time.Minute), until, time.Minute)

	assert.True(t, env.f2b.IsBanned("vllm-1", "llama"))
	assert.True(t, env.f2b.IsBanned("vllm-1", "qwen"))
	assert.False(t, env.f2b.IsBanned("vllm-2", "llama"))
}

func TestAdminBan_SingleModel(t *testing.T) {
	env := newAdminBanEnv(t)

	w := env.master("POST", "/api/ban", `{"credential":"vllm-1","model":"llama","ttl":"5m"}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "llama", decodeBody(t, w)["model"])

	assert.True(t, env.f2b.IsBanned("vllm-1", "llama"))
	assert.False(t, env.f2b.IsBanned("vllm-1", "qwen"))
}

func TestAdminBan_WithoutTTLIsPermanent(t *testing.T) {
	env := newAdminBanEnv(t)

	w := env.master("POST", "/api/ban", `{"credential":"vllm-1"}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	out := decodeBody(t, w)
	assert.Equal(t, true, out["permanent"])
	assert.NotContains(t, out, "until")
	assert.Equal(t, "admin", out["reason"])
	assert.True(t, env.f2b.IsBanned("vllm-1", "llama"))
}

func TestAdminBan_Validation(t *testing.T) {
	env := newAdminBanEnv(t)

	cases := []struct {
		name string
		body string
		want int
	}{
		{"empty body", ``, http.StatusBadRequest},
		{"invalid json", `{`, http.StatusBadRequest},
		{"missing credential", `{"ttl":"1m"}`, http.StatusBadRequest},
		{"blank credential", `{"credential":"  "}`, http.StatusBadRequest},
		{"unparsable ttl", `{"credential":"vllm-1","ttl":"soon"}`, http.StatusBadRequest},
		{"negative ttl", `{"credential":"vllm-1","ttl":"-5m"}`, http.StatusBadRequest},
		{"zero ttl", `{"credential":"vllm-1","ttl":"0s"}`, http.StatusBadRequest},
		{"typo in field must not become a permanent ban", `{"credential":"vllm-1","ttl_seconds":60}`, http.StatusBadRequest},
		{"unknown credential", `{"credential":"nope","ttl":"1m"}`, http.StatusNotFound},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := env.master("POST", "/api/ban", tc.body)
			assert.Equal(t, tc.want, w.Code, w.Body.String())
		})
	}
	assert.Empty(t, env.f2b.GetActiveBans(), "rejected requests must not ban anything")
}

func TestAdminUnban(t *testing.T) {
	env := newAdminBanEnv(t)
	env.f2b.AdminBan("vllm-1", "", time.Hour, "wide")
	env.f2b.AdminBan("vllm-1", "llama", time.Hour, "narrow")

	t.Run("exact model keeps the wildcard ban", func(t *testing.T) {
		w := env.master("POST", "/api/unban", `{"credential":"vllm-1","model":"llama"}`)
		require.Equal(t, http.StatusOK, w.Code)
		assert.EqualValues(t, 1, decodeBody(t, w)["removed"])
		assert.True(t, env.f2b.IsBanned("vllm-1", "llama"))
	})

	t.Run("no model lifts everything", func(t *testing.T) {
		w := env.master("POST", "/api/unban", `{"credential":"vllm-1"}`)
		require.Equal(t, http.StatusOK, w.Code)
		assert.EqualValues(t, 1, decodeBody(t, w)["removed"])
		assert.False(t, env.f2b.IsBanned("vllm-1", "llama"))
	})

	t.Run("idempotent when nothing is banned", func(t *testing.T) {
		w := env.master("POST", "/api/unban", `{"credential":"vllm-1"}`)
		require.Equal(t, http.StatusOK, w.Code)
		assert.EqualValues(t, 0, decodeBody(t, w)["removed"])
	})

	t.Run("unknown credential is 404", func(t *testing.T) {
		w := env.master("POST", "/api/unban", `{"credential":"nope"}`)
		assert.Equal(t, http.StatusNotFound, w.Code)
	})

	t.Run("credential is required", func(t *testing.T) {
		w := env.master("POST", "/api/unban", `{}`)
		assert.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestAdminBans_List(t *testing.T) {
	env := newAdminBanEnv(t)

	w := env.master("GET", "/api/bans", "")
	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, `{"bans":[]}`, strings.TrimSpace(w.Body.String()), "empty list, not null")

	env.f2b.AdminBan("vllm-2", "", 0, "forever")
	env.f2b.AdminBan("vllm-1", "llama", time.Hour, "narrow")
	env.f2b.BanUntil("vllm-1", "qwen", 429, time.Now().Add(time.Hour), "quota")

	w = env.master("GET", "/api/bans", "")
	require.Equal(t, http.StatusOK, w.Code)
	var resp struct {
		Bans []proxy.AdminBanView `json:"bans"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Bans, 3)

	assert.Equal(t, "vllm-1", resp.Bans[0].Credential)
	assert.Equal(t, "llama", resp.Bans[0].Model)
	assert.Equal(t, "admin", resp.Bans[0].Origin)
	require.NotNil(t, resp.Bans[0].Until)

	assert.Equal(t, "qwen", resp.Bans[1].Model)
	assert.Equal(t, "fail2ban", resp.Bans[1].Origin)
	assert.Equal(t, 429, resp.Bans[1].ErrorCode)

	assert.Equal(t, "vllm-2", resp.Bans[2].Credential)
	assert.Equal(t, "*", resp.Bans[2].Model)
	assert.True(t, resp.Bans[2].Permanent)
	assert.Nil(t, resp.Bans[2].Until)
}

func TestAdminBan_ReflectedInHealth(t *testing.T) {
	env := newAdminBanEnv(t)

	env.master("POST", "/api/ban", `{"credential":"vllm-1","ttl":"1h","reason":"drain"}`)

	w := env.master("GET", "/health", "")
	var health struct {
		Credentials map[string]struct {
			IsBanned  bool       `json:"is_banned"`
			BanOrigin string     `json:"ban_origin"`
			BanReason string     `json:"ban_reason"`
			BanUntil  *time.Time `json:"ban_until"`
		} `json:"credentials"`
		Models map[string]struct {
			IsBanned  bool   `json:"is_banned"`
			BanOrigin string `json:"ban_origin"`
		} `json:"models"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &health), w.Body.String())

	c1 := health.Credentials["vllm-1"]
	assert.True(t, c1.IsBanned)
	assert.Equal(t, "admin", c1.BanOrigin)
	assert.Equal(t, "admin: drain", c1.BanReason)
	require.NotNil(t, c1.BanUntil)

	c2 := health.Credentials["vllm-2"]
	assert.False(t, c2.IsBanned)
	assert.Empty(t, c2.BanOrigin)
	assert.Nil(t, c2.BanUntil)

	for _, model := range []string{"llama", "qwen"} {
		m := health.Models["vllm-1:"+model]
		assert.True(t, m.IsBanned, model)
		assert.Equal(t, "admin", m.BanOrigin, model)
		assert.False(t, health.Models["vllm-2:"+model].IsBanned, model)
	}

	env.master("POST", "/api/unban", `{"credential":"vllm-1"}`)
	w = env.master("GET", "/health", "")
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &health))
	assert.False(t, health.Credentials["vllm-1"].IsBanned)
	assert.Empty(t, health.Credentials["vllm-1"].BanOrigin)
}

func TestAdminBan_ExactModelBanWinsOverWildcardInHealth(t *testing.T) {
	env := newAdminBanEnv(t)
	env.f2b.AdminBan("vllm-1", "", time.Hour, "wide")
	env.f2b.BanUntil("vllm-1", "llama", 429, time.Now().Add(time.Hour), "quota")

	w := env.master("GET", "/health", "")
	var health struct {
		Models map[string]struct {
			BanOrigin     string `json:"ban_origin"`
			ProviderError string `json:"provider_error"`
		} `json:"models"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &health))

	assert.Equal(t, "fail2ban", health.Models["vllm-1:llama"].BanOrigin)
	assert.Equal(t, "quota", health.Models["vllm-1:llama"].ProviderError)
	assert.Equal(t, "admin", health.Models["vllm-1:qwen"].BanOrigin)
}

func TestAdminBan_RoutingSkipsBannedCredential(t *testing.T) {
	env := newAdminBanEnv(t)

	env.master("POST", "/api/ban", `{"credential":"vllm-1","ttl":"1h"}`)
	for i := 0; i < 10; i++ {
		cred, err := env.bal.NextForModel("llama")
		require.NoError(t, err)
		assert.Equal(t, "vllm-2", cred.Name, "banned credential must not receive new requests")
	}

	env.master("POST", "/api/ban", `{"credential":"vllm-2","model":"llama"}`)
	_, err := env.bal.NextForModel("llama")
	assert.ErrorIs(t, err, balancer.ErrNoCredentialsAvailable)
	cred, err := env.bal.NextForModel("qwen")
	require.NoError(t, err)
	assert.Equal(t, "vllm-2", cred.Name, "the per-model ban leaves other models routable")

	env.master("POST", "/api/unban", `{"credential":"vllm-1"}`)
	cred, err = env.bal.NextForModel("llama")
	require.NoError(t, err)
	assert.Equal(t, "vllm-1", cred.Name)
}
