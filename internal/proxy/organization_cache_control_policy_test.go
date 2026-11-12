package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type orgCacheControlTestDB struct {
	litellmdb.NoopManager
	tokens map[string]*dbmodels.TokenInfo
}

func (d *orgCacheControlTestDB) IsEnabled() bool           { return true }
func (d *orgCacheControlTestDB) IsHealthy() bool           { return true }
func (d *orgCacheControlTestDB) SpendLoggingEnabled() bool { return true }

func (d *orgCacheControlTestDB) ValidateToken(_ context.Context, rawToken string) (*dbmodels.TokenInfo, error) {
	info := d.tokens[rawToken]
	if info == nil {
		return nil, litellmdb.ErrTokenNotFound
	}
	return info.Clone(), nil
}

func (d *orgCacheControlTestDB) LogSpend(*dbmodels.SpendLogEntry) error { return nil }

// anthropicPassthroughUpstream starts a fake native-Anthropic upstream that
// answers with a minimal message and hands every request body it receives to
// the returned channel.
func anthropicPassthroughUpstream(t *testing.T) (*httptest.Server, <-chan map[string]any) {
	t.Helper()
	captured := make(chan map[string]any, 4)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		captured <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	}))
	t.Cleanup(upstream.Close)
	return upstream, captured
}

// newOrgCacheControlProxy builds a proxy with a single native-Anthropic
// credential (so InjectCacheControl's passthrough branch applies), an
// org-policy registry, and session stickiness + auto cache_control both on
// globally -- the preconditions an organization-level opt-out must override.
func newOrgCacheControlProxy(t *testing.T, upstreamURL string, db *orgCacheControlTestDB, policies []config.OrganizationPolicyConfig) *Proxy {
	t.Helper()
	credential := config.CredentialConfig{
		Name: "anthropic-direct", Type: config.ProviderTypeAnthropic, BaseURL: upstreamURL, APIKey: "provider-key", RPM: 100, TPM: 10000,
	}
	logger := testhelpers.NewTestLogger()
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "claude", Credential: credential.Name},
	})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})

	registry, err := routermodels.LoadOrganizationPolicies(policies, manager, routermodels.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: false,
	})
	require.NoError(t, err)

	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	builder.config.OrganizationPolicies = registry
	prx := builder.Build()
	prx.LiteLLMDB = db
	prx.stickyAutoCacheCtrl = true
	setTestModelPrice(prx, "claude", &routermodels.ModelPrice{InputCostPerToken: 0.001, OutputCostPerToken: 0.002})
	return prx
}

// sendCacheControlProbe sends a session-sticky-eligible /v1/messages request
// (a Session-Id header, so stickyCacheEligible is true regardless of the
// org-policy outcome under test) and returns the body the upstream received.
func sendCacheControlProbe(t *testing.T, prx *Proxy, captured <-chan map[string]any) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","system":"Be helpful.","messages":[{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("Session-Id", "session-1")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	select {
	case body := <-captured:
		return body
	default:
		t.Fatal("the upstream received no request")
		return nil
	}
}

func TestOrganizationPolicy_DisablesAutoCacheControlInjection(t *testing.T) {
	upstream, captured := anthropicPassthroughUpstream(t)
	db := &orgCacheControlTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-no-cache", OrganizationID: "org-no-cache"},
	}}
	prx := newOrgCacheControlProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:          "org-no-cache",
		DisableAutoCacheControl: true,
	}})

	body := sendCacheControlProbe(t, prx, captured)

	assert.Equal(t, "Be helpful.", body["system"],
		"the organization opted out, so system must stay a plain string with no cache_control block")
}

func TestOrganizationPolicy_AutoCacheControlInjectedWithoutOptOut(t *testing.T) {
	upstream, captured := anthropicPassthroughUpstream(t)
	db := &orgCacheControlTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-default", OrganizationID: "org-default"},
	}}
	prx := newOrgCacheControlProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID: "org-default",
	}})

	body := sendCacheControlProbe(t, prx, captured)

	system, ok := body["system"].([]any)
	require.True(t, ok, "system must be promoted to a content-block array carrying cache_control, got %#v", body["system"])
	require.Len(t, system, 1)
	block, ok := system[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, map[string]any{"type": "ephemeral"}, block["cache_control"])
}

func TestOrganizationPolicy_NoPolicyStillInjectsCacheControl(t *testing.T) {
	// A token with no organization at all (effectiveOrganizationPolicy returns
	// nil) must fall back to the global default: injection stays on.
	upstream, captured := anthropicPassthroughUpstream(t)
	db := &orgCacheControlTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1"},
	}}
	prx := newOrgCacheControlProxy(t, upstream.URL, db, nil)

	body := sendCacheControlProbe(t, prx, captured)

	system, ok := body["system"].([]any)
	require.True(t, ok, "system must carry cache_control when no organization policy applies, got %#v", body["system"])
	require.Len(t, system, 1)
	block, ok := system[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, map[string]any{"type": "ephemeral"}, block["cache_control"])
}
