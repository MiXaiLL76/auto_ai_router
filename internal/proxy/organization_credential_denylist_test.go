package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type denylistHostRewriteTransport struct {
	targets map[string]*url.URL
}

func (transport denylistHostRewriteTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	target := transport.targets[request.URL.Hostname()]
	if target == nil {
		return http.DefaultTransport.RoundTrip(request)
	}
	clone := request.Clone(request.Context())
	clone.URL.Scheme = target.Scheme
	clone.URL.Host = target.Host
	clone.Host = target.Host
	return http.DefaultTransport.RoundTrip(clone)
}

func setDenylistHostRewrite(t *testing.T, proxy *Proxy, host, target string) {
	t.Helper()
	parsed, err := url.Parse(target)
	require.NoError(t, err)
	proxy.client.Transport = denylistHostRewriteTransport{targets: map[string]*url.URL{host: parsed}}
}

func denylistProvider(t *testing.T, calls *atomic.Int32, receivedHeader *atomic.Bool) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Header.Get(HeaderAIRCredentialDenylist) != "" {
			receivedHeader.Store(true)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse(
			"chatcmpl-denylist",
			"route-a",
			"ok",
		))
	}))
}

func directDenylistCredential(name, baseURL string) config.CredentialConfig {
	return config.CredentialConfig{
		Name: name, Type: config.ProviderTypeOpenAI, BaseURL: baseURL,
		APIKey: "provider-key", RPM: -1, TPM: -1,
	}
}

func newDenylistRootProxy(
	t *testing.T,
	leafURL string,
	db *organizationPolicyTestDB,
	denylist []string,
) *Proxy {
	t.Helper()
	credential := proxyCred("leaf-router", leafURL, 1)
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{{
		Name: "route-a", Credential: credential.Name,
	}})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})
	registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID:     "org-1",
		PriceProfileID:     "profile-1",
		ModelPricesLink:    writeProxyPolicyPrices(t, `{"route-a":{"input_cost_per_token":0.001,"output_cost_per_token":0.001}}`),
		CredentialDenylist: denylist,
	}}, manager, routermodels.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: false,
	})
	require.NoError(t, err)

	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("root-master-key")
	builder.config.ModelManager = manager
	builder.config.OrganizationPolicies = registry
	root := builder.Build()
	root.LiteLLMDB = db
	return root
}

func newLocalDenylistPolicyProxy(
	t *testing.T,
	credentials []config.CredentialConfig,
	db *organizationPolicyTestDB,
	denylist []string,
) *Proxy {
	t.Helper()
	models := make([]config.ModelRPMConfig, 0, len(credentials))
	for _, credential := range credentials {
		models = append(models, config.ModelRPMConfig{Name: "route-a", Credential: credential.Name})
	}
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, models)
	manager.LoadModelsFromConfig(credentials)
	manager.SetCredentials(credentials)
	registry, err := routermodels.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID:     "org-1",
		PriceProfileID:     "profile-1",
		ModelPricesLink:    writeProxyPolicyPrices(t, `{"route-a":{"input_cost_per_token":0.001,"output_cost_per_token":0.001}}`),
		CredentialDenylist: denylist,
	}}, manager, routermodels.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: false,
	})
	require.NoError(t, err)

	builder := NewTestProxyBuilder().WithCredentials(credentials...).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	builder.config.OrganizationPolicies = registry
	proxy := builder.Build()
	proxy.LiteLLMDB = db
	return proxy
}

func TestOrganizationCredentialDenylistIsOrganizationScopedAndAppliesLocally(t *testing.T) {
	var deniedCalls, allowedCalls atomic.Int32
	var deniedHeader, allowedHeader atomic.Bool
	deniedProvider := denylistProvider(t, &deniedCalls, &deniedHeader)
	defer deniedProvider.Close()
	allowedProvider := denylistProvider(t, &allowedCalls, &allowedHeader)
	defer allowedProvider.Close()
	deniedCredential := directDenylistCredential("denied-provider", deniedProvider.URL)
	deniedCredential.Weight = 100
	credentials := []config.CredentialConfig{
		deniedCredential,
		directDenylistCredential("allowed-provider", allowedProvider.URL),
	}
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"restricted-token": {
			Token: "restricted-hash", UserID: "restricted-user",
			DirectOrganizationID: "org-1", OrganizationID: "org-1",
		},
		"control-token": {
			Token: "control-hash", UserID: "control-user",
			DirectOrganizationID: "org-2", OrganizationID: "org-2",
		},
	}}
	proxy := newLocalDenylistPolicyProxy(t, credentials, db, []string{"denied-provider"})
	setTestModelPrice(proxy, "route-a", &routermodels.ModelPrice{})

	request := func(token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
			`{"model":"route-a","messages":[]}`,
		))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		proxy.ProxyRequest(w, req)
		return w
	}

	require.Equal(t, http.StatusOK, request("restricted-token").Code)
	assert.Zero(t, deniedCalls.Load())
	assert.Equal(t, int32(1), allowedCalls.Load())

	require.Equal(t, http.StatusOK, request("control-token").Code)
	assert.Equal(t, int32(1), deniedCalls.Load())
}

func TestOrganizationCredentialDenylistPropagatesAndKeepsRouterEligible(t *testing.T) {
	var deniedCalls, allowedCalls atomic.Int32
	var deniedHeader, allowedHeader atomic.Bool
	deniedProvider := denylistProvider(t, &deniedCalls, &deniedHeader)
	defer deniedProvider.Close()
	allowedProvider := denylistProvider(t, &allowedCalls, &allowedHeader)
	defer allowedProvider.Close()

	deniedCredential := directDenylistCredential("denied-provider", deniedProvider.URL)
	allowedCredential := directDenylistCredential("allowed-provider", allowedProvider.URL)
	leaf := NewTestProxyBuilder().
		WithCredentials(deniedCredential, allowedCredential).
		WithMasterKey("master-key").
		Build()
	registerTestModel(leaf, deniedCredential.Name, "route-a")
	registerTestModel(leaf, allowedCredential.Name, "route-a")
	leafServer := newIPv4Server(t, http.HandlerFunc(leaf.ProxyRequest))
	defer leafServer.Close()
	middleCredential := proxyCred("leaf-router", "http://air-leaf.production", 1)
	middle := NewTestProxyBuilder().
		WithCredentials(middleCredential).
		WithMasterKey("master-key").
		Build()
	registerTestModel(middle, middleCredential.Name, "route-a")
	setDenylistHostRewrite(t, middle, "air-leaf.production", leafServer.URL)
	middleServer := newIPv4Server(t, http.HandlerFunc(middle.ProxyRequest))
	defer middleServer.Close()

	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"org-token": {
			Token: "token-hash", UserID: "user-1",
			DirectOrganizationID: "org-1", OrganizationID: "org-1",
		},
	}}
	root := newDenylistRootProxy(t, "http://air-middle", db, []string{"denied-provider", "leaf-router"})
	setDenylistHostRewrite(t, root, "air-middle", middleServer.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[{"role":"user","content":"test"}]}`,
	))
	req.Header.Set("Authorization", "Bearer org-token")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	root.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Zero(t, deniedCalls.Load())
	assert.Equal(t, int32(1), allowedCalls.Load())
	assert.False(t, deniedHeader.Load())
	assert.False(t, allowedHeader.Load())
}

func TestOrganizationCredentialDenylistRejectsAllProviderCandidates(t *testing.T) {
	var calls atomic.Int32
	var receivedHeader atomic.Bool
	provider := denylistProvider(t, &calls, &receivedHeader)
	defer provider.Close()
	credential := directDenylistCredential("provider", provider.URL)
	leaf := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").Build()
	registerTestModel(leaf, credential.Name, "route-a")

	header, err := json.Marshal([]string{"provider"})
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Set(HeaderAIRCredentialDenylist, string(header))
	w := httptest.NewRecorder()
	leaf.ProxyRequest(w, req)

	assert.Equal(t, http.StatusTooManyRequests, w.Code)
	assert.Zero(t, calls.Load())
	assert.False(t, receivedHeader.Load())
}

func TestOrganizationCredentialDenylistIgnoresUntrustedHeader(t *testing.T) {
	var calls atomic.Int32
	var receivedHeader atomic.Bool
	provider := denylistProvider(t, &calls, &receivedHeader)
	defer provider.Close()
	credential := directDenylistCredential("provider", provider.URL)
	db := &organizationPolicyTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"virtual-key": {Token: "token-hash", UserID: "user-1"},
	}}
	leaf := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").Build()
	leaf.LiteLLMDB = db
	registerTestModel(leaf, credential.Name, "route-a")
	setTestModelPrice(leaf, "route-a", &routermodels.ModelPrice{})

	for _, testCase := range []struct {
		name   string
		token  string
		marker bool
	}{
		{name: "virtual key with marker", token: "virtual-key", marker: true},
		{name: "master key without marker", token: "master-key", marker: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"route-a","messages":[]}`,
			))
			req.Header.Set("Authorization", "Bearer "+testCase.token)
			req.Header.Set("Content-Type", "application/json")
			if testCase.marker {
				req.Header.Set(HeaderAIRProxyClient, "1")
			}
			req.Header.Set(HeaderAIRCredentialDenylist, `["provider"]`)
			w := httptest.NewRecorder()
			leaf.ProxyRequest(w, req)
			require.Equal(t, http.StatusOK, w.Code)
		})
	}
	assert.Equal(t, int32(2), calls.Load())
	assert.False(t, receivedHeader.Load())
}

func TestOrganizationCredentialDenylistRejectsMalformedTrustedHeader(t *testing.T) {
	var calls atomic.Int32
	var receivedHeader atomic.Bool
	provider := denylistProvider(t, &calls, &receivedHeader)
	defer provider.Close()
	credential := directDenylistCredential("provider", provider.URL)
	leaf := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").Build()
	registerTestModel(leaf, credential.Name, "route-a")
	var logs bytes.Buffer
	leaf.logger = slog.New(slog.NewJSONHandler(&logs, nil))

	const privateHeaderValue = `{"private-review-marker":"not-a-list"}`
	for _, value := range []string{privateHeaderValue, strings.Repeat("x", maxCredentialDenylistBytes+1)} {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
			`{"model":"route-a","messages":[]}`,
		))
		req.Header.Set("Authorization", "Bearer master-key")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set(HeaderAIRProxyClient, "1")
		req.Header.Set(HeaderAIRCredentialDenylist, value)
		w := httptest.NewRecorder()
		leaf.ProxyRequest(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Add(HeaderAIRCredentialDenylist, `["provider"]`)
	req.Header.Add(HeaderAIRCredentialDenylist, `["other-provider"]`)
	w := httptest.NewRecorder()
	leaf.ProxyRequest(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)

	assert.Zero(t, calls.Load())
	assert.False(t, receivedHeader.Load())
	assert.Contains(t, logs.String(), "Rejected invalid internal routing policy")
	assert.NotContains(t, logs.String(), "private-review-marker")
}

func TestParseCredentialDenylistRejectsInvalidValues(t *testing.T) {
	oversizedName, err := json.Marshal([]string{strings.Repeat("x", maxCredentialNameBytes+1)})
	require.NoError(t, err)

	for _, value := range []string{
		`{"not":"a-list"}`,
		`["provider"] trailing`,
		`[" provider"]`,
		`["provider\n"]`,
		string(oversizedName),
	} {
		_, err := parseCredentialDenylist(value)
		require.ErrorIs(t, err, errInvalidCredentialDenylist)
	}

	parsed, err := parseCredentialDenylist(`["provider","provider"]`)
	require.NoError(t, err)
	assert.Equal(t, []string{"provider"}, parsed)
}

func TestCredentialDenylistCodecLimits(t *testing.T) {
	names := make([]string, maxCredentialDenylistEntries)
	for index := range names {
		names[index] = fmt.Sprintf("credential-%04d", index)
	}
	header := make(http.Header)
	require.NoError(t, setCredentialDenylistHeader(header, names))
	parsed, err := parseCredentialDenylist(header.Get(HeaderAIRCredentialDenylist))
	require.NoError(t, err)
	assert.Equal(t, names, parsed)

	overflow := append(names, "credential-overflow")
	header = make(http.Header)
	require.ErrorIs(t, setCredentialDenylistHeader(header, overflow), errInvalidCredentialDenylist)
	assert.Empty(t, header.Get(HeaderAIRCredentialDenylist))
	encoded, err := json.Marshal(overflow)
	require.NoError(t, err)
	_, err = parseCredentialDenylist(string(encoded))
	require.ErrorIs(t, err, errInvalidCredentialDenylist)

	large := make([]string, 300)
	for index := range large {
		prefix := fmt.Sprintf("%03d-", index)
		large[index] = prefix + strings.Repeat("x", maxCredentialNameBytes-len(prefix))
	}
	header = make(http.Header)
	require.Error(t, setCredentialDenylistHeader(header, large))
	assert.Empty(t, header.Get(HeaderAIRCredentialDenylist))
}

func TestOrganizationCredentialDenylistOverridesSessionBinding(t *testing.T) {
	var deniedCalls, allowedCalls atomic.Int32
	var deniedHeader, allowedHeader atomic.Bool
	deniedProvider := denylistProvider(t, &deniedCalls, &deniedHeader)
	defer deniedProvider.Close()
	allowedProvider := denylistProvider(t, &allowedCalls, &allowedHeader)
	defer allowedProvider.Close()

	deniedCredential := directDenylistCredential("denied-provider", deniedProvider.URL)
	allowedCredential := directDenylistCredential("allowed-provider", allowedProvider.URL)
	leaf := NewTestProxyBuilder().
		WithCredentials(deniedCredential, allowedCredential).
		WithMasterKey("master-key").
		WithSessionSticky(5 * time.Minute).
		Build()
	registerTestModel(leaf, deniedCredential.Name, "route-a")
	registerTestModel(leaf, allowedCredential.Name, "route-a")
	leaf.sessionStore.Set("session-1", "route-a", deniedCredential.Name)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[],"user":"session-1"}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Set(HeaderAIRCredentialDenylist, `["denied-provider"]`)
	w := httptest.NewRecorder()
	leaf.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Zero(t, deniedCalls.Load())
	assert.Equal(t, int32(1), allowedCalls.Load())
}

func TestOrganizationCredentialDenylistSurvivesRetry(t *testing.T) {
	var failingCalls, deniedCalls, allowedCalls atomic.Int32
	failingProvider := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		failingCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"failed"}}`))
	}))
	defer failingProvider.Close()
	var deniedHeader, allowedHeader atomic.Bool
	deniedProvider := denylistProvider(t, &deniedCalls, &deniedHeader)
	defer deniedProvider.Close()
	allowedProvider := denylistProvider(t, &allowedCalls, &allowedHeader)
	defer allowedProvider.Close()

	failingCredential := directDenylistCredential("failing-provider", failingProvider.URL)
	failingCredential.Weight = 100
	deniedCredential := directDenylistCredential("denied-provider", deniedProvider.URL)
	allowedCredential := directDenylistCredential("allowed-provider", allowedProvider.URL)
	leaf := NewTestProxyBuilder().
		WithCredentials(failingCredential, deniedCredential, allowedCredential).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()
	for _, credential := range []config.CredentialConfig{failingCredential, deniedCredential, allowedCredential} {
		registerTestModel(leaf, credential.Name, "route-a")
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Set(HeaderAIRCredentialDenylist, `["denied-provider"]`)
	w := httptest.NewRecorder()
	leaf.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), failingCalls.Load())
	assert.Zero(t, deniedCalls.Load())
	assert.Equal(t, int32(1), allowedCalls.Load())
}

func TestOrganizationCredentialDenylistAllowsFallback(t *testing.T) {
	var deniedCalls, fallbackCalls atomic.Int32
	var deniedHeader, fallbackHeader atomic.Bool
	deniedProvider := denylistProvider(t, &deniedCalls, &deniedHeader)
	defer deniedProvider.Close()
	fallbackProvider := denylistProvider(t, &fallbackCalls, &fallbackHeader)
	defer fallbackProvider.Close()

	deniedCredential := directDenylistCredential("denied-provider", deniedProvider.URL)
	fallbackCredential := directDenylistCredential("fallback-provider", fallbackProvider.URL)
	fallbackCredential.IsFallback = true
	leaf := NewTestProxyBuilder().
		WithCredentials(deniedCredential, fallbackCredential).
		WithMasterKey("master-key").
		Build()
	registerTestModel(leaf, deniedCredential.Name, "route-a")
	registerTestModel(leaf, fallbackCredential.Name, "route-a")

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Set(HeaderAIRCredentialDenylist, `["denied-provider"]`)
	w := httptest.NewRecorder()
	leaf.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Zero(t, deniedCalls.Load())
	assert.Equal(t, int32(1), fallbackCalls.Load())
}

func TestOrganizationCredentialDenylistDoesNotReachGenericProxy(t *testing.T) {
	var calls atomic.Int32
	var receivedHeader atomic.Bool
	provider := denylistProvider(t, &calls, &receivedHeader)
	defer provider.Close()
	credential := proxyCred("generic-proxy", provider.URL, 1)
	leaf := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").Build()
	registerTestModel(leaf, credential.Name, "route-a")

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderAIRProxyClient, "1")
	req.Header.Set(HeaderAIRCredentialDenylist, `["other-provider"]`)
	w := httptest.NewRecorder()
	leaf.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), calls.Load())
	assert.False(t, receivedHeader.Load())
}
