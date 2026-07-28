package models

import (
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
)

func TestAggregateModelScopesFromHealth(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"team-a": {Scopes: []string{"team-a"}},
			"team-b": {Scopes: []string{"team-b"}},
		},
		Models: map[string]httputil.ModelHealthStats{
			"a": {Credential: "team-a", Model: "gpt-4"},
			"b": {Credential: "team-b", Model: "claude-3"},
		},
	}

	scopes := AggregateModelScopesFromHealth(health, false)

	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(scopes["gpt-4"].ScopeExpression))
	assert.False(t, scope.NewContext([]string{"team-b"}, nil).AllowsExpression(scopes["gpt-4"].ScopeExpression))
	assert.True(t, scope.NewContext([]string{"team-b"}, nil).AllowsExpression(scopes["claude-3"].ScopeExpression))
}

func TestAggregateProviderScopes_PreservesPathSpecificDeniedScopes(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"team-a": {Scopes: []string{"team-a"}, DeniedScopes: []string{"blocked"}},
		},
	}

	metadata := AggregateProviderScopesFromHealth(health, false)

	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(metadata.ScopeExpression))
	assert.False(t, scope.NewContext([]string{"team-a", "blocked"}, nil).AllowsExpression(metadata.ScopeExpression))
}

func TestAggregateProviderScopes_PreservesChainedRequirements(t *testing.T) {
	expression := scope.And(
		scope.FromScopes([]string{"team-a"}, nil),
		scope.FromScopes([]string{"premium"}, nil),
	)
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"chained": {ScopeExpression: expression},
		},
	}

	metadata := AggregateProviderScopesFromHealth(health, false)

	assert.True(t, scope.NewContext([]string{"team-a", "premium"}, nil).AllowsExpression(metadata.ScopeExpression))
	assert.False(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(metadata.ScopeExpression))
	assert.NotEmpty(t, metadata.Scopes)
}

func TestAggregateModelScopes_PrefersModelExpression(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"proxy": {ScopeExpression: scope.FromScopes([]string{"team-a", "team-b"}, nil)},
		},
		Models: map[string]httputil.ModelHealthStats{
			"claude": {
				Credential:      "proxy",
				Model:           "claude",
				ScopeExpression: scope.FromScopes([]string{"team-b"}, nil),
			},
		},
	}

	metadata := AggregateModelScopesFromHealth(health, false)["claude"]

	assert.False(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(metadata.ScopeExpression))
	assert.True(t, scope.NewContext([]string{"team-b"}, nil).AllowsExpression(metadata.ScopeExpression))
}

func TestAggregateProviderScopes_DistinguishesFalseAndUnrestricted(t *testing.T) {
	falseHealth := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"blocked": {ScopeExpression: scope.FalseExpression()},
		},
	}
	unrestrictedHealth := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"shared": {ScopeExpression: scope.FromScopes(nil, nil)},
		},
	}

	blocked := AggregateProviderScopesFromHealth(falseHealth, false)
	unrestricted := AggregateProviderScopesFromHealth(unrestrictedHealth, false)

	assert.False(t, scope.PublicContext().AllowsExpression(blocked.ScopeExpression))
	assert.True(t, scope.PublicContext().AllowsExpression(unrestricted.ScopeExpression))
}

func TestAggregateProviderScopes_ManyUnrestrictedCredentialsRemainVisible(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: make(map[string]httputil.CredentialHealthStats, 300),
	}
	for i := 0; i < 300; i++ {
		health.Credentials[fmt.Sprintf("credential-%d", i)] = httputil.CredentialHealthStats{}
	}

	metadata := AggregateProviderScopesFromHealth(health, false)

	assert.True(t, scope.PublicContext().AllowsExpression(metadata.ScopeExpression))
	assert.Len(t, metadata.ScopeExpression.Alternatives, 1)
}
