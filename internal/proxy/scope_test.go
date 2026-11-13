package proxy

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

type scopeTestDB struct {
	*litellmdb.NoopManager
	info *dbmodels.TokenInfo
}

func (db scopeTestDB) IsEnabled() bool           { return true }
func (db scopeTestDB) SpendLoggingEnabled() bool { return false }

func (db scopeTestDB) IsHealthy() bool { return true }

func (db scopeTestDB) ValidateToken(context.Context, string) (*dbmodels.TokenInfo, error) {
	return db.info, nil
}

func TestScopeContextFromTokenInfo_FallsBackToKeyName(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{KeyName: "Team-A"})

	if !ctx.Allows([]string{"team-a"}, nil) {
		t.Fatal("key_name must become the request scope")
	}
	if ctx.Allows([]string{"team-b"}, nil) {
		t.Fatal("different credential scope must be hidden")
	}
}

func TestScopeContextFromTokenInfo_MetadataScopesAndDeniedScopes(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		KeyName: "ignored",
		Metadata: map[string]interface{}{
			"air_scopes":        []interface{}{"team-b"},
			"air_denied_scopes": "premium,blocked",
		},
	})

	if !ctx.Allows([]string{"team-b"}, nil) {
		t.Fatal("metadata air_scopes must be used")
	}
	if ctx.Allows([]string{"ignored"}, nil) {
		t.Fatal("metadata scopes must override key_name fallback")
	}
	if ctx.Allows([]string{"team-b", "premium"}, nil) {
		t.Fatal("metadata denied scopes must override allow")
	}
}

func TestScopeContextFromTokenInfo_ForbiddenScopesAlias(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		Metadata: map[string]interface{}{
			"air_scopes":           []interface{}{"team-a"},
			"air_forbidden_scopes": []interface{}{"blocked"},
		},
	})

	if ctx.Allows([]string{"team-a", "blocked"}, nil) {
		t.Fatal("air_forbidden_scopes must deny matching credentials")
	}
}

func TestScopeContextFromTokenInfo_TeamMetadataGrantsScope(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		KeyName: "ignored",
		TeamMetadata: map[string]interface{}{
			"air_scopes": []interface{}{"yandex_search"},
		},
	})

	if !ctx.Allows([]string{"yandex_search"}, nil) {
		t.Fatal("a scope granted at team level must apply even with no key-level air_scopes")
	}
}

func TestScopeContextFromTokenInfo_OrganizationMetadataGrantsScope(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		OrganizationMetadata: map[string]interface{}{
			"air_scopes": []interface{}{"yandex_search"},
		},
	})

	if !ctx.Allows([]string{"yandex_search"}, nil) {
		t.Fatal("a scope granted at organization level must apply")
	}
}

func TestScopeContextFromTokenInfo_KeyTeamOrgScopesUnion(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		Metadata:             map[string]interface{}{"air_scopes": []interface{}{"from-key"}},
		TeamMetadata:         map[string]interface{}{"air_scopes": []interface{}{"from-team"}},
		OrganizationMetadata: map[string]interface{}{"air_scopes": []interface{}{"from-org"}},
	})

	for _, want := range []string{"from-key", "from-team", "from-org"} {
		if !ctx.Allows([]string{want}, nil) {
			t.Fatalf("scope %q from any of key/team/org metadata must be allowed", want)
		}
	}
}

func TestScopeContextFromTokenInfo_TeamDeniedScopeAppliesEvenIfKeyAllows(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		Metadata:     map[string]interface{}{"air_scopes": []interface{}{"yandex_search"}},
		TeamMetadata: map[string]interface{}{"air_denied_scopes": []interface{}{"yandex_search"}},
	})

	if ctx.Allows([]string{"yandex_search"}, nil) {
		t.Fatal("a team-level denial must win over a key-level allow")
	}
}

func TestScopeContextFromTokenInfo_NoScopesAnywhereKeepsKeyNameFallback(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		KeyName:              "Team-A",
		TeamMetadata:         map[string]interface{}{},
		OrganizationMetadata: map[string]interface{}{},
	})

	if !ctx.Allows([]string{"team-a"}, nil) {
		t.Fatal("with no air_scopes set anywhere, the key_name fallback must still apply unchanged")
	}
}

func TestScopeContextFromTokenInfo_LiteLLMMasterKeyIsAdmin(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		KeyName:     liteLLMMasterKeyIdentity,
		UserID:      liteLLMMasterKeyIdentity,
		IsMasterKey: true,
	})

	if !ctx.Admin {
		t.Fatal("LiteLLM master key token info must become admin context")
	}
	if !ctx.Allows([]string{"private"}, []string{"private"}) {
		t.Fatal("admin context must bypass credential scope filters")
	}
}

func TestScopeContextFromTokenInfo_MasterUserIDDoesNotGrantAdmin(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		KeyName: "regular-key",
		UserID:  liteLLMMasterKeyIdentity,
	})

	if ctx.Admin {
		t.Fatal("a user-controlled user_id must not grant admin visibility")
	}
	if !ctx.Allows([]string{"regular-key"}, nil) {
		t.Fatal("a regular token with the reserved user_id must retain its normal key scope")
	}
}

func TestScopeContextForRequest_LiteLLMMasterKeyFromDBIsAdmin(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithMasterKey("config-master").
		Build()
	prx.LiteLLMDB = scopeTestDB{
		NoopManager: litellmdb.NewNoopManager(),
		info: &dbmodels.TokenInfo{
			KeyName:     liteLLMMasterKeyIdentity,
			UserID:      liteLLMMasterKeyIdentity,
			IsMasterKey: true,
		},
	}

	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Set("Authorization", "Bearer db-master")

	ctx, err := prx.ScopeContextForRequest(req)
	if err != nil {
		t.Fatalf("ScopeContextForRequest returned error: %v", err)
	}
	if !ctx.Admin {
		t.Fatal("DB-loaded LiteLLM master key must use admin visibility")
	}
}

func TestScopeContextForRequestAcceptsXAPIKey(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	prx.LiteLLMDB = scopeTestDB{
		NoopManager: litellmdb.NewNoopManager(),
		info: &dbmodels.TokenInfo{
			KeyName: "private",
		},
	}

	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Set("x-api-key", "virtual-key")

	ctx, err := prx.ScopeContextForRequest(req)
	if err != nil {
		t.Fatalf("ScopeContextForRequest returned error: %v", err)
	}
	if !ctx.Allows([]string{"private"}, nil) {
		t.Fatal("x-api-key must produce the same visibility scope as Bearer auth")
	}
}
