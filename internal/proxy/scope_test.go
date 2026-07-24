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

func (db scopeTestDB) IsEnabled() bool { return true }

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

func TestScopeContextFromTokenInfo_LiteLLMMasterKeyIsAdmin(t *testing.T) {
	ctx := scopeContextFromTokenInfo(&dbmodels.TokenInfo{
		KeyName: liteLLMMasterKeyIdentity,
		UserID:  liteLLMMasterKeyIdentity,
	})

	if !ctx.Admin {
		t.Fatal("LiteLLM master key token info must become admin context")
	}
	if !ctx.Allows([]string{"private"}, []string{"private"}) {
		t.Fatal("admin context must bypass credential scope filters")
	}
}

func TestScopeContextForRequest_LiteLLMMasterKeyFromDBIsAdmin(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithMasterKey("config-master").
		Build()
	prx.LiteLLMDB = scopeTestDB{
		NoopManager: litellmdb.NewNoopManager(),
		info: &dbmodels.TokenInfo{
			KeyName: liteLLMMasterKeyIdentity,
			UserID:  liteLLMMasterKeyIdentity,
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
