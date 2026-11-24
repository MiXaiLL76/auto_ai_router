package scope

import "testing"

func TestContextAllows_DefaultCredentialVisibleToEveryone(t *testing.T) {
	ctx := PublicContext()

	if !ctx.Allows(nil, nil) {
		t.Fatal("unscoped credential must be visible to public context")
	}
}

func TestContextAllows_AllowAndDeny(t *testing.T) {
	ctx := NewContext([]string{"Team-A"}, nil)

	if !ctx.Allows([]string{"team-a"}, nil) {
		t.Fatal("matching allow scope must pass")
	}
	if ctx.Allows([]string{"team-b"}, nil) {
		t.Fatal("non-matching allow scope must not pass")
	}
	if ctx.Allows([]string{"team-a"}, []string{"team-a"}) {
		t.Fatal("credential deny scope must override allow")
	}
}

func TestContextAllows_RequestDeniedScopesOverrideCredentialAllow(t *testing.T) {
	ctx := NewContext([]string{"team-a"}, []string{"premium"})

	if ctx.Allows([]string{"team-a", "premium"}, nil) {
		t.Fatal("request denied scope must override credential allow")
	}
}

func TestContextAllows_AdminBypassesScopes(t *testing.T) {
	ctx := AdminContext()

	if !ctx.Allows([]string{"private"}, []string{"private"}) {
		t.Fatal("admin context must bypass scope rules")
	}
}

func TestContextKey_DistinguishesDelimitedScopeValues(t *testing.T) {
	separate := NewContext([]string{"a", "b"}, nil)
	combined := NewContext([]string{"a,b"}, nil)

	if separate.Key() == combined.Key() {
		t.Fatal("different scope sets must not share a cache key")
	}
}

func TestCredentialIDRestriction(t *testing.T) {
	public := PublicContext()
	empty := public.RestrictCredentialIDs(nil)
	restricted := public.RestrictCredentialIDs([]string{"route-02", "route-01"})
	narrowed := restricted.RestrictCredentialIDs([]string{"route-02", "route-03"})
	if !public.AllowsCredentialID("") || public.HasCredentialRestriction() {
		t.Fatal("ordinary visibility must remain unrestricted")
	}
	if empty.AllowsCredentialID("route-01") || !empty.HasCredentialRestriction() || !empty.HasScopes() {
		t.Fatal("an empty restriction must deny all credentials")
	}
	if !narrowed.AllowsCredentialID("route-02") || narrowed.AllowsCredentialID("route-01") ||
		narrowed.AllowsCredentialID("route-03") || !restricted.AllowsCredentialID("route-01") {
		t.Fatal("narrowing must intersect the restrictions without mutating the original context")
	}
	if public.Key() == empty.Key() || restricted.Key() == narrowed.Key() {
		t.Fatal("contexts with different restrictions must have distinct cache keys")
	}
	if restricted.Key() != public.RestrictCredentialIDs([]string{"route-01", "route-02"}).Key() {
		t.Fatal("equivalent restrictions must have the same cache key")
	}
}
