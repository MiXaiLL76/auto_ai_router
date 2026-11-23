package scope

import "testing"

func TestCredentialRestrictionIntersectsAndDoesNotMutateParent(t *testing.T) {
	parent := NewContext([]string{"group"}, nil)
	first := parent.RestrictCredentials([]string{"A", "B"})
	second := first.RestrictCredentials([]string{"B", "C"})
	if parent.HasCredentialRestriction() || !parent.AllowsCredential("C") {
		t.Fatal("parent visibility was mutated")
	}
	if !first.AllowsCredential("A") || first.AllowsCredential("C") || first.AllowsCredential("a") {
		t.Fatal("credential identities must match exactly")
	}
	if second.AllowsCredential("A") || second.AllowsCredential("C") || !second.AllowsCredential("B") {
		t.Fatal("restrictions must intersect")
	}
	if parent.Key() == first.Key() || first.Key() == second.Key() {
		t.Fatal("visibility cache keys must distinguish restrictions")
	}
	if first.Key() != parent.RestrictCredentials([]string{"B", "A", "A"}).Key() {
		t.Fatal("equivalent restrictions need stable cache keys")
	}
}

func TestCredentialRestrictionEmptyDeniesEvenAdmin(t *testing.T) {
	for _, context := range []Context{PublicContext(), AdminContext()} {
		denied := context.RestrictCredentials(nil)
		if !denied.HasCredentialRestriction() || !denied.HasScopes() || denied.AllowsCredential("anything") || denied.Key() == context.Key() {
			t.Fatal("empty restriction must deny all, including for admin scopes")
		}
	}
}
