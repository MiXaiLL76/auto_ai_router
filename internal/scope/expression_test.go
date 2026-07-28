package scope

import (
	"fmt"
	"testing"
)

func TestContextAllowsExpression_AndRequirements(t *testing.T) {
	expression := And(
		FromScopes([]string{"team-a"}, nil),
		FromScopes([]string{"premium"}, nil),
	)

	if !NewContext([]string{"team-a", "premium"}, nil).AllowsExpression(expression) {
		t.Fatal("all requirement groups must match")
	}
	if NewContext([]string{"team-a"}, nil).AllowsExpression(expression) {
		t.Fatal("missing requirement group must reject")
	}
}

func TestContextAllowsExpression_OrAlternativesPreserveDenies(t *testing.T) {
	expression := Or(
		FromScopes([]string{"team-a"}, []string{"blocked"}),
		FromScopes([]string{"team-b"}, nil),
	)

	if NewContext([]string{"team-a", "blocked"}, nil).AllowsExpression(expression) {
		t.Fatal("path-specific deny must reject its alternative")
	}
	if !NewContext([]string{"team-b", "blocked"}, nil).AllowsExpression(expression) {
		t.Fatal("a different alternative must remain available")
	}
}

func TestContextAllowsExpression_FalseAndUnrestricted(t *testing.T) {
	ctx := PublicContext()

	if ctx.AllowsExpression(FalseExpression()) {
		t.Fatal("empty alternatives must be false")
	}
	if !ctx.AllowsExpression(FromScopes(nil, nil)) {
		t.Fatal("one empty alternative must be unrestricted")
	}
	if AdminContext().AllowsExpression(FalseExpression()) {
		t.Fatal("admin context cannot bypass an expression with no route")
	}
	if !AdminContext().AllowsExpression(FromScopes(nil, []string{"*"})) {
		t.Fatal("admin context must bypass scope constraints on an existing route")
	}
}

func TestContextAllowsExpression_WildcardRequiresScopedRequest(t *testing.T) {
	expression := FromScopes([]string{"*"}, nil)

	if PublicContext().AllowsExpression(expression) {
		t.Fatal("public context must not match wildcard scope")
	}
	if !NewContext([]string{"team-a"}, nil).AllowsExpression(expression) {
		t.Fatal("scoped request must match wildcard scope")
	}
	if NewContext([]string{"team-a"}, []string{"blocked"}).AllowsExpression(expression) {
		t.Fatal("request denied scopes must override wildcard scope")
	}
}

func TestExpressionLegacyProjection_ComplexExpressionFailsClosed(t *testing.T) {
	expression := And(
		FromScopes([]string{"team-a"}, nil),
		FromScopes([]string{"premium"}, nil),
	)

	scopes, deniedScopes := expression.LegacyProjection()

	if len(scopes) != 1 || scopes[0] == "" {
		t.Fatal("complex expression must project to a non-empty no-match scope")
	}
	if len(deniedScopes) != 0 {
		t.Fatal("complex expression must not use denied wildcard as a legacy marker")
	}
	if NewContext([]string{"*"}, nil).Allows(scopes, deniedScopes) {
		t.Fatal("wildcard request scope must not match the legacy no-route marker")
	}
}

func TestContextAllowsExpression_EmptyRequirementFailsClosed(t *testing.T) {
	for _, requirement := range [][][]string{{nil}, {{}}, {{"   "}}} {
		expression := &Expression{Alternatives: []Alternative{{Requirements: requirement}}}
		if PublicContext().AllowsExpression(expression) {
			t.Fatal("empty requirement must not become public")
		}
		if NewContext([]string{"*"}, nil).AllowsExpression(expression) {
			t.Fatal("wildcard scope must not match an empty requirement")
		}
		if AdminContext().AllowsExpression(expression) {
			t.Fatal("invalid alternative must not become an admin route")
		}
	}

	if !PublicContext().AllowsExpression(FromScopes([]string{"   "}, nil)) {
		t.Fatal("blank legacy scopes must retain unrestricted semantics")
	}
}

func TestNormalizeExpression_TooManyAlternativesFailsClosed(t *testing.T) {
	alternatives := make([]Alternative, maxExpressionAlternatives+1)
	for i := range alternatives {
		alternatives[i].Requirements = [][]string{{fmt.Sprintf("scope-%d", i)}}
	}

	expression := NormalizeExpression(&Expression{Alternatives: alternatives})
	if PublicContext().AllowsExpression(expression) || AdminContext().AllowsExpression(expression) {
		t.Fatal("oversized expression must not expose a route")
	}
}

func TestOr_DeduplicatesBeforeAlternativeLimit(t *testing.T) {
	expressions := make([]*Expression, maxExpressionAlternatives+1)
	for i := range expressions {
		expressions[i] = FromScopes([]string{"team-a"}, nil)
	}

	expression := Or(expressions...)
	if !NewContext([]string{"team-a"}, nil).AllowsExpression(expression) {
		t.Fatal("duplicate alternatives must not make an OR expression fail closed")
	}
	if len(expression.Alternatives) != 1 {
		t.Fatalf("expected one deduplicated alternative, got %d", len(expression.Alternatives))
	}
}

func TestOr_UnrestrictedAlternativeSimplifiesOversizedExpression(t *testing.T) {
	expressions := make([]*Expression, 0, maxExpressionAlternatives+2)
	for i := 0; i <= maxExpressionAlternatives; i++ {
		expressions = append(expressions, FromScopes([]string{fmt.Sprintf("scope-%d", i)}, nil))
	}
	// Keep the unrestricted path last to prove that the limit is applied only
	// after semantic simplification, independently of map/input order.
	expressions = append(expressions, FromScopes(nil, nil))

	expression := Or(expressions...)
	if !PublicContext().AllowsExpression(expression) {
		t.Fatal("an unrestricted alternative must keep the entire OR expression unrestricted")
	}
	if len(expression.Alternatives) != 1 || !isUnrestrictedAlternative(expression.Alternatives[0]) {
		t.Fatal("oversized expression must simplify to one unrestricted alternative")
	}
}
