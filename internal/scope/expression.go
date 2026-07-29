package scope

import (
	"slices"
	"strconv"
	"strings"
)

const maxExpressionAlternatives = 256

type Alternative struct {
	Requirements [][]string `json:"requirements,omitempty"`
	DeniedScopes []string   `json:"denied_scopes,omitempty"`
}

type Expression struct {
	Alternatives []Alternative `json:"alternatives"`
}

func FromScopes(scopes, deniedScopes []string) *Expression {
	normalizedScopes := NormalizeList(scopes)
	alternative := Alternative{DeniedScopes: NormalizeList(deniedScopes)}
	if len(normalizedScopes) > 0 {
		alternative.Requirements = [][]string{normalizedScopes}
	}
	return NormalizeExpression(&Expression{Alternatives: []Alternative{alternative}})
}

func FalseExpression() *Expression {
	return &Expression{Alternatives: []Alternative{}}
}

func NormalizeExpression(expression *Expression) *Expression {
	if expression == nil {
		return nil
	}
	return normalizeAlternatives(expression.Alternatives)
}

func normalizeAlternatives(groups ...[]Alternative) *Expression {
	// Hot path: CredentialConfig.ScopeExpression() calls into this on every
	// balancer credential-selection scan, and FromScopes (the overwhelmingly
	// common caller, invoked for both own and provider scopes) always passes
	// exactly one alternative. Handle that shape without allocating the
	// dedup map/slice at all — most credentials carry no scopes, so this
	// skips a 256-capacity map+slice pair per call in the common case.
	if len(groups) == 1 && len(groups[0]) == 1 {
		normalized, ok := normalizeAlternative(groups[0][0])
		if !ok {
			return &Expression{Alternatives: []Alternative{}}
		}
		if isUnrestrictedAlternative(normalized) {
			return unrestrictedExpression()
		}
		return &Expression{Alternatives: []Alternative{normalized}}
	}

	total := 0
	for _, group := range groups {
		total += len(group)
	}
	capHint := min(total, maxExpressionAlternatives)
	alternatives := make([]Alternative, 0, capHint)
	seen := make(map[string]struct{}, capHint)
	oversized := false
	for _, group := range groups {
		for _, alternative := range group {
			normalized, ok := normalizeAlternative(alternative)
			if !ok {
				continue
			}
			if isUnrestrictedAlternative(normalized) {
				return unrestrictedExpression()
			}
			if oversized {
				continue
			}
			key := alternativeKey(normalized)
			if _, exists := seen[key]; exists {
				continue
			}
			if len(alternatives) == maxExpressionAlternatives {
				// Continue scanning: a later unrestricted alternative still
				// simplifies the entire OR expression to true.
				oversized = true
				continue
			}
			seen[key] = struct{}{}
			alternatives = append(alternatives, normalized)
		}
	}
	if oversized {
		return FalseExpression()
	}
	return &Expression{Alternatives: alternatives}
}

func And(expressions ...*Expression) *Expression {
	result := &Expression{Alternatives: []Alternative{{}}}
	used := false
	for _, expression := range expressions {
		if expression == nil {
			continue
		}
		used = true
		normalized := NormalizeExpression(expression)
		if len(result.Alternatives) == 0 || len(normalized.Alternatives) == 0 {
			return FalseExpression()
		}
		if len(result.Alternatives) > maxExpressionAlternatives/len(normalized.Alternatives) {
			return FalseExpression()
		}
		combined := make([]Alternative, 0, len(result.Alternatives)*len(normalized.Alternatives))
		for _, left := range result.Alternatives {
			for _, right := range normalized.Alternatives {
				combined = append(combined, Alternative{
					Requirements: appendRequirements(left.Requirements, right.Requirements),
					DeniedScopes: appendScopes(left.DeniedScopes, right.DeniedScopes),
				})
			}
		}
		result = NormalizeExpression(&Expression{Alternatives: combined})
	}
	if !used {
		return nil
	}
	return result
}

func Or(expressions ...*Expression) *Expression {
	if len(expressions) == 0 {
		return FalseExpression()
	}
	groups := make([][]Alternative, 0, len(expressions))
	for _, expression := range expressions {
		if expression == nil {
			return unrestrictedExpression()
		}
		groups = append(groups, expression.Alternatives)
	}
	return normalizeAlternatives(groups...)
}

func (c Context) AllowsExpression(expression *Expression) bool {
	normalized := NormalizeExpression(expression)
	if normalized == nil {
		return true
	}
	if len(normalized.Alternatives) == 0 {
		return false
	}
	if c.Admin {
		return true
	}
	for _, alternative := range normalized.Alternatives {
		if c.allowsAlternative(alternative) {
			return true
		}
	}
	return false
}

func (c Context) allowsAlternative(alternative Alternative) bool {
	if !c.Allows(nil, alternative.DeniedScopes) {
		return false
	}
	for _, requirement := range alternative.Requirements {
		if !c.Allows(requirement, nil) {
			return false
		}
	}
	return true
}

func (e *Expression) LegacyProjection() ([]string, []string) {
	normalized := NormalizeExpression(e)
	if normalized == nil {
		return nil, nil
	}
	if len(normalized.Alternatives) == 0 {
		return legacyNoMatchScopes(), nil
	}
	for _, alternative := range normalized.Alternatives {
		if hasWildcardList(alternative.DeniedScopes) {
			return legacyNoMatchScopes(), nil
		}
	}
	if len(normalized.Alternatives) == 1 {
		alternative := normalized.Alternatives[0]
		switch len(alternative.Requirements) {
		case 0:
			return nil, append([]string(nil), alternative.DeniedScopes...)
		case 1:
			return append([]string(nil), alternative.Requirements[0]...), append([]string(nil), alternative.DeniedScopes...)
		default:
			return legacyNoMatchScopes(), nil
		}
	}

	deniedScopes := normalized.Alternatives[0].DeniedScopes
	scopes := make([]string, 0)
	for _, alternative := range normalized.Alternatives {
		if !slices.Equal(deniedScopes, alternative.DeniedScopes) || len(alternative.Requirements) > 1 {
			return legacyNoMatchScopes(), nil
		}
		if len(alternative.Requirements) == 0 {
			return nil, append([]string(nil), deniedScopes...)
		}
		scopes = append(scopes, alternative.Requirements[0]...)
	}
	return sortedNormalized(scopes), append([]string(nil), deniedScopes...)
}

func normalizeAlternative(alternative Alternative) (Alternative, bool) {
	deniedScopes := sortedNormalized(alternative.DeniedScopes)

	requirements := make([][]string, 0, len(alternative.Requirements))
	seen := make(map[string]struct{}, len(alternative.Requirements))
	for _, requirement := range alternative.Requirements {
		normalized := sortedNormalized(requirement)
		if len(normalized) == 0 {
			return Alternative{}, false
		}
		key := keyValues(normalized)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		requirements = append(requirements, normalized)
	}
	slices.SortFunc(requirements, func(a, b []string) int {
		return strings.Compare(keyValues(a), keyValues(b))
	})

	return Alternative{Requirements: requirements, DeniedScopes: deniedScopes}, true
}

func isUnrestrictedAlternative(alternative Alternative) bool {
	return len(alternative.Requirements) == 0 && len(alternative.DeniedScopes) == 0
}

func unrestrictedExpression() *Expression {
	return &Expression{Alternatives: []Alternative{{}}}
}

func appendRequirements(left, right [][]string) [][]string {
	result := make([][]string, 0, len(left)+len(right))
	for _, requirement := range left {
		result = append(result, append([]string(nil), requirement...))
	}
	for _, requirement := range right {
		result = append(result, append([]string(nil), requirement...))
	}
	return result
}

func appendScopes(left, right []string) []string {
	result := make([]string, 0, len(left)+len(right))
	result = append(result, left...)
	result = append(result, right...)
	return result
}

func sortedNormalized(values []string) []string {
	normalized := NormalizeList(values)
	slices.Sort(normalized)
	return normalized
}

func alternativeKey(alternative Alternative) string {
	var key strings.Builder
	for _, requirement := range alternative.Requirements {
		key.WriteString(keyValues(requirement))
		key.WriteByte('|')
	}
	key.WriteByte('!')
	key.WriteString(keyValues(alternative.DeniedScopes))
	return key.String()
}

func keyValues(values []string) string {
	var key strings.Builder
	for _, value := range values {
		key.WriteString(strconv.Itoa(len(value)))
		key.WriteByte(':')
		key.WriteString(value)
	}
	return key.String()
}

func legacyNoMatchScopes() []string {
	return []string{legacyNoMatchScope}
}
