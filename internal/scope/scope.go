package scope

import (
	"sort"
	"strconv"
	"strings"
)

const legacyNoMatchScope = "\x00air:no-matching-scope"

type Context struct {
	Allowed     map[string]struct{}
	Denied      map[string]struct{}
	Admin       bool
	credentials map[string]struct{}
}

func AdminContext() Context {
	return Context{Admin: true}
}

func PublicContext() Context {
	return Context{}
}

func NewContext(allowed, denied []string) Context {
	return Context{
		Allowed: toSet(allowed),
		Denied:  toSet(denied),
	}
}

func NormalizeList(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		normalized := Normalize(value)
		if normalized == "" {
			continue
		}
		if _, ok := seen[normalized]; ok {
			continue
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	return result
}

func Normalize(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func (c Context) Allows(credentialScopes, credentialDeniedScopes []string) bool {
	if hasLegacyNoMatchScope(credentialScopes) {
		return false
	}
	if c.Admin {
		return true
	}

	if hasWildcardList(credentialDeniedScopes) || hasWildcardSet(c.Denied) || c.intersects(credentialDeniedScopes) {
		return false
	}
	if intersectsSet(c.Denied, credentialScopes) {
		return false
	}
	if hasWildcardSet(c.Allowed) {
		return true
	}
	if len(credentialScopes) == 0 {
		return true
	}
	return c.intersects(credentialScopes)
}

func hasLegacyNoMatchScope(values []string) bool {
	for _, value := range values {
		if value == legacyNoMatchScope {
			return true
		}
	}
	return false
}

func (c Context) HasScopes() bool {
	return c.Admin || len(c.Allowed) > 0 || len(c.Denied) > 0 || c.HasCredentialRestriction()
}

func (c Context) Key() string {
	if c.HasCredentialRestriction() {
		names := setToList(c.credentials)
		c.credentials = nil
		return c.Key() + "|c:" + encodeKeyValues(names)
	}
	if c.Admin {
		return "admin"
	}
	allowed := c.AllowedList()
	denied := c.DeniedList()
	if len(allowed) == 0 && len(denied) == 0 {
		return "public"
	}
	return "a:" + encodeKeyValues(allowed) + "|d:" + encodeKeyValues(denied)
}

func encodeKeyValues(values []string) string {
	var key strings.Builder
	for _, value := range values {
		key.WriteString(strconv.Itoa(len(value)))
		key.WriteByte(':')
		key.WriteString(value)
	}
	return key.String()
}

func (c Context) AllowedList() []string {
	return setToList(c.Allowed)
}

func (c Context) DeniedList() []string {
	return setToList(c.Denied)
}

func (c Context) intersects(values []string) bool {
	return intersectsSet(c.Allowed, values)
}

func toSet(values []string) map[string]struct{} {
	normalized := NormalizeList(values)
	if len(normalized) == 0 {
		return nil
	}
	result := make(map[string]struct{}, len(normalized))
	for _, value := range normalized {
		result[value] = struct{}{}
	}
	return result
}

func intersectsSet(set map[string]struct{}, values []string) bool {
	if len(set) == 0 || len(values) == 0 {
		return false
	}
	if hasWildcardSet(set) {
		return true
	}
	for _, value := range values {
		normalized := Normalize(value)
		if normalized == "*" {
			return true
		}
		if _, ok := set[normalized]; ok {
			return true
		}
	}
	return false
}

func hasWildcardSet(set map[string]struct{}) bool {
	_, ok := set["*"]
	return ok
}

func hasWildcardList(values []string) bool {
	for _, value := range values {
		if Normalize(value) == "*" {
			return true
		}
	}
	return false
}

func setToList(set map[string]struct{}) []string {
	if len(set) == 0 {
		return nil
	}
	result := make([]string, 0, len(set))
	for value := range set {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
