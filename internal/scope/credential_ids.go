package scope

import "regexp"

var credentialRoutingIDPattern = regexp.MustCompile(`^route-[0-9]{1,12}$`)

// WithoutCredentialRestriction preserves existing diagnostic visibility
// without correlating the routing IDs in metadata with account names.
func (c Context) WithoutCredentialRestriction() Context {
	c.credentialIDs = nil
	return c
}

// ValidCredentialRoutingID checks the syntax of provider-neutral routing labels.
func ValidCredentialRoutingID(id string) bool {
	return credentialRoutingIDPattern.MatchString(id)
}

// RestrictCredentialIDs narrows credential visibility without changing ordinary scopes.
// Calling it with no IDs explicitly removes access to all credentials.
func (c Context) RestrictCredentialIDs(ids []string) Context {
	allowed := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if c.AllowsCredentialID(id) {
			allowed[id] = struct{}{}
		}
	}
	c.credentialIDs = allowed
	return c
}

func (c Context) HasCredentialRestriction() bool {
	return c.credentialIDs != nil
}

func (c Context) AllowsCredentialID(id string) bool {
	if !c.HasCredentialRestriction() {
		return true
	}
	_, allowed := c.credentialIDs[id]
	return allowed
}
