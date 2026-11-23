package scope

// RestrictCredentials returns a copy with an additional mandatory credential
// allowlist. An empty list denies all credentials; a nil restriction in Context
// means unrestricted. Restrictions intersect and are not bypassed by Admin.
func (c Context) RestrictCredentials(names []string) Context {
	allowed := make(map[string]struct{}, len(names))
	for _, name := range names {
		if c.AllowsCredential(name) {
			allowed[name] = struct{}{}
		}
	}
	c.credentials = allowed
	return c
}

func (c Context) HasCredentialRestriction() bool {
	return c.credentials != nil
}

func (c Context) AllowsCredential(name string) bool {
	if c.credentials == nil {
		return true
	}
	_, allowed := c.credentials[name]
	return allowed
}
