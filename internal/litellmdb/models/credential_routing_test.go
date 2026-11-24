package models

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAllowedCredentialRoutingIDs(t *testing.T) {
	metadata := func(value interface{}) map[string]interface{} {
		return map[string]interface{}{"allowed_cred": value}
	}

	for _, tc := range []struct {
		name       string
		info       *TokenInfo
		wantIDs    []string
		restricted bool
	}{
		{name: "nil token"},
		{
			name: "missing field leaves normal routing",
			info: &TokenInfo{Metadata: map[string]interface{}{"air_scopes": []string{"existing"}}},
		},
		{
			name:       "team restriction is inherited",
			info:       &TokenInfo{TeamMetadata: metadata([]interface{}{"route-7004", "route-7001", "route-7004"})},
			wantIDs:    []string{"route-7001", "route-7004"},
			restricted: true,
		},
		{
			name: "all present restrictions intersect",
			info: &TokenInfo{
				OrganizationMetadata: metadata([]string{"route-7001", "route-7002"}),
				TeamMetadata:         metadata([]interface{}{"route-7001", "route-7003"}),
				Metadata:             metadata([]interface{}{"route-7001", "route-7004"}),
			},
			wantIDs:    []string{"route-7001"},
			restricted: true,
		},
		{
			name: "disjoint lists deny all",
			info: &TokenInfo{
				OrganizationMetadata: metadata([]string{"route-7001"}),
				Metadata:             metadata([]string{"route-7004"}),
			},
			restricted: true,
		},
		{
			name: "explicit empty list blocks a key despite org authorization",
			info: &TokenInfo{
				OrganizationMetadata: metadata([]string{"route-7001"}),
				Metadata:             metadata([]interface{}{}),
			},
			restricted: true,
		},
		{
			name: "organization null blocks descendants",
			info: &TokenInfo{
				OrganizationMetadata: metadata(nil),
				Metadata:             metadata([]string{"route-7001"}),
			},
			restricted: true,
		},
		{
			name:       "wrong metadata type denies all",
			info:       &TokenInfo{Metadata: metadata("route-7001")},
			restricted: true,
		},
		{
			name:       "partly malformed list denies all",
			info:       &TokenInfo{TeamMetadata: metadata([]interface{}{"route-7001", 123})},
			restricted: true,
		},
		{
			name:       "invalid ID denies all",
			info:       &TokenInfo{OrganizationMetadata: metadata([]string{"route-7001", "route-1*"})},
			restricted: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ids, restricted := tc.info.AllowedCredentialRoutingIDs()
			require.Equal(t, tc.restricted, restricted)
			require.Equal(t, tc.wantIDs, ids)
		})
	}
}
