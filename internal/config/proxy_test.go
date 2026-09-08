package config

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestCredentialProxyURL(t *testing.T) {
	t.Setenv("CREDENTIAL_PROXY", "http://user:secret@localhost:3128")
	var cred CredentialConfig
	require.NoError(t, yaml.Unmarshal([]byte("name: test\ntype: openai\nproxy_url: os.environ/CREDENTIAL_PROXY"), &cred))
	require.Equal(t, "http://user:secret@localhost:3128", cred.ProxyURL)
	data, err := yaml.Marshal(cred)
	require.NoError(t, err)
	var restored CredentialConfig
	require.NoError(t, yaml.Unmarshal(data, &restored))
	require.Equal(t, cred.ProxyURL, restored.ProxyURL)
	changed := cred
	changed.ProxyURL = "http://another:3128"
	require.False(t, cred.SameProviderIdentity(changed))
}

func TestParseProxyURL(t *testing.T) {
	for _, raw := range []string{"", "http://proxy:3128", "https://user:p%40ss@proxy", "socks5://proxy:1080", "socks5h://[::1]:1080", "http://proxy/"} {
		t.Run(raw, func(t *testing.T) {
			_, err := ParseProxyURL(raw)
			require.NoError(t, err)
		})
	}
	for _, raw := range []string{"proxy:3128", "ftp://proxy", "http:///", "http://proxy:0", "http://proxy:65536", "http://proxy:nope", "http://proxy/path", "http://proxy?", "http://proxy?q=secret", "http://proxy#secret", "http://user:secret%@proxy"} {
		t.Run(raw, func(t *testing.T) {
			_, err := ParseProxyURL(raw)
			require.Error(t, err)
			require.NotContains(t, err.Error(), "secret")
			var cred CredentialConfig
			err = yaml.Unmarshal([]byte("name: test\nproxy_url: "+raw), &cred)
			require.Error(t, err)
			require.Contains(t, err.Error(), "credential test:")
			require.NotContains(t, err.Error(), "secret")
		})
	}
}
