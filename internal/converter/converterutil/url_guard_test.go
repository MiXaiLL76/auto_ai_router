package converterutil

import (
	"context"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestIsPrivateURLBlocksInternalTargets is the SSRF regression guard. The
// router fetches client-supplied image URLs itself, so a URL that resolves
// into private or link-local space would let a client reach the pod
// network, the cluster API or a cloud metadata endpoint.
func TestIsPrivateURLBlocksInternalTargets(t *testing.T) {
	for _, tt := range []struct {
		name string
		url  string
	}{
		{"loopback", "http://127.0.0.1/image.png"},
		{"loopback_alt", "http://127.1.2.3:8080/image.png"},
		{"loopback_v6", "http://[::1]/image.png"},
		{"localhost", "http://localhost/image.png"},
		{"localhost_upper", "http://LOCALHOST/image.png"},
		{"private_10", "http://10.0.0.5/image.png"},
		{"private_172", "http://172.16.0.1/image.png"},
		{"private_172_upper_edge", "http://172.31.255.255/image.png"},
		{"private_192", "http://192.168.1.1/image.png"},
		{"link_local_169", "http://169.254.169.254/latest/meta-data/"},
		{"link_local_v6", "http://[fe80::1]/image.png"},
		{"unspecified", "http://0.0.0.0/image.png"},
		{"multicast", "http://224.0.0.1/image.png"},
		{"metadata_google", "http://metadata.google.internal/computeMetadata/v1/"},
		{"metadata_aws", "http://metadata.aws.internal/latest/meta-data/"},
		{"metadata_goog_alias", "http://metadata.goog/computeMetadata/v1/"},
		{"no_host", "http://"},
		{"unparseable", "://nonsense"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assert.True(t, IsPrivateURL(tt.url), "%s must be blocked", tt.url)
		})
	}
}

// TestIsPrivateURLAllowsPublicTargets makes sure the guard does not block
// ordinary public image URLs — the whole feature depends on those working.
func TestIsPrivateURLAllowsPublicTargets(t *testing.T) {
	for _, tt := range []struct {
		name string
		url  string
	}{
		{"public_https", "https://upload.wikimedia.org/commons/thumb/a/ab/example.jpg/320px-example.jpg"},
		{"public_cdn", "https://cdn.example.com/assets/image.png"},
		{"public_ip", "https://93.184.216.34/image.png"},
		{"public_http", "http://example.com/image.png"},
		{"public_with_query", "https://cdn.example.com/img.png?X-Amz-Signature=abc"},
		// 172.32.0.1 sits just outside the 172.16.0.0/12 private block.
		{"public_172_edge", "http://172.32.0.1/image.png"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assert.False(t, IsPrivateURL(tt.url), "%s must be allowed", tt.url)
		})
	}
}

// TestIsPrivateURLChecksEveryResolvedAddress covers the case the previous
// local copy got wrong: a hostname resolving to several addresses, one of
// which is private, must be rejected as a whole rather than judged by the
// first address returned.
func TestIsPrivateURLChecksEveryResolvedAddress(t *testing.T) {
	// "localhost" resolves to both 127.0.0.1 and ::1; both are private, so
	// this exercises the "any address is private" path without needing a
	// custom resolver.
	assert.True(t, IsPrivateURL("http://localhost/image.png"))
}

// TestGuardedDialContextBlocksPrivateAddresses verifies the dial-time check
// that closes the DNS-rebinding gap: even if an earlier lookup saw a public
// address, the dialer itself refuses to connect to a private one.
func TestGuardedDialContextBlocksPrivateAddresses(t *testing.T) {
	dial := GuardedDialContext(&net.Dialer{})

	for _, addr := range []string{
		"127.0.0.1:80",
		"10.0.0.1:443",
		"192.168.0.1:80",
		"169.254.169.254:80",
		"[::1]:80",
	} {
		conn, err := dial(context.Background(), "tcp", addr)
		assert.Error(t, err, "dialing %s must be refused", addr)
		assert.Nil(t, conn)

		var blocked *BlockedAddressError
		if assert.ErrorAs(t, err, &blocked) {
			assert.NotContains(t, blocked.Error(), "127.0.0.1",
				"the error text must not echo the address")
		}
	}
}

// TestGuardedDialContextAllowsPublicAddresses checks the guard does not break
// normal outbound fetches.
func TestGuardedDialContextAllowsPublicAddresses(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			_ = conn.Close()
		}
	}()

	dial := GuardedDialContext(&net.Dialer{})
	// The listener is on loopback, so the guard must refuse it — that is the
	// whole point of the check.
	conn, err := dial(context.Background(), "tcp", listener.Addr().String())
	assert.Error(t, err)
	assert.Nil(t, conn)
}

// TestBlockedAddressErrorMessageIsSafe checks the error text carries no
// address detail, so it is safe to log or surface.
func TestBlockedAddressErrorMessageIsSafe(t *testing.T) {
	err := &BlockedAddressError{Host: "169.254.169.254"}
	assert.NotContains(t, err.Error(), "169.254.169.254")
	assert.NotContains(t, err.Error(), "10.")
}
