package sosana

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateResultURLBlocksUnsafeHosts(t *testing.T) {
	tests := []string{
		"http://main-r2.sosana.blog/image.png",
		"https://localhost/image.png",
		"https://127.0.0.1/image.png",
		"https://169.254.169.254/latest/meta-data",
		"https://100.64.0.1/image.png",
		"https://example.com/image.png",
	}

	for _, rawURL := range tests {
		t.Run(rawURL, func(t *testing.T) {
			parsed, err := parseResultURL(rawURL)
			require.NoError(t, err)
			require.Error(t, validateResultURL(context.Background(), parsed))
		})
	}
}

func TestValidateResultURLAllowsLocalOnlyWithTestHook(t *testing.T) {
	restore := SetAllowPrivateResultURLForTests(func(parsed *url.URL) bool {
		return parsed.Scheme == "http" && parsed.Hostname() == "127.0.0.1"
	})
	t.Cleanup(restore)

	parsed, err := parseResultURL("http://127.0.0.1/image.png")
	require.NoError(t, err)
	require.NoError(t, validateResultURL(context.Background(), parsed))
}

func TestDialResultAddressRejectsPrivateIP(t *testing.T) {
	conn, err := dialResultAddress(context.Background(), "tcp", net.JoinHostPort("127.0.0.1", "443"))

	require.Error(t, err)
	assert.Nil(t, conn)
	assert.Contains(t, err.Error(), "private address")
}

func TestDownloadResultImageRejectsUnsafeProductionURL(t *testing.T) {
	called := false
	imageServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer imageServer.Close()

	resultURL := imageServer.URL + "/private.png"
	image, err := DownloadResultImage(context.Background(), http.DefaultClient, BananaTaskResponse{
		Status:        StatusCompleted,
		ResultFileURL: &resultURL,
	})

	require.Error(t, err)
	assert.Empty(t, image.Bytes)
	assert.False(t, called)
	var imageErr *ResultImageError
	require.ErrorAs(t, err, &imageErr)
	assert.Equal(t, http.StatusBadGateway, imageErr.StatusCode)
	assert.Contains(t, err.Error(), "host is not allowed")
}

func AllowPrivateResultURLsForTest(t *testing.T) {
	t.Helper()

	restore := SetAllowPrivateResultURLForTests(func(parsed *url.URL) bool {
		if parsed.Scheme != "http" {
			return false
		}
		host := parsed.Hostname()
		if strings.EqualFold(host, "localhost") {
			return true
		}
		ip := net.ParseIP(host)
		return ip != nil && IsUnsafeResultIP(ip)
	})
	t.Cleanup(restore)
}
