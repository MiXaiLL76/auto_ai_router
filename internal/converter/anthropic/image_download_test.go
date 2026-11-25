package anthropic

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// withPermissiveImageFetch relaxes both the SSRF URL guard and the dialer
// for the duration of a test, so the downloader can reach a loopback
// httptest server. The guard itself is covered by converterutil's own tests.
func withPermissiveImageFetch(t *testing.T) {
	t.Helper()
	origGuard := imageURLGuard
	origDial := imageDialContext
	imageURLGuard = func(string) bool { return false }
	imageDialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, network, addr)
	}
	t.Cleanup(func() {
		imageURLGuard = origGuard
		imageDialContext = origDial
	})
}

// TestDownloadAndEncodeImageSucceeds covers the happy path: a real image
// behind a URL is fetched and handed over as base64.
func TestDownloadAndEncodeImageSucceeds(t *testing.T) {
	withPermissiveImageFetch(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write([]byte{0xff, 0xd8, 0xff, 0xd9})
	}))
	defer server.Close()

	block, err := DownloadAndEncodeImage(server.URL+"/image.jpg", imageDownloadParam)
	require.NoError(t, err)
	require.NotNil(t, block)
	assert.Equal(t, "image", block.Type)
	require.NotNil(t, block.Source)
	assert.Equal(t, "base64", block.Source.Type)
	assert.Equal(t, "image/jpeg", block.Source.MediaType)
	assert.NotEmpty(t, block.Source.Data)
}

// TestDownloadAndEncodeImageSniffsFormat covers hosts that serve a valid
// image with a useless content type (or none at all) — the bytes are
// authoritative, not the header.
func TestDownloadAndEncodeImageSniffsFormat(t *testing.T) {
	withPermissiveImageFetch(t)

	pngBytes := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 0}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(pngBytes)
	}))
	defer server.Close()

	block, err := DownloadAndEncodeImage(server.URL+"/image.png", imageDownloadParam)
	require.NoError(t, err)
	require.NotNil(t, block)
	assert.Equal(t, "image/png", block.Source.MediaType)
}

// TestDownloadAndEncodeImageFailures walks every failure kind the downloader
// can hit and asserts two things each time: the error is an ImageError (so
// the proxy answers 4xx, not a generic 500), and the error text names no
// URL, host or upstream status.
func TestDownloadAndEncodeImageFailures(t *testing.T) {
	withPermissiveImageFetch(t)

	notFound := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("not found"))
	}))
	defer notFound.Close()

	serverError := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer serverError.Close()

	notAnImage := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html><body>login required</body></html>"))
	}))
	defer notAnImage.Close()

	emptyBody := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
	}))
	defer emptyBody.Close()

	for _, tt := range []struct {
		name   string
		url    string
		reason converterutil.ImageErrorReason
	}{
		{"not_found", notFound.URL + "/missing.png", converterutil.ImageErrorFetchFailed},
		{"server_error", serverError.URL + "/boom.png", converterutil.ImageErrorFetchFailed},
		{"not_an_image", notAnImage.URL + "/page.html", converterutil.ImageErrorInvalidData},
		{"empty_body", emptyBody.URL + "/empty.png", converterutil.ImageErrorInvalidData},
	} {
		t.Run(tt.name, func(t *testing.T) {
			block, err := DownloadAndEncodeImage(tt.url, imageDownloadParam)
			require.Error(t, err)
			assert.Nil(t, block)

			var imageErr *converterutil.ImageError
			require.True(t, errors.As(err, &imageErr), "must be an *ImageError")
			assert.Equal(t, tt.reason, imageErr.Reason)

			// The client must not learn anything about the target.
			assert.NotContains(t, err.Error(), tt.url)
			assert.NotContains(t, err.Error(), "127.0.0.1")
			assert.NotContains(t, err.Error(), "http://")
			assert.NotContains(t, err.Error(), "https://")
		})
	}
}

// TestDownloadAndEncodeImageRejectsPrivateURL is the SSRF guard at the
// downloader level: a client-supplied URL pointing into the cluster network
// is refused before any request is made.
func TestDownloadAndEncodeImageRejectsPrivateURL(t *testing.T) {
	for _, url := range []string{
		"http://169.254.169.254/latest/meta-data/",
		"http://10.0.0.1/image.png",
		"http://127.0.0.1/image.png",
		"http://192.168.1.10/image.png",
	} {
		t.Run(url, func(t *testing.T) {
			block, err := DownloadAndEncodeImage(url, imageDownloadParam)
			require.Error(t, err)
			assert.Nil(t, block)

			var imageErr *converterutil.ImageError
			require.True(t, errors.As(err, &imageErr))
			assert.Equal(t, converterutil.ImageErrorURLNotAllowed, imageErr.Reason)
			assert.NotContains(t, err.Error(), url)
		})
	}
}

// TestDownloadAndEncodeImageRejectsOversized checks the download size cap:
// a client must not be able to make the router pull an unbounded payload
// into memory and then into the base64 request body.
func TestDownloadAndEncodeImageRejectsOversized(t *testing.T) {
	withPermissiveImageFetch(t)

	origLimit := maxImageDownloadBytes
	maxImageDownloadBytes = 1024
	t.Cleanup(func() { maxImageDownloadBytes = origLimit })

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(make([]byte, 4096))
	}))
	defer server.Close()

	block, err := DownloadAndEncodeImage(server.URL+"/big.png", imageDownloadParam)
	require.Error(t, err)
	assert.Nil(t, block)

	var imageErr *converterutil.ImageError
	require.True(t, errors.As(err, &imageErr))
	assert.Equal(t, converterutil.ImageErrorTooLarge, imageErr.Reason)
	assert.Equal(t, 413, imageErr.StatusCode)
}

// TestConvertImageURLToAnthropicReportsClientError is the end-to-end check
// of the original bug: an unreachable image URL used to produce a plain
// error, which the proxy turned into a generic 500. It must now be a
// classified client error.
func TestConvertImageURLToAnthropicReportsClientError(t *testing.T) {
	withPermissiveImageFetch(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	result, err := convertImageURLToAnthropic(server.URL + "/missing.png")
	require.Error(t, err)
	assert.Nil(t, result)

	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr),
		"the proxy classifies errors via *RequestValidationError")
	assert.Equal(t, 400, validationErr.StatusCode)
	assert.Equal(t, "image_fetch_failed", validationErr.Code)
	assert.Equal(t, "messages.content.image_url.url", validationErr.Param)
	assert.NotContains(t, validationErr.Message, "404")
	assert.NotContains(t, validationErr.Message, server.URL)
}

// TestDownloadAndEncodeImageSendsBrowserLikeUserAgent keeps the anti-bot
// header in place: some CDNs reject unknown clients outright.
func TestDownloadAndEncodeImageSendsBrowserLikeUserAgent(t *testing.T) {
	withPermissiveImageFetch(t)

	var gotUA, gotAccept string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUA = r.Header.Get("User-Agent")
		gotAccept = r.Header.Get("Accept")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer server.Close()

	_, err := DownloadAndEncodeImage(server.URL+"/image.png", imageDownloadParam)
	require.NoError(t, err)
	assert.Contains(t, gotUA, "Mozilla/5.0")
	assert.True(t, strings.Contains(gotAccept, "image/*"), "Accept header should allow images, got %q", gotAccept)
}
