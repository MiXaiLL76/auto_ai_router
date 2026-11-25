package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestImageDownloadConfig_UnmarshalYAML(t *testing.T) {
	t.Setenv("TEST_IMAGE_MAX", "20MB")
	var cfg ImageDownloadConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
max_bytes: os.environ/TEST_IMAGE_MAX
max_total_bytes: 50MB
max_count: "7"
timeout: 30s
timeout_per_image: 4s
connect_timeout: 2s
`), &cfg))
	assert.Equal(t, ImageDownloadConfig{
		MaxBytes:        20 * 1024 * 1024,
		MaxTotalBytes:   50 * 1024 * 1024,
		MaxCount:        7,
		Timeout:         30 * time.Second,
		TimeoutPerImage: 4 * time.Second,
		ConnectTimeout:  2 * time.Second,
	}, cfg)
}

// TestImageDownloadConfig_Defaults checks that an omitted field gets the
// default and that an explicit 0 keeps meaning "no limit" (the operator
// asked for it, so ApplyDefaults must not fill it in).
func TestImageDownloadConfig_Defaults(t *testing.T) {
	var cfg ImageDownloadConfig
	require.NoError(t, yaml.Unmarshal([]byte("max_count: 5\n"), &cfg))
	assert.Equal(t, DefaultImageMaxBytes, cfg.MaxBytes)
	assert.Equal(t, DefaultImageMaxTotalBytes, cfg.MaxTotalBytes)
	assert.Equal(t, 5, cfg.MaxCount)
	assert.Equal(t, DefaultImageFetchTimeout, cfg.Timeout)

	cfg = ImageDownloadConfig{}
	require.NoError(t, yaml.Unmarshal([]byte("max_bytes: 0\nmax_count: 0\n"), &cfg))
	cfg.ApplyDefaults()
	assert.Equal(t, int64(0), cfg.MaxBytes, "explicit 0 means no limit and survives ApplyDefaults")
	assert.Equal(t, 0, cfg.MaxCount, "explicit 0 means no limit and survives ApplyDefaults")
	assert.Equal(t, DefaultImageFetchTimeout, cfg.Timeout,
		"a duration of 0 means unset, so the default applies")
}

// TestImageDownloadConfig_ByteSizes covers the size formats an operator may
// write: a plain byte count, and a unit suffix.
func TestImageDownloadConfig_ByteSizes(t *testing.T) {
	for _, tt := range []struct {
		value string
		want  int64
	}{
		{"1024", 1024},
		{"10MB", 10 * 1024 * 1024},
		{"10mb", 10 * 1024 * 1024},
		{"512KB", 512 * 1024},
		{"1gb", 1024 * 1024 * 1024},
		{"1.5mb", 1024 * 1024 * 3 / 2},
		{"2048b", 2048},
	} {
		t.Run(tt.value, func(t *testing.T) {
			var cfg ImageDownloadConfig
			require.NoError(t, yaml.Unmarshal([]byte("max_bytes: \""+tt.value+"\"\n"), &cfg))
			assert.Equal(t, tt.want, cfg.MaxBytes)
		})
	}
}

func TestImageDownloadConfig_Invalid(t *testing.T) {
	var cfg ImageDownloadConfig
	assert.ErrorContains(t, yaml.Unmarshal([]byte("max_bytes: \"not a size\"\n"), &cfg), "image_download.max_bytes")
	assert.ErrorContains(t, yaml.Unmarshal([]byte("timeout: \"not a duration\"\n"), &cfg), "image_download.timeout")
	assert.ErrorContains(t, yaml.Unmarshal([]byte("max_count: \"many\"\n"), &cfg), "image_download.max_count")
}

func TestImageDownloadConfig_Validate(t *testing.T) {
	cfg := ImageDownloadConfig{
		MaxBytes: 10, MaxTotalBytes: 100, MaxCount: 5,
		Timeout: time.Second, TimeoutPerImage: time.Second, ConnectTimeout: time.Second,
	}
	assert.NoError(t, cfg.Validate())

	cfg.MaxBytes = -1
	assert.ErrorContains(t, cfg.Validate(), "must not be negative")

	cfg = ImageDownloadConfig{MaxBytes: 10, MaxTotalBytes: 100, Timeout: -time.Second}
	assert.ErrorContains(t, cfg.Validate(), "must not be negative")

	// A single image larger than the whole request's allowance can never fit.
	cfg = ImageDownloadConfig{MaxBytes: 200, MaxTotalBytes: 100}
	assert.ErrorContains(t, cfg.Validate(), "must not exceed")
}

// TestImageDownloadConfig_InRootConfig checks that the section is read as
// part of the router's configuration, with its defaults applied when it is
// absent.
func TestImageDownloadConfig_InRootConfig(t *testing.T) {
	path := writeTempConfig(t, "server:\n  port: 8080\n  master_key: sk-test\ncredentials:\n  - name: c1\n    type: openai\n    base_url: https://example.invalid\n    api_key: sk-test\n")
	cfg, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, int64(DefaultImageMaxBytes), cfg.ImageDownload.MaxBytes)
	assert.Equal(t, int64(DefaultImageMaxTotalBytes), cfg.ImageDownload.MaxTotalBytes)
	assert.Equal(t, DefaultImageMaxCount, cfg.ImageDownload.MaxCount)
	assert.Equal(t, DefaultImageFetchTimeout, cfg.ImageDownload.Timeout)

	path = writeTempConfig(t, "server:\n  port: 8080\n  master_key: sk-test\nimage_download:\n  max_bytes: 1MB\n  max_count: 3\ncredentials:\n  - name: c1\n    type: openai\n    base_url: https://example.invalid\n    api_key: sk-test\n")
	cfg, err = Load(path)
	require.NoError(t, err)
	assert.Equal(t, int64(1024*1024), cfg.ImageDownload.MaxBytes)
	assert.Equal(t, 3, cfg.ImageDownload.MaxCount)
	assert.Equal(t, DefaultImageFetchTimeout, cfg.ImageDownload.Timeout,
		"the fields left out keep their defaults")
}

// TestImageDownloadConfig_NullSection checks that a section present but
// empty ("image_download:" with everything commented out) still gets the
// defaults -- it never reaches UnmarshalYAML.
func TestImageDownloadConfig_NullSection(t *testing.T) {
	path := writeTempConfig(t, "server:\n  port: 8080\n  master_key: sk-test\nimage_download:\ncredentials:\n  - name: c1\n    type: openai\n    base_url: https://example.invalid\n    api_key: sk-test\n")
	cfg, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, int64(DefaultImageMaxBytes), cfg.ImageDownload.MaxBytes)
	assert.Equal(t, DefaultImageMaxCount, cfg.ImageDownload.MaxCount)
}
