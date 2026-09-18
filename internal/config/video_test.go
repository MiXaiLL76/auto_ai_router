package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestVideoConfigDefaultsAndEnvironment(t *testing.T) {
	t.Setenv("VIDEO_RUNWAY_KEY", "runway-secret")
	t.Setenv("VIDEO_S3_SECRET", "s3-secret")

	var cfg VideoConfig
	err := yaml.Unmarshal([]byte(`
enabled: true
runway_api_key: os.environ/VIDEO_RUNWAY_KEY
s3_endpoint: https://s3.example.com
s3_region: ru-1
s3_bucket: video
s3_access_key: access
s3_secret_key: os.environ/VIDEO_S3_SECRET
s3_prefix: /air/video/
upload_signing_key: upload-signing-key-with-32-bytes-minimum
models:
  - name: runway/gen4.5
    provider_model: gen4.5
  - name: runway/gen4_turbo
    provider_model: gen4_turbo
`), &cfg)

	require.NoError(t, err)
	require.NoError(t, cfg.Validate())
	assert.Equal(t, "runway-secret", cfg.RunwayAPIKey)
	assert.Equal(t, "s3-secret", cfg.S3SecretKey)
	assert.Equal(t, defaultRunwayBaseURL, cfg.RunwayBaseURL)
	assert.Equal(t, defaultRunwayAPIVersion, cfg.RunwayAPIVersion)
	assert.Equal(t, 5*time.Second, cfg.PollInterval)
	assert.Equal(t, 5*time.Minute, cfg.LeaseTTL)
	assert.Equal(t, 15*time.Minute, cfg.UploadTTL)
	assert.Equal(t, int64(1<<30), cfg.MaxArtifactBytes)
	assert.Equal(t, 1, cfg.WorkerConcurrency)
	assert.Equal(t, "air/video", cfg.S3Prefix)
	assert.Equal(t, []string{VideoModelGen45, VideoModelGen4Turbo}, cfg.ModelIDs())
}

func TestVideoConfigRejectsIncompleteModelSurface(t *testing.T) {
	cfg := validVideoConfig()
	cfg.Models = cfg.Models[:1]
	require.ErrorContains(t, cfg.Validate(), "must contain exactly")

	cfg = validVideoConfig()
	cfg.Models[1].ProviderModel = "gen4.5"
	require.ErrorContains(t, cfg.Validate(), "invalid video model mapping")
}

func TestVideoConfigRejectsMissingDependencies(t *testing.T) {
	cfg := validVideoConfig()
	cfg.S3SecretKey = ""
	require.ErrorContains(t, cfg.Validate(), "video.s3_secret_key is required")

	cfg = validVideoConfig()
	cfg.PollInterval = 0
	require.ErrorContains(t, cfg.Validate(), "video.poll_interval must be positive")

	cfg = validVideoConfig()
	cfg.UploadSigningKey = "short"
	require.ErrorContains(t, cfg.Validate(), "at least 32 bytes")
}

func TestDisabledVideoConfigNeedsNoDependencies(t *testing.T) {
	require.NoError(t, (VideoConfig{}).Validate())
	assert.Nil(t, (VideoConfig{}).ModelIDs())
}

func validVideoConfig() VideoConfig {
	return VideoConfig{
		Enabled:           true,
		RunwayAPIKey:      "runway-secret",
		RunwayBaseURL:     defaultRunwayBaseURL,
		RunwayAPIVersion:  defaultRunwayAPIVersion,
		S3Endpoint:        "https://s3.example.com",
		S3Region:          "ru-1",
		S3Bucket:          "video",
		S3AccessKey:       "access",
		S3SecretKey:       "secret",
		UploadSigningKey:  "upload-signing-key-with-32-bytes-minimum",
		PollInterval:      defaultVideoPollInterval,
		LeaseTTL:          defaultVideoLeaseTTL,
		UploadTTL:         defaultVideoUploadTTL,
		MaxArtifactBytes:  defaultMaxArtifactBytes,
		WorkerConcurrency: 1,
		Models: []VideoModelConfig{
			{Name: VideoModelGen45, ProviderModel: "gen4.5"},
			{Name: VideoModelGen4Turbo, ProviderModel: "gen4_turbo"},
		},
	}
}
