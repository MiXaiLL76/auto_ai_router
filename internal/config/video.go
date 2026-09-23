package config

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	VideoModelGen45     = "runway/gen4.5"
	VideoModelGen4Turbo = "runway/gen4_turbo"

	defaultRunwayBaseURL     = "https://api.dev.runwayml.com"
	defaultRunwayAPIVersion  = "2024-11-06"
	defaultVideoPollInterval = 5 * time.Second
	defaultVideoLeaseTTL     = 5 * time.Minute
	defaultVideoUploadTTL    = 15 * time.Minute
	defaultMaxArtifactBytes  = int64(1 << 30)
)

type VideoModelConfig struct {
	Name          string `yaml:"name"`
	ProviderModel string `yaml:"provider_model"`
}

type VideoConfig struct {
	Enabled           bool               `yaml:"enabled"`
	RunwayAPIKey      string             `yaml:"runway_api_key"`
	RunwayBaseURL     string             `yaml:"runway_base_url"`
	RunwayAPIVersion  string             `yaml:"runway_api_version"`
	S3Endpoint        string             `yaml:"s3_endpoint"`
	S3Region          string             `yaml:"s3_region"`
	S3Bucket          string             `yaml:"s3_bucket"`
	S3AccessKey       string             `yaml:"s3_access_key"`
	S3SecretKey       string             `yaml:"s3_secret_key"`
	S3Prefix          string             `yaml:"s3_prefix,omitempty"`
	ArtifactProxyURL  string             `yaml:"artifact_proxy_url,omitempty"`
	UploadSigningKey  string             `yaml:"upload_signing_key"`
	PollInterval      time.Duration      `yaml:"poll_interval"`
	LeaseTTL          time.Duration      `yaml:"lease_ttl"`
	UploadTTL         time.Duration      `yaml:"upload_ttl"`
	MaxArtifactBytes  int64              `yaml:"max_artifact_bytes"`
	WorkerConcurrency int                `yaml:"worker_concurrency"`
	Models            []VideoModelConfig `yaml:"models"`
}

func (c *VideoConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawVideoConfig struct {
		Enabled           string             `yaml:"enabled"`
		RunwayAPIKey      string             `yaml:"runway_api_key"`
		RunwayBaseURL     string             `yaml:"runway_base_url"`
		RunwayAPIVersion  string             `yaml:"runway_api_version"`
		S3Endpoint        string             `yaml:"s3_endpoint"`
		S3Region          string             `yaml:"s3_region"`
		S3Bucket          string             `yaml:"s3_bucket"`
		S3AccessKey       string             `yaml:"s3_access_key"`
		S3SecretKey       string             `yaml:"s3_secret_key"`
		S3Prefix          string             `yaml:"s3_prefix,omitempty"`
		ArtifactProxyURL  string             `yaml:"artifact_proxy_url,omitempty"`
		UploadSigningKey  string             `yaml:"upload_signing_key"`
		PollInterval      string             `yaml:"poll_interval"`
		LeaseTTL          string             `yaml:"lease_ttl"`
		UploadTTL         string             `yaml:"upload_ttl"`
		MaxArtifactBytes  string             `yaml:"max_artifact_bytes"`
		WorkerConcurrency string             `yaml:"worker_concurrency"`
		Models            []VideoModelConfig `yaml:"models"`
	}

	var raw rawVideoConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}

	var err error
	if c.Enabled, err = parseField(raw.Enabled, false, strconv.ParseBool, "video.enabled"); err != nil {
		return err
	}
	if c.PollInterval, err = parseField(raw.PollInterval, defaultVideoPollInterval, time.ParseDuration, "video.poll_interval"); err != nil {
		return err
	}
	if c.LeaseTTL, err = parseField(raw.LeaseTTL, defaultVideoLeaseTTL, time.ParseDuration, "video.lease_ttl"); err != nil {
		return err
	}
	if c.UploadTTL, err = parseField(raw.UploadTTL, defaultVideoUploadTTL, time.ParseDuration, "video.upload_ttl"); err != nil {
		return err
	}
	if c.MaxArtifactBytes, err = parseField(raw.MaxArtifactBytes, defaultMaxArtifactBytes, func(value string) (int64, error) {
		return strconv.ParseInt(value, 10, 64)
	}, "video.max_artifact_bytes"); err != nil {
		return err
	}
	if c.WorkerConcurrency, err = parseField(raw.WorkerConcurrency, 1, strconv.Atoi, "video.worker_concurrency"); err != nil {
		return err
	}

	c.RunwayAPIKey = resolveEnvString(raw.RunwayAPIKey)
	c.RunwayBaseURL = valueOrDefault(resolveEnvString(raw.RunwayBaseURL), defaultRunwayBaseURL)
	c.RunwayAPIVersion = valueOrDefault(resolveEnvString(raw.RunwayAPIVersion), defaultRunwayAPIVersion)
	c.S3Endpoint = resolveEnvString(raw.S3Endpoint)
	c.S3Region = resolveEnvString(raw.S3Region)
	c.S3Bucket = resolveEnvString(raw.S3Bucket)
	c.S3AccessKey = resolveEnvString(raw.S3AccessKey)
	c.S3SecretKey = resolveEnvString(raw.S3SecretKey)
	c.S3Prefix = strings.Trim(resolveEnvString(raw.S3Prefix), "/")
	c.ArtifactProxyURL = resolveEnvString(raw.ArtifactProxyURL)
	c.UploadSigningKey = resolveEnvString(raw.UploadSigningKey)
	c.Models = raw.Models
	for index := range c.Models {
		c.Models[index].Name = resolveEnvString(c.Models[index].Name)
		c.Models[index].ProviderModel = resolveEnvString(c.Models[index].ProviderModel)
	}
	return nil
}

func (c VideoConfig) Validate() error {
	if !c.Enabled {
		return nil
	}
	required := map[string]string{
		"runway_api_key":     c.RunwayAPIKey,
		"runway_base_url":    c.RunwayBaseURL,
		"runway_api_version": c.RunwayAPIVersion,
		"s3_endpoint":        c.S3Endpoint,
		"s3_region":          c.S3Region,
		"s3_bucket":          c.S3Bucket,
		"s3_access_key":      c.S3AccessKey,
		"s3_secret_key":      c.S3SecretKey,
		"upload_signing_key": c.UploadSigningKey,
	}
	for field, value := range required {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("video.%s is required when video is enabled", field)
		}
	}
	if len(c.UploadSigningKey) < 32 {
		return fmt.Errorf("video.upload_signing_key must contain at least 32 bytes")
	}
	if err := validateBaseURL("video.runway", c.RunwayBaseURL); err != nil {
		return err
	}
	if err := validateBaseURL("video.s3", c.S3Endpoint); err != nil {
		return err
	}
	for field, rawURL := range map[string]string{
		"video.runway_base_url": c.RunwayBaseURL,
		"video.s3_endpoint":     c.S3Endpoint,
	} {
		parsed, err := url.Parse(rawURL)
		if err != nil || parsed.Scheme != "https" {
			return fmt.Errorf("%s must use https", field)
		}
	}
	if c.ArtifactProxyURL != "" {
		parsed, err := url.Parse(c.ArtifactProxyURL)
		if err != nil || parsed.Host == "" || parsed.Scheme != "http" && parsed.Scheme != "https" {
			return fmt.Errorf("video.artifact_proxy_url must use http or https")
		}
	}
	if c.PollInterval <= 0 {
		return fmt.Errorf("video.poll_interval must be positive")
	}
	if c.LeaseTTL <= 0 {
		return fmt.Errorf("video.lease_ttl must be positive")
	}
	if c.UploadTTL <= 0 {
		return fmt.Errorf("video.upload_ttl must be positive")
	}
	if c.MaxArtifactBytes <= 0 {
		return fmt.Errorf("video.max_artifact_bytes must be positive")
	}
	if c.WorkerConcurrency <= 0 {
		return fmt.Errorf("video.worker_concurrency must be positive")
	}
	want := map[string]string{
		VideoModelGen45:     "gen4.5",
		VideoModelGen4Turbo: "gen4_turbo",
	}
	if len(c.Models) != len(want) {
		return fmt.Errorf("video.models must contain exactly %s and %s", VideoModelGen45, VideoModelGen4Turbo)
	}
	for _, model := range c.Models {
		providerModel, ok := want[model.Name]
		if !ok || model.ProviderModel != providerModel {
			return fmt.Errorf("invalid video model mapping %q to %q", model.Name, model.ProviderModel)
		}
		delete(want, model.Name)
	}
	if len(want) != 0 {
		return fmt.Errorf("video.models must contain both supported models")
	}
	return nil
}

func (c VideoConfig) ModelIDs() []string {
	if !c.Enabled {
		return nil
	}
	modelIDs := make([]string, 0, len(c.Models))
	for _, model := range c.Models {
		modelIDs = append(modelIDs, model.Name)
	}
	return modelIDs
}

func valueOrDefault(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
