package config

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Defaults for the image_download section.
//
// The per-image limit is the providers' own: Anthropic accepts 10 MB per
// image (5 MB on Bedrock) and rejects anything larger, so downloading more
// than that only wastes the router's time and memory before the provider
// refuses the request. Gemini's limit is on the whole request (20 MB), and
// OpenAI's on the payload (512 MB), both above this.
//
// The total limit bounds what one request may pull in: the router fetches
// the images itself, base64-encodes them (a third more bytes) and sends
// them upstream, so a request with many large images would exceed the
// router's own body limit (server.max_body_size_mb) before the provider
// ever saw it.
//
// The count limit is the provider's floor rather than its ceiling: Anthropic
// takes 100 images per request on a 200k-context model and 600 otherwise,
// Gemini 3600, OpenAI 1500. This is a backstop against a request that
// carries hundreds of images, not a limit anyone should hit; production
// traffic peaks at 11 images per request.
//
// The timeout bounds how long the images of one request may take together:
// each fetch has its own, shorter timeout, so without a shared deadline a
// message with N images could keep the router busy for N times that.
const (
	// DefaultImageMaxBytes is the largest single image the router fetches,
	// in bytes (10 MB, Anthropic's per-image limit).
	DefaultImageMaxBytes int64 = 10 * 1024 * 1024
	// DefaultImageMaxTotalBytes is the largest total the images of one
	// request may occupy, in bytes (100 MB, the router's default
	// server.max_body_size_mb).
	DefaultImageMaxTotalBytes int64 = 100 * 1024 * 1024
	// DefaultImageMaxCount is the most images one request may carry.
	DefaultImageMaxCount = 100
	// DefaultImageFetchTimeout is the total time the images of one request
	// may take to download.
	DefaultImageFetchTimeout = 15 * time.Second
	// DefaultImageFetchTimeoutPerImage is the timeout of a single image
	// fetch. The shorter of it and the request's shared deadline wins.
	DefaultImageFetchTimeoutPerImage = 8 * time.Second
	// DefaultImageFetchConnectTimeout is the timeout of establishing the
	// connection to the image host.
	DefaultImageFetchConnectTimeout = 3 * time.Second
)

// ImageDownloadConfig is the top-level image_download section: the limits
// on the images the router fetches on the client's behalf.
type ImageDownloadConfig struct {
	// MaxBytes is the largest single image the router fetches. Above it the
	// request is answered 413 (image_too_large) instead of downloading
	// something the provider would refuse anyway.
	MaxBytes int64 `yaml:"max_bytes"`
	// MaxTotalBytes is the largest total the images of one request may
	// occupy. Above it the request is answered 413.
	MaxTotalBytes int64 `yaml:"max_total_bytes"`
	// MaxCount is the most images one request may carry. Above it the
	// request is answered 400 (too_many_images). 0 = no limit.
	MaxCount int `yaml:"max_count"`
	// Timeout is the total time the images of one request may take to
	// download, shared by all of them.
	Timeout time.Duration `yaml:"timeout"`
	// TimeoutPerImage is the timeout of a single image fetch.
	TimeoutPerImage time.Duration `yaml:"timeout_per_image"`
	// ConnectTimeout is the timeout of establishing the connection to an
	// image host.
	ConnectTimeout time.Duration `yaml:"connect_timeout"`
}

// UnmarshalYAML reads the section, applying the defaults to the fields left
// out. A duration of 0 means "no limit" and a byte limit of 0 means "no
// limit" too, so an explicit 0 is different from an omitted field.
func (c *ImageDownloadConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawImageDownloadConfig struct {
		MaxBytes        string `yaml:"max_bytes"`
		MaxTotalBytes   string `yaml:"max_total_bytes"`
		MaxCount        string `yaml:"max_count"`
		Timeout         string `yaml:"timeout"`
		TimeoutPerImage string `yaml:"timeout_per_image"`
		ConnectTimeout  string `yaml:"connect_timeout"`
	}
	var raw rawImageDownloadConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}

	var err error
	if c.MaxBytes, err = parseField(raw.MaxBytes, DefaultImageMaxBytes, parseByteSize, "image_download.max_bytes"); err != nil {
		return err
	}
	if c.MaxTotalBytes, err = parseField(raw.MaxTotalBytes, DefaultImageMaxTotalBytes, parseByteSize, "image_download.max_total_bytes"); err != nil {
		return err
	}
	if c.MaxCount, err = parseField(raw.MaxCount, DefaultImageMaxCount, strconv.Atoi, "image_download.max_count"); err != nil {
		return err
	}
	if c.Timeout, err = parseField(raw.Timeout, DefaultImageFetchTimeout, time.ParseDuration, "image_download.timeout"); err != nil {
		return err
	}
	if c.TimeoutPerImage, err = parseField(raw.TimeoutPerImage, DefaultImageFetchTimeoutPerImage, time.ParseDuration, "image_download.timeout_per_image"); err != nil {
		return err
	}
	if c.ConnectTimeout, err = parseField(raw.ConnectTimeout, DefaultImageFetchConnectTimeout, time.ParseDuration, "image_download.connect_timeout"); err != nil {
		return err
	}
	return nil
}

// parseByteSize reads a byte count: a plain number of bytes, or a size with
// a unit suffix ("10MB", "512kb", "1.5gb"). A bare number is bytes.
func parseByteSize(raw string) (int64, error) {
	s := strings.ToLower(strings.TrimSpace(raw))
	if s == "" {
		return 0, nil
	}
	multiplier := int64(1)
	for suffix, mult := range map[string]int64{
		"kb": 1024, "mb": 1024 * 1024, "gb": 1024 * 1024 * 1024,
	} {
		if strings.HasSuffix(s, suffix) {
			s = strings.TrimSpace(strings.TrimSuffix(s, suffix))
			multiplier = mult
			break
		}
	}
	// A plain "b" suffix is bytes, already the multiplier's unit.
	s = strings.TrimSuffix(strings.TrimSpace(s), "b")
	value, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("not a byte size: %q", raw)
	}
	return int64(value * float64(multiplier)), nil
}

// defaultImageDownloadConfig is used when the image_download section is
// absent (or present but empty).
func defaultImageDownloadConfig() ImageDownloadConfig {
	return ImageDownloadConfig{
		MaxBytes:        DefaultImageMaxBytes,
		MaxTotalBytes:   DefaultImageMaxTotalBytes,
		MaxCount:        DefaultImageMaxCount,
		Timeout:         DefaultImageFetchTimeout,
		TimeoutPerImage: DefaultImageFetchTimeoutPerImage,
		ConnectTimeout:  DefaultImageFetchConnectTimeout,
	}
}

// ApplyDefaults fills the fields left unset.
//
// A zero value in a parsed config means "no limit" (the operator asked for
// it), so the defaults are NOT applied here: UnmarshalYAML already gave
// every omitted field its default, and an explicit 0 must survive. This
// method only covers a config built in code, which never went through
// UnmarshalYAML -- and it cannot tell an explicit 0 from an unset field, so
// it leaves the byte and count limits alone and fills only the durations,
// where a zero really does mean "unset".
func (c *ImageDownloadConfig) ApplyDefaults() {
	if c.Timeout == 0 {
		c.Timeout = DefaultImageFetchTimeout
	}
	if c.TimeoutPerImage == 0 {
		c.TimeoutPerImage = DefaultImageFetchTimeoutPerImage
	}
	if c.ConnectTimeout == 0 {
		c.ConnectTimeout = DefaultImageFetchConnectTimeout
	}
}

func (c *ImageDownloadConfig) Validate() error {
	if c.MaxBytes < 0 || c.MaxTotalBytes < 0 || c.MaxCount < 0 {
		return fmt.Errorf("image_download.max_bytes, max_total_bytes and max_count must not be negative")
	}
	if c.Timeout < 0 || c.TimeoutPerImage < 0 || c.ConnectTimeout < 0 {
		return fmt.Errorf("image_download.timeout, timeout_per_image and connect_timeout must not be negative")
	}
	if c.MaxBytes > 0 && c.MaxTotalBytes > 0 && c.MaxBytes > c.MaxTotalBytes {
		return fmt.Errorf("image_download.max_bytes (%d) must not exceed image_download.max_total_bytes (%d)",
			c.MaxBytes, c.MaxTotalBytes)
	}
	return nil
}
