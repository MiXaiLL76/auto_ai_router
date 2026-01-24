package config

import (
	"fmt"
	"net/url"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server      ServerConfig      `yaml:"server"`
	Fail2Ban    Fail2BanConfig    `yaml:"fail2ban"`
	Credentials []CredentialConfig `yaml:"credentials"`
	Monitoring  MonitoringConfig  `yaml:"monitoring"`
}

type ServerConfig struct {
	Port           int           `yaml:"port"`
	MaxBodySizeMB  int           `yaml:"max_body_size_mb"`
	RequestTimeout time.Duration `yaml:"request_timeout"`
	Debug          bool          `yaml:"debug"`
}

type Fail2BanConfig struct {
	MaxAttempts int           `yaml:"max_attempts"`
	BanDuration time.Duration `yaml:"ban_duration"`
	ErrorCodes  []int         `yaml:"error_codes"`
}

type CredentialConfig struct {
	Name    string `yaml:"name"`
	Type    string `yaml:"type"`
	APIKey  string `yaml:"api_key"`
	BaseURL string `yaml:"base_url"`
	RPM     int    `yaml:"rpm"`
}

type MonitoringConfig struct {
	PrometheusEnabled bool   `yaml:"prometheus_enabled"`
	HealthCheckPath   string `yaml:"health_check_path"`
}

// UnmarshalYAML implements custom unmarshaling for Fail2BanConfig
func (f *Fail2BanConfig) UnmarshalYAML(value *yaml.Node) error {
	// Create a temporary struct with string ban_duration
	type tempConfig struct {
		MaxAttempts int    `yaml:"max_attempts"`
		BanDuration string `yaml:"ban_duration"`
		ErrorCodes  []int  `yaml:"error_codes"`
	}

	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	f.MaxAttempts = temp.MaxAttempts
	f.ErrorCodes = temp.ErrorCodes

	// Parse ban_duration
	if temp.BanDuration == "permanent" || temp.BanDuration == "" {
		f.BanDuration = 0 // 0 means permanent ban
	} else {
		duration, err := time.ParseDuration(temp.BanDuration)
		if err != nil {
			return fmt.Errorf("invalid ban_duration: %w", err)
		}
		f.BanDuration = duration
	}

	return nil
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &cfg, nil
}

func (c *Config) Validate() error {
	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Server.Port)
	}

	if c.Server.MaxBodySizeMB <= 0 {
		return fmt.Errorf("invalid max_body_size_mb: %d", c.Server.MaxBodySizeMB)
	}

	if c.Server.RequestTimeout <= 0 {
		return fmt.Errorf("invalid request_timeout: %v", c.Server.RequestTimeout)
	}

	if c.Fail2Ban.MaxAttempts <= 0 {
		return fmt.Errorf("invalid max_attempts: %d", c.Fail2Ban.MaxAttempts)
	}

	if len(c.Credentials) == 0 {
		return fmt.Errorf("no credentials configured")
	}

	for i, cred := range c.Credentials {
		if cred.Name == "" {
			return fmt.Errorf("credential %d: name is required", i)
		}
		if cred.APIKey == "" {
			return fmt.Errorf("credential %s: api_key is required", cred.Name)
		}
		if cred.BaseURL == "" {
			return fmt.Errorf("credential %s: base_url is required", cred.Name)
		}
		// Validate base_url is a valid URL
		parsedURL, err := url.Parse(cred.BaseURL)
		if err != nil {
			return fmt.Errorf("credential %s: invalid base_url: %w", cred.Name, err)
		}
		if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
			return fmt.Errorf("credential %s: base_url must use http or https scheme, got: %s", cred.Name, parsedURL.Scheme)
		}
		if parsedURL.Host == "" {
			return fmt.Errorf("credential %s: base_url must have a host", cred.Name)
		}
		if cred.RPM <= 0 {
			return fmt.Errorf("credential %s: invalid rpm: %d", cred.Name, cred.RPM)
		}
	}

	return nil
}
