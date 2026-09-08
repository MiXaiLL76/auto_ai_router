package config

import (
	"fmt"
	"net/url"
	"strconv"
)

func ParseProxyURL(raw string) (*url.URL, error) {
	if raw == "" {
		return nil, nil
	}
	u, err := url.Parse(raw)
	if err != nil || u.Hostname() == "" || u.Opaque != "" {
		return nil, fmt.Errorf("proxy_url must be an absolute proxy URL with a host")
	}
	switch u.Scheme {
	case "http", "https", "socks5", "socks5h":
	default:
		return nil, fmt.Errorf("proxy_url scheme must be http, https, socks5, or socks5h")
	}
	if (u.Path != "" && u.Path != "/") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
		return nil, fmt.Errorf("proxy_url must not contain a path, query, or fragment")
	}
	if port := u.Port(); port != "" {
		n, err := strconv.Atoi(port)
		if err != nil || n < 1 || n > 65535 {
			return nil, fmt.Errorf("proxy_url port must be between 1 and 65535")
		}
	}
	return u, nil
}
