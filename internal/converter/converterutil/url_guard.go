package converterutil

import (
	"context"
	"net"
	"net/url"
	"strings"
)

// blockedMetadataHosts are well-known cloud metadata endpoints. Fetching one
// would hand the caller the node's own credentials, so they are rejected by
// name in addition to the private-range check below.
var blockedMetadataHosts = []string{
	"metadata.google.internal",
	"metadata.aws.internal",
	"metadata.goog",
	"metadata",
}

// privateIPRanges are the address ranges that must never be reachable
// through a client-supplied URL.
var privateIPRanges = []string{
	"127.0.0.0/8",
	"10.0.0.0/8",
	"172.16.0.0/12",
	"192.168.0.0/16",
	"169.254.0.0/16",
	"::1/128",
	"fc00::/7",
	"fe80::/10",
}

var privateNetworks = func() []*net.IPNet {
	nets := make([]*net.IPNet, 0, len(privateIPRanges))
	for _, cidr := range privateIPRanges {
		_, network, err := net.ParseCIDR(cidr)
		if err != nil {
			continue
		}
		nets = append(nets, network)
	}
	return nets
}()

// IsPrivateURL reports whether a URL points at private or link-local address
// space, a loopback address, or a cloud metadata endpoint.
//
// This is the SSRF guard for any URL the router fetches on the client's
// behalf. It resolves the hostname once and checks every returned address,
// so a hostname with several A records cannot smuggle a private one through.
//
// DNS rebinding remains possible in principle: an attacker-controlled
// resolver can answer with a public IP here and a private one at connect
// time. Callers that fetch the URL themselves should therefore also pin the
// address they connect to — see GuardedDialContext, which re-checks the
// resolved address at dial time and closes that gap.
func IsPrivateURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		// An unparseable URL is treated as private: fail closed.
		return true
	}

	hostname := parsed.Hostname()
	if hostname == "" {
		return true
	}

	for _, blocked := range blockedMetadataHosts {
		if strings.EqualFold(hostname, blocked) {
			return true
		}
	}
	if strings.EqualFold(hostname, "localhost") {
		return true
	}

	if ip := net.ParseIP(hostname); ip != nil {
		return isPrivateIP(ip)
	}

	ips, err := net.DefaultResolver.LookupIPAddr(context.Background(), hostname)
	if err != nil || len(ips) == 0 {
		// A hostname that does not resolve cannot reach a private address
		// either; the fetch itself will fail and be reported as such.
		return false
	}
	for _, resolved := range ips {
		if isPrivateIP(resolved.IP) {
			return true
		}
	}
	return false
}

// isPrivateIP reports whether an IP address belongs to a blocked range.
func isPrivateIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() ||
		ip.IsMulticast() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return true
	}
	for _, network := range privateNetworks {
		if network.Contains(ip) {
			return true
		}
	}
	return false
}

// GuardedDialContext returns a DialContext function that refuses to connect
// to a private or link-local address, checking the address again at connect
// time rather than trusting a separate earlier lookup.
//
// Pass it to an http.Transport's DialContext to close the DNS-rebinding gap
// that IsPrivateURL alone leaves open: the hostname is resolved by the
// dialer, and the address actually being connected to is verified before
// the connection is established.
func GuardedDialContext(dialer *net.Dialer) func(ctx context.Context, network, addr string) (net.Conn, error) {
	if dialer == nil {
		dialer = &net.Dialer{}
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, _, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, err
		}
		if ip := net.ParseIP(host); ip != nil {
			if isPrivateIP(ip) {
				return nil, &BlockedAddressError{Host: host}
			}
			return dialer.DialContext(ctx, network, addr)
		}
		// Resolve ourselves and dial the vetted address directly, so the
		// address we checked is the address we connect to.
		ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}
		var lastErr error
		for _, ip := range ips {
			if isPrivateIP(ip.IP) {
				return nil, &BlockedAddressError{Host: host}
			}
			conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(ip.IP.String(), portOf(addr)))
			if err == nil {
				return conn, nil
			}
			lastErr = err
		}
		if lastErr == nil {
			lastErr = &net.DNSError{Err: "no addresses", Name: host}
		}
		return nil, lastErr
	}
}

// portOf returns the port part of a host:port pair.
func portOf(addr string) string {
	_, port, err := net.SplitHostPort(addr)
	if err != nil {
		return ""
	}
	return port
}

// BlockedAddressError is returned when a dial is refused because the target
// address is not allowed. It carries no details beyond the host so it can be
// logged safely.
type BlockedAddressError struct {
	Host string
}

func (e *BlockedAddressError) Error() string {
	return "connection to a non-public address is not allowed"
}
