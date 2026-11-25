package converterutil

import (
	"context"
	"net"
	"net/url"
	"strings"
	"sync"
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
//
// The list covers the RFC 1918 and RFC 6598 private space, the loopback and
// link-local ranges, and the documentation/b benchmarking ranges. The
// documentation ranges are here because a host resolving to one is never a
// real image host, and the benchmarking range (198.18.0.0/15) because it is
// used by transparent proxies that sit between the router and the internet.
//
// The same set is denied by internal/video/storage.go for artifact fetches;
// both lists cover the same ground so a URL cannot be fetchable through one
// path and blocked through the other.
var privateIPRanges = []string{
	"0.0.0.0/8",
	"10.0.0.0/8",
	"100.64.0.0/10", // RFC 6598 shared address space (CGNAT)
	"127.0.0.0/8",
	"169.254.0.0/16",
	"172.16.0.0/12",
	"192.0.0.0/24", // IETF protocol assignments
	"192.0.2.0/24", // TEST-NET-1
	"192.168.0.0/16",
	"198.18.0.0/15",   // benchmarking
	"198.51.100.0/24", // TEST-NET-2
	"203.0.113.0/24",  // TEST-NET-3
	"224.0.0.0/4",     // multicast
	"240.0.0.0/4",     // reserved, includes 255.255.255.255
	"::/128",
	"::1/128",
	"100::/64",      // discard-only
	"2001:db8::/32", // documentation
	"fc00::/7",      // unique local
	"fe80::/10",     // link-local
	"ff00::/8",      // multicast
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

// guardSettings holds the deployment's own additions to the guard: the
// address ranges its cluster uses that the built-in list does not know
// about, and the host names that may be fetched from even though they
// resolve to a blocked address.
var guardSettings struct {
	sync.RWMutex
	extraNetworks []*net.IPNet
	allowedHosts  map[string]struct{}
}

// SetBlockedCIDRs adds address ranges to the ones the guard refuses. A
// deployment whose cluster uses a range the built-in list does not cover
// adds it here, so a client cannot reach it through an image URL.
//
// The built-in ranges are never dropped: this only adds to them.
func SetBlockedCIDRs(cidrs []string) {
	networks := make([]*net.IPNet, 0, len(cidrs))
	for _, cidr := range cidrs {
		_, network, err := net.ParseCIDR(strings.TrimSpace(cidr))
		if err != nil {
			continue
		}
		networks = append(networks, network)
	}

	guardSettings.Lock()
	defer guardSettings.Unlock()
	guardSettings.extraNetworks = networks
}

// SetAllowedHosts names the hosts the router may fetch from even when they
// resolve to a blocked address.
//
// The exception is by host name, not by address: the same name may resolve
// to a public address from the router's side and to a private one from the
// client's, and only the name is known to both. A client whose images live
// on its own storage (a presigned URL on an internal S3, say) needs this:
// the image host is inside the network by design.
//
// The names are matched exactly, so a name has to be given as it appears in
// the URL.
func SetAllowedHosts(hosts []string) {
	allowed := make(map[string]struct{}, len(hosts))
	for _, host := range hosts {
		name := strings.ToLower(strings.TrimSpace(host))
		if name == "" {
			continue
		}
		allowed[name] = struct{}{}
	}

	guardSettings.Lock()
	defer guardSettings.Unlock()
	guardSettings.allowedHosts = allowed
}

// isAllowedHost reports whether a host name is on the deployment's
// allowlist.
func isAllowedHost(hostname string) bool {
	guardSettings.RLock()
	defer guardSettings.RUnlock()
	_, ok := guardSettings.allowedHosts[strings.ToLower(hostname)]
	return ok
}

// extraBlockedNetworks returns the deployment's additional blocked ranges.
func extraBlockedNetworks() []*net.IPNet {
	guardSettings.RLock()
	defer guardSettings.RUnlock()
	return guardSettings.extraNetworks
}

// IsBlockedHostURL reports whether a URL is refused by name or by its
// literal address, without resolving anything.
//
// It covers the cases that need no DNS: a malformed URL, an empty host, a
// known metadata host, "localhost", and a literal IP in a blocked range.
//
// A hostname is NOT resolved here: the caller that connects to the URL
// checks the resolved address at dial time (see GuardedDialContext), so
// resolving here too would mean two lookups of the same name on the
// successful path — paid on every image fetch, not only on failures.
func IsBlockedHostURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		// An unparseable URL is treated as blocked: fail closed.
		return true
	}

	hostname := parsed.Hostname()
	if hostname == "" {
		return true
	}

	// The deployment's own hosts are fetched even when they resolve to a
	// blocked address: a client whose images live on its own storage
	// (a presigned URL on an internal S3) needs that.
	if isAllowedHost(hostname) {
		return false
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
	return false
}

// IsPrivateURL reports whether a URL points at private or link-local address
// space, a loopback address, or a cloud metadata endpoint.
//
// This is the SSRF guard for a URL the router does NOT fetch itself, but
// hands to a provider to fetch (the Vertex path): there is no dial-time
// check to rely on, so the hostname has to be resolved and vetted here.
//
// A caller that fetches the URL itself should use IsBlockedHostURL for the
// cheap pre-check and let GuardedDialContext do the DNS-bound check at
// connect time — that way the name is resolved once, not twice.
//
// DNS rebinding remains possible in principle: an attacker-controlled
// resolver can answer with a public IP here and a private one at connect
// time. GuardedDialContext re-checks the address it is about to connect to,
// which closes that gap.
func IsPrivateURL(rawURL string) bool {
	if IsBlockedHostURL(rawURL) {
		return true
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return true
	}
	hostname := parsed.Hostname()
	if hostname == "" {
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
	// An IPv4 address reached through an IPv6 socket arrives as a
	// 4-in-6 form (::ffff:10.0.0.1). Unmap it first, or the range check
	// below would compare it against the IPv6 prefixes only and let it
	// through.
	if v4 := ip.To4(); v4 != nil {
		ip = v4
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
	// The deployment's own ranges: a cluster may use an address range the
	// built-in list does not know about.
	for _, network := range extraBlockedNetworks() {
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
		// The deployment's own hosts are dialed as they are: the guard on
		// the address is skipped for them, exactly as it is in the URL
		// pre-check. Both places must agree, or an allowed host would pass
		// the first check and be refused at connect time.
		if isAllowedHost(host) {
			return dialer.DialContext(ctx, network, addr)
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
