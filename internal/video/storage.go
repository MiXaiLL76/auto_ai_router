package video

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
	"time"
)

type MemoryObjectStore struct {
	mu      sync.Mutex
	objects map[string]memoryObject
}
type memoryObject struct {
	data              []byte
	contentType, etag string
}
type bytesReadCloser struct{ *bytes.Reader }

func (b bytesReadCloser) Close() error { return nil }

func NewMemoryObjectStore() *MemoryObjectStore {
	return &MemoryObjectStore{objects: map[string]memoryObject{}}
}
func (s *MemoryObjectStore) PutUpload(_ context.Context, u *Upload, ct string, r io.Reader) error {
	data, err := validatedUpload(u, ct, r)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.objects[u.ObjectKey] = memoryObject{data: data, contentType: u.MIME, etag: digest(data)}
	s.mu.Unlock()
	return nil
}
func (s *MemoryObjectStore) VerifyUpload(_ context.Context, u *Upload) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	o, ok := s.objects[u.ObjectKey]
	if !ok {
		return ErrNotReady
	}
	_, err := validatedUpload(u, u.MIME, bytes.NewReader(o.data))
	return err
}
func (s *MemoryObjectStore) OpenUpload(_ context.Context, u *Upload) (*Object, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	o, ok := s.objects[u.ObjectKey]
	if !ok {
		return nil, ErrNotReady
	}
	return &Object{Body: bytesReadCloser{bytes.NewReader(append([]byte(nil), o.data...))}, ContentType: o.contentType, Size: int64(len(o.data)), ETag: o.etag}, nil
}
func (s *MemoryObjectStore) DeleteUpload(_ context.Context, u *Upload) error {
	s.mu.Lock()
	delete(s.objects, u.ObjectKey)
	s.mu.Unlock()
	return nil
}
func (s *MemoryObjectStore) FetchResult(_ context.Context, j *Job, resultURL string) (Object, error) {
	if !strings.HasPrefix(resultURL, "data:video/") {
		return Object{}, ErrInvalid
	}
	comma := strings.IndexByte(resultURL, ',')
	if comma < 0 || !strings.Contains(resultURL[:comma], ";base64") {
		return Object{}, ErrInvalid
	}
	data, err := base64.StdEncoding.DecodeString(resultURL[comma+1:])
	if err != nil || int64(len(data)) > MaxArtifactBytes {
		return Object{}, ErrInvalid
	}
	ct := strings.TrimPrefix(strings.Split(resultURL[:comma], ";")[0], "data:")
	return s.PutResult(j, ct, data), nil
}
func (s *MemoryObjectStore) PutResult(j *Job, ct string, data []byte) Object {
	s.mu.Lock()
	defer s.mu.Unlock()
	o := memoryObject{data: append([]byte(nil), data...), contentType: ct, etag: digest(data)}
	s.objects[resultKey(j.OrganizationID, j.ID)] = o
	return Object{ContentType: ct, Size: int64(len(data)), ETag: o.etag}
}
func (s *MemoryObjectStore) OpenResult(_ context.Context, j *Job) (*Object, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	o, ok := s.objects[resultKey(j.OrganizationID, j.ID)]
	if !ok {
		return nil, ErrNotFound
	}
	return &Object{Body: bytesReadCloser{bytes.NewReader(append([]byte(nil), o.data...))}, ContentType: o.contentType, Size: int64(len(o.data)), ETag: o.etag}, nil
}

type S3Config struct {
	Endpoint, Region, Bucket, Prefix, AccessKey, SecretKey string
	ArtifactProxyURL                                       string
	HTTPClient                                             *http.Client
	MaxArtifactBytes                                       int64
	AllowPrivateArtifactHosts                              bool
}
type S3Store struct {
	cfg      S3Config
	endpoint *url.URL
	http     *http.Client
}

func NewS3Store(cfg S3Config) (*S3Store, error) {
	u, err := url.Parse(strings.TrimRight(cfg.Endpoint, "/"))
	if err != nil || !secureEndpoint(u) || cfg.Bucket == "" || cfg.AccessKey == "" || cfg.SecretKey == "" {
		return nil, ErrInvalid
	}
	if cfg.Region == "" {
		cfg.Region = "us-east-1"
	}
	if cfg.ArtifactProxyURL != "" {
		proxyURL, proxyErr := url.Parse(cfg.ArtifactProxyURL)
		if proxyErr != nil || proxyURL.Host == "" || proxyURL.Scheme != "http" && proxyURL.Scheme != "https" {
			return nil, ErrInvalid
		}
	}
	if cfg.MaxArtifactBytes <= 0 {
		cfg.MaxArtifactBytes = MaxArtifactBytes
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	} else {
		client := *cfg.HTTPClient
		if client.Timeout <= 0 || client.Timeout > 2*time.Minute {
			client.Timeout = 2 * time.Minute
		}
		cfg.HTTPClient = &client
	}
	return &S3Store{cfg: cfg, endpoint: u, http: cfg.HTTPClient}, nil
}

func (s *S3Store) PutUpload(ctx context.Context, u *Upload, ct string, r io.Reader) error {
	tmp, err := stageUpload(u, ct, r)
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer func() { _ = os.Remove(name) }()
	defer func() { _ = tmp.Close() }()
	return s.putReader(ctx, u.ObjectKey, u.MIME, tmp, u.SizeBytes)
}
func (s *S3Store) VerifyUpload(ctx context.Context, u *Upload) error {
	o, err := s.get(ctx, u.ObjectKey)
	if err != nil {
		return err
	}
	defer func() { _ = o.Body.Close() }()
	return verifyUpload(u, u.MIME, o.Body)
}
func (s *S3Store) OpenUpload(ctx context.Context, u *Upload) (*Object, error) {
	if err := s.VerifyUpload(ctx, u); err != nil {
		return nil, err
	}
	return s.get(ctx, u.ObjectKey)
}
func (s *S3Store) DeleteUpload(ctx context.Context, u *Upload) error {
	return s.delete(ctx, u.ObjectKey)
}

func (s *S3Store) FetchResult(ctx context.Context, j *Job, rawURL string) (Object, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return Object{}, ErrInvalid
	}
	if err = validateArtifactURL(ctx, u, s.cfg.AllowPrivateArtifactHosts); err != nil {
		return Object{}, err
	}
	client, err := safeArtifactClient(s.cfg.AllowPrivateArtifactHosts, s.cfg.ArtifactProxyURL)
	if err != nil {
		return Object{}, err
	}
	client.Timeout = s.http.Timeout
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) > 4 {
			return errorsNew("too many redirects")
		}
		return validateArtifactURL(req.Context(), req.URL, s.cfg.AllowPrivateArtifactHosts)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return Object{}, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return Object{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Object{}, fmt.Errorf("artifact status %d", resp.StatusCode)
	}
	if resp.ContentLength > s.cfg.MaxArtifactBytes {
		return Object{}, ErrInvalid
	}
	ct := strings.ToLower(strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0]))
	if !strings.HasPrefix(ct, "video/") {
		return Object{}, ErrInvalid
	}
	tmp, err := os.CreateTemp("", "air-video-result-*")
	if err != nil {
		return Object{}, err
	}
	name := tmp.Name()
	defer func() { _ = os.Remove(name) }()
	hasher := sha256.New()
	size, copyErr := io.Copy(io.MultiWriter(tmp, hasher), io.LimitReader(resp.Body, s.cfg.MaxArtifactBytes+1))
	if copyErr != nil {
		_ = tmp.Close()
		return Object{}, copyErr
	}
	if size > s.cfg.MaxArtifactBytes || resp.ContentLength >= 0 && resp.ContentLength != size {
		_ = tmp.Close()
		return Object{}, ErrInvalid
	}
	if _, err = tmp.Seek(0, io.SeekStart); err != nil {
		_ = tmp.Close()
		return Object{}, err
	}
	prefix := make([]byte, 12)
	read, readErr := io.ReadFull(tmp, prefix)
	if readErr != nil && readErr != io.ErrUnexpectedEOF {
		_ = tmp.Close()
		return Object{}, readErr
	}
	if !validSignature(ct, prefix[:read]) {
		_ = tmp.Close()
		return Object{}, ErrInvalid
	}
	if _, err = tmp.Seek(0, io.SeekStart); err != nil {
		_ = tmp.Close()
		return Object{}, err
	}
	if err = s.putReader(ctx, resultKey(j.OrganizationID, j.ID), ct, tmp, size); err != nil {
		_ = tmp.Close()
		return Object{}, err
	}
	_ = tmp.Close()
	return Object{ContentType: ct, Size: size, ETag: hex.EncodeToString(hasher.Sum(nil))}, nil
}
func (s *S3Store) OpenResult(ctx context.Context, j *Job) (*Object, error) {
	o, err := s.get(ctx, resultKey(j.OrganizationID, j.ID))
	if err != nil {
		return nil, err
	}
	if j.ResultContentType != "" {
		o.ContentType = j.ResultContentType
	}
	if j.ResultSize > 0 {
		o.Size = j.ResultSize
	}
	if j.ResultETag != "" {
		o.ETag = j.ResultETag
	}
	return o, nil
}

func (s *S3Store) putReader(ctx context.Context, key, ct string, body io.Reader, size int64) error {
	req, err := s.request(ctx, http.MethodPut, key, ct, body)
	if err != nil {
		return err
	}
	req.ContentLength = size
	resp, err := s.http.Do(req) //nolint:gosec // the endpoint is validated configuration, not request input
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("s3 put status %d", resp.StatusCode)
	}
	return nil
}
func (s *S3Store) get(ctx context.Context, key string) (*Object, error) {
	req, err := s.request(ctx, http.MethodGet, key, "", nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.http.Do(req) //nolint:gosec // the endpoint is validated configuration, not request input
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusNotFound {
		_ = resp.Body.Close()
		return nil, ErrNotFound
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("s3 get status %d", resp.StatusCode)
	}
	return &Object{Body: resp.Body, ContentType: resp.Header.Get("Content-Type"), Size: resp.ContentLength, ETag: strings.Trim(resp.Header.Get("ETag"), "\"")}, nil
}
func (s *S3Store) delete(ctx context.Context, key string) error {
	req, err := s.request(ctx, http.MethodDelete, key, "", nil)
	if err != nil {
		return err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNotFound && (resp.StatusCode < 200 || resp.StatusCode >= 300) {
		return fmt.Errorf("s3 delete status %d", resp.StatusCode)
	}
	return nil
}
func (s *S3Store) request(ctx context.Context, method, key, ct string, body io.Reader) (*http.Request, error) {
	u := *s.endpoint
	u.Path = path.Join(u.Path, s.cfg.Bucket, strings.Trim(s.cfg.Prefix, "/"), key)
	req, err := http.NewRequestWithContext(ctx, method, u.String(), body) //nolint:gosec // the endpoint is validated configuration
	if err != nil {
		return nil, err
	}
	if ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	s.sign(req)
	return req, nil
}
func (s *S3Store) sign(req *http.Request) {
	now := time.Now().UTC()
	date := now.Format("20060102")
	amz := now.Format("20060102T150405Z")
	payload := "UNSIGNED-PAYLOAD"
	req.Header.Set("X-Amz-Date", amz)
	req.Header.Set("X-Amz-Content-Sha256", payload)
	headers := "host;x-amz-content-sha256;x-amz-date"
	canonicalHeaders := "host:" + req.URL.Host + "\n" + "x-amz-content-sha256:" + payload + "\n" + "x-amz-date:" + amz + "\n"
	canonical := strings.Join([]string{req.Method, req.URL.EscapedPath(), req.URL.RawQuery, canonicalHeaders, headers, payload}, "\n")
	scope := date + "/" + s.cfg.Region + "/s3/aws4_request"
	toSign := "AWS4-HMAC-SHA256\n" + amz + "\n" + scope + "\n" + digest([]byte(canonical))
	kDate := mac([]byte("AWS4"+s.cfg.SecretKey), []byte(date))
	kRegion := mac(kDate, []byte(s.cfg.Region))
	kService := mac(kRegion, []byte("s3"))
	signature := hex.EncodeToString(mac(mac(kService, []byte("aws4_request")), []byte(toSign)))
	req.Header.Set("Authorization", "AWS4-HMAC-SHA256 Credential="+s.cfg.AccessKey+"/"+scope+", SignedHeaders="+headers+", Signature="+signature)
}

func validatedUpload(u *Upload, ct string, r io.Reader) ([]byte, error) {
	if u == nil || r == nil || !strings.EqualFold(strings.TrimSpace(strings.Split(ct, ";")[0]), u.MIME) {
		return nil, ErrInvalid
	}
	data, err := io.ReadAll(io.LimitReader(r, u.SizeBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != u.SizeBytes || digest(data) != strings.ToLower(u.SHA256) || !validSignature(u.MIME, data) {
		return nil, ErrInvalid
	}
	return data, nil
}

func stageUpload(u *Upload, ct string, r io.Reader) (*os.File, error) {
	if u == nil || r == nil || !strings.EqualFold(strings.TrimSpace(strings.Split(ct, ";")[0]), u.MIME) {
		return nil, ErrInvalid
	}
	tmp, err := os.CreateTemp("", "air-video-upload-*")
	if err != nil {
		return nil, err
	}
	name := tmp.Name()
	h := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(tmp, h), io.LimitReader(r, u.SizeBytes+1))
	if copyErr != nil || n != u.SizeBytes || hex.EncodeToString(h.Sum(nil)) != strings.ToLower(u.SHA256) {
		_ = tmp.Close()
		_ = os.Remove(name)
		return nil, ErrInvalid
	}
	if _, err = tmp.Seek(0, io.SeekStart); err != nil {
		_ = tmp.Close()
		_ = os.Remove(name)
		return nil, err
	}
	prefix := make([]byte, 12)
	read, _ := io.ReadFull(tmp, prefix)
	prefix = prefix[:read]
	if !validSignature(u.MIME, prefix) {
		_ = tmp.Close()
		_ = os.Remove(name)
		return nil, ErrInvalid
	}
	_, err = tmp.Seek(0, io.SeekStart)
	if err != nil {
		_ = tmp.Close()
		_ = os.Remove(name)
		return nil, err
	}
	return tmp, nil
}

func verifyUpload(u *Upload, ct string, r io.Reader) error {
	if u == nil || r == nil || !strings.EqualFold(strings.TrimSpace(strings.Split(ct, ";")[0]), u.MIME) {
		return ErrInvalid
	}
	h := sha256.New()
	prefix := &prefixWriter{limit: 12}
	n, err := io.Copy(io.MultiWriter(h, prefix), io.LimitReader(r, u.SizeBytes+1))
	if err != nil {
		return err
	}
	if n != u.SizeBytes || hex.EncodeToString(h.Sum(nil)) != strings.ToLower(u.SHA256) || !validSignature(u.MIME, prefix.data) {
		return ErrInvalid
	}
	return nil
}

type prefixWriter struct {
	data  []byte
	limit int
}

func (w *prefixWriter) Write(p []byte) (int, error) {
	n := len(p)
	remaining := w.limit - len(w.data)
	if remaining > len(p) {
		remaining = len(p)
	}
	if remaining > 0 {
		w.data = append(w.data, p[:remaining]...)
	}
	return n, nil
}
func validSignature(mime string, data []byte) bool {
	switch mime {
	case "image/png":
		return len(data) >= 8 && bytes.Equal(data[:8], []byte{137, 80, 78, 71, 13, 10, 26, 10})
	case "image/jpeg":
		return len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff
	case "image/webp":
		return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP"
	case "video/mp4", "video/quicktime":
		return len(data) >= 12 && string(data[4:8]) == "ftyp"
	case "video/webm":
		return len(data) >= 4 && bytes.Equal(data[:4], []byte{0x1a, 0x45, 0xdf, 0xa3})
	}
	return false
}
func resultKey(org, id string) string { return safePart(org) + "/videos/" + safePart(id) + "/result" }
func digest(data []byte) string       { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func mac(key, data []byte) []byte     { h := hmac.New(sha256.New, key); h.Write(data); return h.Sum(nil) }

func validateArtifactURL(ctx context.Context, u *url.URL, allowPrivate bool) error {
	if u == nil || !secureEndpoint(u) || u.User != nil {
		return ErrInvalid
	}
	if allowPrivate {
		return nil
	}
	ips, err := net.DefaultResolver.LookupIPAddr(ctx, u.Hostname())
	if err != nil || len(ips) == 0 {
		return ErrInvalid
	}
	for _, ip := range ips {
		if unsafeIP(ip.IP) {
			return ErrInvalid
		}
	}
	return nil
}
func unsafeIP(ip net.IP) bool {
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() || ip.IsMulticast() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return true
	}
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return true
	}
	addr = addr.Unmap()
	for _, prefix := range deniedArtifactPrefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

var deniedArtifactPrefixes = []netip.Prefix{
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("240.0.0.0/4"),
	netip.MustParsePrefix("2001:db8::/32"),
}

func secureEndpoint(u *url.URL) bool {
	if u == nil || u.Host == "" {
		return false
	}
	if u.Scheme == "https" {
		return true
	}
	if u.Scheme != "http" {
		return false
	}
	host := strings.Trim(strings.ToLower(u.Hostname()), "[]")
	return host == "localhost" || net.ParseIP(host) != nil && net.ParseIP(host).IsLoopback()
}
func safeArtifactClient(allowPrivate bool, proxyRawURL string) (*http.Client, error) {
	if allowPrivate {
		return &http.Client{}, nil
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	if proxyRawURL != "" {
		proxyURL, err := url.Parse(proxyRawURL)
		if err != nil {
			return nil, err
		}
		transport.Proxy = http.ProxyURL(proxyURL)
		return &http.Client{Transport: transport}, nil
	}
	transport.DialContext = func(dialCtx context.Context, _ string, address string) (net.Conn, error) {
		hostname, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, ErrInvalid
		}
		ips, err := net.DefaultResolver.LookupIPAddr(dialCtx, hostname)
		if err != nil {
			return nil, err
		}
		d := net.Dialer{Timeout: 5 * time.Second}
		for _, wantIPv4 := range []bool{true, false} {
			for _, addr := range ips {
				isIPv4 := addr.IP.To4() != nil
				if isIPv4 != wantIPv4 || unsafeIP(addr.IP) {
					continue
				}
				dialNetwork := "tcp6"
				if isIPv4 {
					dialNetwork = "tcp4"
				}
				conn, err := d.DialContext(dialCtx, dialNetwork, net.JoinHostPort(addr.IP.String(), port))
				if err == nil {
					return conn, nil
				}
			}
		}
		return nil, ErrInvalid
	}
	return &http.Client{Transport: transport}, nil
}
func errorsNew(s string) error { return fmt.Errorf("%s", s) }
