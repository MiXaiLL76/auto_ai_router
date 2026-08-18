package sosana

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	PollInterval              = 2 * time.Second
	MaxResultImageBytes int64 = 32 * 1024 * 1024
	MaxResultErrorBytes int64 = 16 * 1024
)

var allowPrivateResultURLForTests func(*url.URL) bool

type TaskHTTPResult struct {
	Task       BananaTaskResponse
	RawBody    []byte
	StatusCode int
}

type ResultImage struct {
	Bytes       []byte
	ContentType string
	Host        string
}

type ResultImageError struct {
	StatusCode         int
	ResponseBody       []byte
	Host               string
	ContentType        string
	SniffedContentType string
	UpstreamStatus     int
	Err                error
}

func (e *ResultImageError) Error() string {
	if e.Err != nil {
		return e.Err.Error()
	}
	return "sosana result image download failed"
}

func (e *ResultImageError) Unwrap() error {
	return e.Err
}

func SetAllowPrivateResultURLForTests(fn func(*url.URL) bool) func() {
	previous := allowPrivateResultURLForTests
	allowPrivateResultURLForTests = fn
	return func() {
		allowPrivateResultURLForTests = previous
	}
}

func DoTaskRequest(ctx context.Context, client *http.Client, method, url string, apiKey string, body []byte) (TaskHTTPResult, error) {
	var reader *bytes.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	} else {
		reader = bytes.NewReader(nil)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return TaskHTTPResult{StatusCode: http.StatusInternalServerError}, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return TaskHTTPResult{StatusCode: http.StatusBadGateway}, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	rawBody, err := readLimitedResultBody(resp.Body, MaxResultImageBytes)
	if err != nil {
		return TaskHTTPResult{StatusCode: http.StatusBadGateway}, err
	}
	var task BananaTaskResponse
	if len(rawBody) > 0 {
		_ = json.Unmarshal(rawBody, &task)
	}
	return TaskHTTPResult{Task: task, RawBody: rawBody, StatusCode: resp.StatusCode}, nil
}

func DownloadResultImage(ctx context.Context, client *http.Client, task BananaTaskResponse) (ResultImage, error) {
	resultURL := ""
	if task.ResultFileURL != nil {
		resultURL = strings.TrimSpace(*task.ResultFileURL)
	}
	parsed, err := parseResultURL(resultURL)
	if err != nil {
		return ResultImage{}, resultImageError(http.StatusBadGateway, "", err)
	}
	host := parsed.Hostname()
	if err := validateResultURL(ctx, parsed); err != nil {
		return ResultImage{}, resultImageError(http.StatusBadGateway, host, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, resultURL, nil)
	if err != nil {
		return ResultImage{}, resultImageError(http.StatusBadGateway, host, err)
	}
	req.Header.Set("Accept", "image/*")

	resp, err := doResultImageRequest(client, req)
	if err != nil {
		statusCode := http.StatusBadGateway
		if isResultTimeout(ctx, err) {
			statusCode = http.StatusRequestTimeout
		}
		return ResultImage{}, resultImageError(statusCode, host, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	contentType := strings.TrimSpace(resp.Header.Get("Content-Type"))
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return ResultImage{}, &ResultImageError{
			StatusCode:     http.StatusBadGateway,
			ResponseBody:   readTextResultBody(resp.Body, contentType),
			Host:           host,
			ContentType:    contentType,
			UpstreamStatus: resp.StatusCode,
			Err:            errors.New("sosana result image download returned error status"),
		}
	}

	image, err := readLimitedResultImage(resp.Body)
	if err != nil {
		return ResultImage{}, resultImageError(http.StatusBadGateway, host, err)
	}
	sniffedType := http.DetectContentType(image)
	if !IsPNGContentType(contentType) && !IsPNGContentType(sniffedType) {
		return ResultImage{}, &ResultImageError{
			StatusCode:         http.StatusBadGateway,
			ResponseBody:       textResultBodyPrefix(image, contentType, sniffedType),
			Host:               host,
			ContentType:        contentType,
			SniffedContentType: sniffedType,
			Err:                errors.New("sosana result URL returned non-PNG content"),
		}
	}
	if !IsPNGContentType(contentType) {
		contentType = sniffedType
	}
	return ResultImage{Bytes: image, ContentType: contentType, Host: host}, nil
}

func ResultHost(task BananaTaskResponse) string {
	if task.ResultFileURL == nil {
		return ""
	}
	parsed, err := url.Parse(strings.TrimSpace(*task.ResultFileURL))
	if err != nil {
		return ""
	}
	return parsed.Hostname()
}

func IsUnsafeResultIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if v4 := ip.To4(); v4 != nil && v4[0] == 100 && v4[1]&0xc0 == 64 {
		return true
	}
	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsMulticast() ||
		ip.IsUnspecified()
}

func IsPNGContentType(contentType string) bool {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	return contentType == "image/png" || strings.HasPrefix(contentType, "image/png;")
}

func isTextContentType(contentType string) bool {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	return strings.HasPrefix(contentType, "text/") ||
		strings.Contains(contentType, "json") ||
		strings.Contains(contentType, "xml")
}

func resultImageError(statusCode int, host string, err error) *ResultImageError {
	return &ResultImageError{StatusCode: statusCode, Host: host, Err: err}
}

func doResultImageRequest(client *http.Client, req *http.Request) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}
	resultClient := *client
	resultClient.Transport = resultImageTransport()
	resultClient.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return resultClient.Do(req)
}

func resultImageTransport() http.RoundTripper {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DisableKeepAlives = true
	transport.DialContext = dialResultAddress
	return transport
}

func dialResultAddress(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}

	dialer := &net.Dialer{}
	if allowPrivateResultHostForTests(host) {
		return dialer.DialContext(ctx, network, address)
	}
	if ip := net.ParseIP(host); ip != nil {
		if IsUnsafeResultIP(ip) {
			return nil, errors.New("sosana result_file_url resolves to a private address")
		}
		return dialer.DialContext(ctx, network, address)
	}

	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, err
	}
	if len(addrs) == 0 {
		return nil, errors.New("sosana result_file_url host has no addresses")
	}
	for _, addr := range addrs {
		if IsUnsafeResultIP(addr.IP) {
			return nil, errors.New("sosana result_file_url resolves to a private address")
		}
	}

	var dialErr error
	for _, addr := range addrs {
		conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(addr.IP.String(), port))
		if err == nil {
			return conn, nil
		}
		dialErr = err
	}
	if dialErr != nil {
		return nil, dialErr
	}
	return nil, errors.New("sosana result_file_url host has no dialable addresses")
}

func allowPrivateResultHostForTests(host string) bool {
	if allowPrivateResultURLForTests == nil {
		return false
	}
	return allowPrivateResultURLForTests(&url.URL{Scheme: "http", Host: host})
}

func parseResultURL(raw string) (*url.URL, error) {
	if raw == "" {
		return nil, errors.New("sosana task completed without result_file_url")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return nil, err
	}
	if parsed.Hostname() == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("sosana result_file_url must be an http or https URL")
	}
	return parsed, nil
}

func validateResultURL(ctx context.Context, parsed *url.URL) error {
	if allowPrivateResultURLForTests != nil && allowPrivateResultURLForTests(parsed) {
		return nil
	}
	if parsed.Scheme != "https" {
		return errors.New("sosana result_file_url must use https")
	}

	host := parsed.Hostname()
	if strings.EqualFold(host, "localhost") {
		return errors.New("sosana result_file_url host is not allowed")
	}
	if !isAllowedResultHost(host) {
		return errors.New("sosana result_file_url host is not allowed")
	}
	if ip := net.ParseIP(host); ip != nil {
		if IsUnsafeResultIP(ip) {
			return errors.New("sosana result_file_url resolves to a private address")
		}
		return nil
	}

	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return err
	}
	if len(addrs) == 0 {
		return errors.New("sosana result_file_url host has no addresses")
	}
	for _, addr := range addrs {
		if IsUnsafeResultIP(addr.IP) {
			return errors.New("sosana result_file_url resolves to a private address")
		}
	}
	return nil
}

func isAllowedResultHost(host string) bool {
	host = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(host)), ".")
	for _, suffix := range []string{
		"sosana.blog",
		"sosana.art",
		"storage.yandexcloud.net",
	} {
		if host == suffix || strings.HasSuffix(host, "."+suffix) {
			return true
		}
	}
	return false
}

func readLimitedResultImage(body io.Reader) ([]byte, error) {
	data, err := readLimitedResultBody(body, MaxResultImageBytes)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, errors.New("sosana result image body is empty")
	}
	return data, nil
}

func readLimitedResultBody(body io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, errors.New("sosana response body is too large")
	}
	return data, nil
}

func readTextResultBody(body io.Reader, contentType string) []byte {
	if !isTextContentType(contentType) {
		return nil
	}
	data, err := io.ReadAll(io.LimitReader(body, MaxResultErrorBytes))
	if err != nil {
		return nil
	}
	return data
}

func textResultBodyPrefix(body []byte, contentTypes ...string) []byte {
	for _, contentType := range contentTypes {
		if isTextContentType(contentType) {
			if int64(len(body)) > MaxResultErrorBytes {
				return body[:MaxResultErrorBytes]
			}
			return body
		}
	}
	return nil
}

func isResultTimeout(ctx context.Context, err error) bool {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}
