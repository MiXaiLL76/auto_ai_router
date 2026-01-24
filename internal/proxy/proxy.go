package proxy

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

type Proxy struct {
	balancer       *balancer.RoundRobin
	client         *http.Client
	logger         *slog.Logger
	maxBodySizeMB  int
	requestTimeout time.Duration
	metrics        *monitoring.Metrics
}

func New(bal *balancer.RoundRobin, logger *slog.Logger, maxBodySizeMB int, requestTimeout time.Duration, metrics *monitoring.Metrics) *Proxy {
	return &Proxy{
		balancer:       bal,
		logger:         logger,
		maxBodySizeMB:  maxBodySizeMB,
		requestTimeout: requestTimeout,
		metrics:        metrics,
		client: &http.Client{
			Timeout: requestTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

func (p *Proxy) ProxyRequest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	cred, err := p.balancer.Next()
	if err != nil {
		p.logger.Error("Failed to get credential", "error", err)
		http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
		return
	}

	p.logger.Info("Selected credential",
		"credential", cred.Name,
		"method", r.Method,
		"path", r.URL.Path,
	)

	body, err := io.ReadAll(io.LimitReader(r.Body, int64(p.maxBodySizeMB)*1024*1024))
	if err != nil {
		p.logger.Error("Failed to read request body", "error", err)
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	targetURL := strings.TrimSuffix(cred.BaseURL, "/") + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		p.logger.Error("Failed to create proxy request", "error", err, "url", targetURL)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	for key, values := range r.Header {
		if key == "Authorization" {
			continue
		}
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)

	resp, err := p.client.Do(proxyReq)
	if err != nil {
		p.logger.Error("Upstream request failed",
			"credential", cred.Name,
			"error", err,
			"url", targetURL,
		)
		statusCode := http.StatusBadGateway
		p.balancer.RecordResponse(cred.Name, statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
		http.Error(w, "Bad Gateway", statusCode)
		return
	}
	defer resp.Body.Close()

	p.balancer.RecordResponse(cred.Name, resp.StatusCode)
	p.metrics.RecordRequest(cred.Name, r.URL.Path, resp.StatusCode, time.Since(start))

	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	w.WriteHeader(resp.StatusCode)

	if isStreamingResponse(resp) {
		p.handleStreaming(w, resp, cred.Name)
	} else {
		if _, err := io.Copy(w, resp.Body); err != nil {
			p.logger.Error("Failed to copy response body", "error", err)
		}
	}
}

func (p *Proxy) handleStreaming(w http.ResponseWriter, resp *http.Response, credName string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported")
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return
	}

	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				p.logger.Error("Failed to write streaming chunk",
					"error", writeErr,
					"credential", credName,
				)
				return
			}
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				p.logger.Error("Streaming read error",
					"error", err,
					"credential", credName,
				)
			}
			break
		}
	}
}

func isStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json")
}

func (p *Proxy) HealthCheck() (bool, map[string]interface{}) {
	totalCreds := len(p.balancer.GetCredentials())
	availableCreds := p.balancer.GetAvailableCount()
	bannedCreds := p.balancer.GetBannedCount()

	healthy := availableCreds > 0

	status := map[string]interface{}{
		"status":                "healthy",
		"credentials_available": availableCreds,
		"credentials_banned":    bannedCreds,
		"total_credentials":     totalCreds,
	}

	if !healthy {
		status["status"] = "unhealthy"
	}

	return healthy, status
}
