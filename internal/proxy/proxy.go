package proxy

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/vertex_transform"
)

//go:embed health.html
var healthHTML string

type Proxy struct {
	balancer       *balancer.RoundRobin
	client         *http.Client
	logger         *slog.Logger
	maxBodySizeMB  int
	requestTimeout time.Duration
	metrics        *monitoring.Metrics
	masterKey      string
	rateLimiter    *ratelimit.RPMLimiter
	tokenManager   *auth.VertexTokenManager
	healthTemplate *template.Template // Cached template
}

var (
	Version = "dev"
	Commit  = "unknown"
)

func New(bal *balancer.RoundRobin, logger *slog.Logger, maxBodySizeMB int, requestTimeout time.Duration, metrics *monitoring.Metrics, masterKey string, rateLimiter *ratelimit.RPMLimiter, tokenManager *auth.VertexTokenManager, version, commit string) *Proxy {
	// Parse template once at startup
	tmpl, err := template.New("health").Funcs(template.FuncMap{
		"div": func(a, b int) int {
			if b == 0 {
				return 0
			}
			return a / b
		},
		"mul": func(a, b int) int {
			return a * b
		},
		"version": func() string {
			return version
		},
		"commit": func() string {
			return commit
		},
	}).Parse(healthHTML)
	if err != nil {
		logger.Error("Failed to parse health template at startup", "error", err)
		// Continue without template - will cause error on /vhealth requests
	}

	return &Proxy{
		balancer:       bal,
		logger:         logger,
		maxBodySizeMB:  maxBodySizeMB,
		requestTimeout: requestTimeout,
		metrics:        metrics,
		masterKey:      masterKey,
		rateLimiter:    rateLimiter,
		tokenManager:   tokenManager,
		healthTemplate: tmpl,
		client: &http.Client{
			Timeout: requestTimeout,
			Transport: &http.Transport{
				Proxy:               http.ProxyFromEnvironment, // Support HTTP_PROXY, HTTPS_PROXY, NO_PROXY
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				DisableKeepAlives:   false,
			},
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

func (p *Proxy) ProxyRequest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Verify master_key from Authorization header
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		p.logger.Error("Missing Authorization header")
		http.Error(w, "Unauthorized: Missing Authorization header", http.StatusUnauthorized)
		return
	}

	// Extract token from "Bearer <token>"
	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == authHeader {
		// No "Bearer " prefix found
		p.logger.Error("Invalid Authorization header format")
		http.Error(w, "Unauthorized: Invalid Authorization header format", http.StatusUnauthorized)
		return
	}

	// Check if token matches master_key
	if token != p.masterKey {
		p.logger.Error("Invalid master key", "provided_key_prefix", maskKey(token))
		http.Error(w, "Unauthorized: Invalid master key", http.StatusUnauthorized)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, int64(p.maxBodySizeMB)*1024*1024))
	if err != nil {
		p.logger.Error("Failed to read request body", "error", err)
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}
	defer func() {
		if closeErr := r.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close request body", "error", closeErr)
		}
	}()

	// Extract model from request body if present
	modelID := extractModelFromBody(body)

	// Select credential based on model availability
	var cred *balancer.Credential
	if modelID != "" {
		cred, err = p.balancer.NextForModel(modelID)
	} else {
		cred, err = p.balancer.Next()
	}

	if err != nil {
		p.logger.Error("Failed to get credential", "error", err, "model", modelID)
		if errors.Is(err, balancer.ErrRateLimitExceeded) {
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
		} else {
			http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
		}
		return
	}

	// Log request details at DEBUG level
	p.logger.Debug("Processing request",
		"credential", cred.Name,
		"method", r.Method,
		"path", r.URL.Path,
		"model", modelID,
		"type", cred.Type,
	)

	// Build target URL based on credential type
	var targetURL string
	var vertexToken string
	var requestBody = body // Default to original body
	var isImageGeneration = strings.Contains(r.URL.Path, "/images/generations")

	if cred.Type == "vertex-ai" {
		// For Vertex AI, build URL dynamically
		if modelID == "" {
			p.logger.Error("Model ID is required for Vertex AI requests", "credential", cred.Name)
			http.Error(w, "Bad Request: model field is required for Vertex AI", http.StatusBadRequest)
			return
		}

		if isImageGeneration {
			// For image generation, use different endpoint
			targetURL = buildVertexImageURL(cred, modelID)
			// Transform OpenAI image request to Vertex AI format
			vertexBody, err := vertex_transform.OpenAIImageToVertex(body)
			if err != nil {
				p.logger.Error("Failed to vertex_transform image request to Vertex AI format",
					"credential", cred.Name,
					"error", err,
				)
				http.Error(w, "Internal Server Error: Failed to vertex_transform image request", http.StatusInternalServerError)
				return
			}
			requestBody = vertexBody
		} else {
			// For text generation
			streaming := isStreamingRequest(body)
			targetURL = buildVertexURL(cred, modelID, streaming)
			// Transform OpenAI request to Vertex AI format
			vertexBody, err := vertex_transform.OpenAIToVertex(body)
			if err != nil {
				p.logger.Error("Failed to vertex_transform request to Vertex AI format",
					"credential", cred.Name,
					"error", err,
				)
				http.Error(w, "Internal Server Error: Failed to vertex_transform request", http.StatusInternalServerError)
				return
			}
			requestBody = vertexBody
		}

		// Get OAuth2 token for Vertex AI
		vertexToken, err = p.tokenManager.GetToken(cred.Name, cred.CredentialsFile, cred.CredentialsJSON)
		if err != nil {
			p.logger.Error("Failed to get Vertex AI token",
				"credential", cred.Name,
				"error", err,
			)
			http.Error(w, "Internal Server Error: Failed to authenticate with Vertex AI", http.StatusInternalServerError)
			return
		}
	} else {
		// For OpenAI and other providers, use baseURL + path
		baseURL := strings.TrimSuffix(cred.BaseURL, "/")
		path := r.URL.Path

		// If baseURL already ends with /v1 and path starts with /v1, remove /v1 from path to avoid duplication
		// Example: baseURL="https://api.openai.azure.com/openai/v1" + path="/v1/chat/completions"
		// Should become: "https://api.openai.azure.com/openai/v1/chat/completions" (not /v1/v1/...)
		if strings.HasSuffix(baseURL, "/v1") && strings.HasPrefix(path, "/v1") {
			path = strings.TrimPrefix(path, "/v1")
		}

		targetURL = baseURL + path
		if r.URL.RawQuery != "" {
			targetURL += "?" + r.URL.RawQuery
		}
	}

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(requestBody))
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

	// Set Authorization header based on credential type
	if cred.Type == "vertex-ai" {
		// For Vertex AI, use OAuth2 token
		proxyReq.Header.Set("Authorization", "Bearer "+vertexToken)
	} else {
		// For OpenAI and other providers, use API key
		proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
	}

	// Detailed debug logging (truncate long fields for readability)
	p.logger.Debug("Proxy request details",
		"target_url", targetURL,
		"credential", cred.Name,
		"request_body", truncateLongFields(string(requestBody), 500),
	)

	// Log headers (mask Authorization for security)
	debugHeaders := make(map[string]string)
	for key, values := range proxyReq.Header {
		if key == "Authorization" {
			debugHeaders[key] = "Bearer " + maskKey(cred.APIKey)
		} else {
			debugHeaders[key] = strings.Join(values, ", ")
		}
	}
	p.logger.Debug("Proxy request headers", "headers", debugHeaders)

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
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close response body", "error", closeErr)
		}
	}()

	p.balancer.RecordResponse(cred.Name, resp.StatusCode)
	p.metrics.RecordRequest(cred.Name, r.URL.Path, resp.StatusCode, time.Since(start))

	// Log response headers
	debugRespHeaders := make(map[string]string)
	for key, values := range resp.Header {
		debugRespHeaders[key] = strings.Join(values, ", ")
	}
	p.logger.Debug("Proxy response received",
		"status_code", resp.StatusCode,
		"credential", cred.Name,
		"headers", debugRespHeaders,
	)

	// Read and log response body for non-streaming responses
	var responseBody []byte
	var finalResponseBody []byte // Define here for broader scope
	if isStreamingResponse(resp) {
		p.logger.Debug("Response is streaming", "credential", cred.Name)
	} else {
		responseBody, err = io.ReadAll(resp.Body)
		if err != nil {
			p.logger.Error("Failed to read response body", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		// Decode the response body for logging (handles gzip, etc.)
		contentEncoding := resp.Header.Get("Content-Encoding")
		decodedBody := decodeResponseBody(responseBody, contentEncoding)

		// Extract and record token usage
		tokens := extractTokensFromResponse(decodedBody, cred.Type)
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(cred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
			}
			p.logger.Debug("Token usage recorded",
				"credential", cred.Name,
				"model", modelID,
				"tokens", tokens,
			)
		}

		// Transform Vertex AI response back to OpenAI format
		if cred.Type == "vertex-ai" {
			if isImageGeneration {
				// Transform image response
				vertex_transformedBody, err := vertex_transform.VertexImageToOpenAI([]byte(decodedBody))
				if err != nil {
					p.logger.Error("Failed to vertex_transform Vertex AI image response",
						"credential", cred.Name,
						"error", err,
					)
					finalResponseBody = responseBody
				} else {
					finalResponseBody = vertex_transformedBody
					p.logger.Debug("Transformed Vertex AI image response to OpenAI format",
						"credential", cred.Name,
						"vertex_transformed_body", truncateLongFields(string(vertex_transformedBody), 200),
					)
				}
			} else {
				// Transform text response
				vertex_transformedBody, err := vertex_transform.VertexToOpenAI([]byte(decodedBody), modelID)
				if err != nil {
					p.logger.Error("Failed to vertex_transform Vertex AI response",
						"credential", cred.Name,
						"error", err,
					)
					finalResponseBody = responseBody
				} else {
					finalResponseBody = vertex_transformedBody
					p.logger.Debug("Transformed Vertex AI response to OpenAI format",
						"credential", cred.Name,
						"vertex_transformed_body", truncateLongFields(string(vertex_transformedBody), 500),
					)
				}
			}
		} else {
			finalResponseBody = responseBody
		}

		p.logger.Debug("Proxy response body",
			"credential", cred.Name,
			"content_encoding", contentEncoding,
			"body", truncateLongFields(decodedBody, 500),
		)
		// Replace resp.Body with a new reader for subsequent processing
		resp.Body = io.NopCloser(bytes.NewReader(finalResponseBody))
	}

	for key, values := range resp.Header {
		// Skip Content-Length and Content-Encoding as we may have vertex_transformed the response
		if key == "Content-Length" && cred.Type == "vertex-ai" {
			continue
		}
		if key == "Content-Encoding" && cred.Type == "vertex-ai" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Set correct Content-Length for vertex_transformed responses
	if cred.Type == "vertex-ai" && len(finalResponseBody) > 0 {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(finalResponseBody)))
	}

	w.WriteHeader(resp.StatusCode)

	if isStreamingResponse(resp) {
		if cred.Type == "vertex-ai" {
			// Transform Vertex AI streaming to OpenAI format
			err := vertex_transform.TransformVertexStreamToOpenAI(resp.Body, modelID, w)
			if err != nil {
				p.logger.Error("Failed to vertex_transform streaming response", "error", err)
			}
		} else {
			p.handleStreaming(w, resp, cred.Name)
		}
	} else {
		if _, err := io.Copy(w, resp.Body); err != nil {
			p.logger.Error("Failed to copy response body", "error", err)
		}
	}
}

func (p *Proxy) handleStreaming(w http.ResponseWriter, resp *http.Response, credName string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported", "credential", credName)
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return
	}

	p.logger.Debug("Starting streaming response", "credential", credName)

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

	p.logger.Debug("Streaming response completed", "credential", credName)
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

	// Collect credentials info
	credentialsInfo := make(map[string]interface{})
	for _, cred := range p.balancer.GetCredentials() {
		credentialsInfo[cred.Name] = map[string]interface{}{
			"current_rpm": p.rateLimiter.GetCurrentRPM(cred.Name),
			"current_tpm": p.rateLimiter.GetCurrentTPM(cred.Name),
			"limit_rpm":   cred.RPM,
			"limit_tpm":   cred.TPM,
		}
	}

	// Collect models info
	modelsInfo := make(map[string]interface{})
	for _, modelKey := range p.rateLimiter.GetAllModels() {
		// Parse "credential:model" format
		parts := strings.Split(modelKey, ":")
		if len(parts) != 2 {
			continue
		}
		credentialName := parts[0]
		modelName := parts[1]

		// Use "credential:model" as key to handle same model across different credentials
		modelsInfo[modelKey] = map[string]interface{}{
			"credential":  credentialName,
			"model":       modelName,
			"current_rpm": p.rateLimiter.GetCurrentModelRPM(credentialName, modelName),
			"current_tpm": p.rateLimiter.GetCurrentModelTPM(credentialName, modelName),
			"limit_rpm":   p.rateLimiter.GetModelLimitRPM(credentialName, modelName),
			"limit_tpm":   p.rateLimiter.GetModelLimitTPM(credentialName, modelName),
		}
	}

	status := map[string]interface{}{
		"status":                "healthy",
		"credentials_available": availableCreds,
		"credentials_banned":    bannedCreds,
		"total_credentials":     totalCreds,
		"credentials":           credentialsInfo,
		"models":                modelsInfo,
	}

	if !healthy {
		status["status"] = "unhealthy"
	}

	return healthy, status
}

// extractModelFromBody extracts the model ID from the request body
func extractModelFromBody(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	var reqBody struct {
		Model string `json:"model"`
	}

	if err := json.Unmarshal(body, &reqBody); err != nil {
		return ""
	}

	return reqBody.Model
}

// decodeResponseBody decodes the response body based on Content-Encoding
func decodeResponseBody(body []byte, encoding string) string {
	// Check if response is gzip-encoded
	if strings.Contains(strings.ToLower(encoding), "gzip") {
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return string(body) // Return as-is if can't decode
		}
		defer func() {
			_ = reader.Close()
		}()

		decoded, err := io.ReadAll(reader)
		if err != nil {
			return string(body) // Return as-is if can't read
		}
		return string(decoded)
	}

	// Return as plain text
	return string(body)
}

// extractTokensFromResponse extracts total_tokens from the response body
// Supports both OpenAI format (usage.total_tokens) and Vertex AI format (usageMetadata.totalTokenCount)
func extractTokensFromResponse(body string, credType string) int {
	if body == "" {
		return 0
	}

	// For Vertex AI, use usageMetadata format
	if credType == "vertex-ai" {
		var vertexResp struct {
			UsageMetadata struct {
				TotalTokenCount int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}

		if err := json.Unmarshal([]byte(body), &vertexResp); err != nil {
			return 0
		}

		return vertexResp.UsageMetadata.TotalTokenCount
	}

	// For OpenAI and other providers, use standard format
	var openAIResp struct {
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal([]byte(body), &openAIResp); err != nil {
		return 0
	}

	return openAIResp.Usage.TotalTokens
}

// truncateLongFields truncates long fields in JSON for logging purposes
// This prevents extremely long base64 strings, embeddings, etc. from cluttering logs
func truncateLongFields(body string, maxFieldLength int) string {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(body), &data); err != nil {
		return body // Return as-is if not valid JSON
	}

	truncateValue(data, maxFieldLength)

	truncated, err := json.Marshal(data)
	if err != nil {
		return body // Return original if marshaling fails
	}

	return string(truncated)
}

// truncateValue recursively truncates long string values in a map or slice
func truncateValue(v interface{}, maxLength int) {
	switch val := v.(type) {
	case map[string]interface{}:
		for key, value := range val {
			switch key {
			case "embedding", "b64_json", "content":
				// Truncate known long fields more aggressively
				if str, ok := value.(string); ok && len(str) > 50 {
					val[key] = fmt.Sprintf("%s... [truncated %d chars]", str[:50], len(str)-50)
				}
			case "messages":
				// For messages array, truncate each message content
				if arr, ok := value.([]interface{}); ok {
					for i := range arr {
						truncateValue(arr[i], maxLength)
					}
				}
			default:
				// For other fields, use standard truncation or recurse
				if str, ok := value.(string); ok && len(str) > maxLength {
					val[key] = str[:maxLength] + "... [truncated]"
				} else {
					truncateValue(value, maxLength)
				}
			}
		}
	case []interface{}:
		for _, item := range val {
			truncateValue(item, maxLength)
		}
	}
}

// maskKey masks the API key for logging (shows only first 7 chars)
func maskKey(key string) string {
	if len(key) <= 7 {
		return "***"
	}
	return key[:7] + "..."
}

// determineVertexPublisher determines the Vertex AI publisher based on the model ID
func determineVertexPublisher(modelID string) string {
	modelLower := strings.ToLower(modelID)
	if strings.Contains(modelLower, "claude") {
		return "anthropic"
	}
	// Default to Google for Gemini and other models
	return "google"
}

// buildVertexImageURL constructs the Vertex AI URL for image generation
// Format: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func buildVertexImageURL(cred *balancer.Credential, modelID string) string {
	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1/projects/%s/locations/global/publishers/google/models/%s:predict",
			cred.ProjectID, modelID,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		cred.Location, cred.ProjectID, cred.Location, modelID,
	)
}

// buildVertexURL constructs the Vertex AI URL dynamically
// Format: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:{endpoint}
func buildVertexURL(cred *balancer.Credential, modelID string, streaming bool) string {
	publisher := determineVertexPublisher(modelID)

	endpoint := "generateContent"
	if streaming {
		endpoint = "streamGenerateContent?alt=sse"
	}

	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1/projects/%s/locations/global/publishers/%s/models/%s:%s",
			cred.ProjectID, publisher, modelID, endpoint,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/%s/models/%s:%s",
		cred.Location, cred.ProjectID, cred.Location, publisher, modelID, endpoint,
	)
}

// isStreamingRequest determines if the request is for streaming based on the request body
func isStreamingRequest(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	var reqBody struct {
		Stream bool `json:"stream"`
	}

	if err := json.Unmarshal(body, &reqBody); err != nil {
		return false
	}

	return reqBody.Stream
}

// VisualHealthCheck renders an HTML dashboard with health check information
func (p *Proxy) VisualHealthCheck(w http.ResponseWriter, r *http.Request) {
	_, status := p.HealthCheck()

	if p.healthTemplate == nil {
		p.logger.Error("Health template not available")
		http.Error(w, "Internal Server Error: Template not available", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)

	if err := p.healthTemplate.Execute(w, status); err != nil {
		p.logger.Error("Failed to execute health template", "error", err)
	}
}
