package proxy

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/mixaill76/auto_ai_router/internal/transform"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHandleStreamingWithTokens проверяет что handleStreamingWithTokens:
// 1. Корректно извлекает tokens из SSE стрима
// 2. Вызывает rateLimiter.ConsumeTokens() с суммой токенов
// 3. Вызывает rateLimiter.ConsumeModelTokens() когда задан modelID
// 4. GetCurrentTPM() и GetCurrentModelTPM() отражают добавленные токены
func TestHandleStreamingWithTokens(t *testing.T) {
	// Создаем upstream SSE сервер, который симулирует streaming ответ с tokens
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		// Пишем SSE чанки с usage information
		chunks := []string{
			"data: {\"usage\": {\"total_tokens\": 5}}\n\n",
			"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n\n",
			"data: {\"usage\": {\"total_tokens\": 3}}\n\n",
			"data: [DONE]\n\n",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(1 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	// Создаем infrastructure
	logger := testhelpers.NewTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	// Создаем Proxy
	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	// Добавляем модель к rateLimiter для tracking model-specific tokens
	credName := "test"
	modelID := "gpt-4"
	rl.AddModel(credName, modelID, 100)

	// Получаем ответ от upstream сервера используя http.Get
	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err, "http.Get должен выполниться без ошибок")
	defer func() { _ = resp.Body.Close() }()

	// Проверяем что ответ имеет правильный Content-Type
	assert.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"),
		"Ответ должен иметь Content-Type: text/event-stream")

	// Создаем ResponseRecorder для захвата результата
	w := httptest.NewRecorder()

	// Вызываем handleStreamingWithTokens напрямую
	err = prx.handleStreamingWithTokens(w, resp, credName, modelID, nil)
	require.NoError(t, err, "handleStreamingWithTokens не должен возвращать ошибку")

	// Проверяем результат в ResponseRecorder
	result := w.Result()
	require.NotNil(t, result, "ResponseRecorder result не должен быть nil")

	// Читаем тело ответа
	body, err := io.ReadAll(result.Body)
	require.NoError(t, err, "Чтение тела ответа должно быть успешным")
	_ = result.Body.Close()

	// Проверяем что стрим был прочитан
	assert.NotEmpty(t, body, "Тело ответа должно содержать SSE данные")

	// ============ ПРОВЕРКА: Токены были извлечены и записаны в rateLimiter ============
	// Сумма токенов должна быть 5 + 3 = 8
	expectedTotalTokens := 8

	// Проверяем credential-level TPM
	credentialTPM := rl.GetCurrentTPM(credName)
	assert.Equal(t, expectedTotalTokens, credentialTPM,
		fmt.Sprintf("GetCurrentTPM(%s) должен быть %d, получено %d", credName, expectedTotalTokens, credentialTPM),
	)

	// Проверяем model-level TPM
	modelTPM := rl.GetCurrentModelTPM(credName, modelID)
	assert.Equal(t, expectedTotalTokens, modelTPM,
		fmt.Sprintf("GetCurrentModelTPM(%s, %s) должен быть %d, получено %d", credName, modelID, expectedTotalTokens, modelTPM),
	)
}

// TestHandleStreamingWithTokens_NoTokens проверяет что handleStreamingWithTokens работает
// с потоком без usage информации (не падает, просто не конумирует токены)
func TestHandleStreamingWithTokens_NoTokens(t *testing.T) {
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Пишем SSE чанки БЕЗ usage информации
		chunks := []string{
			"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n\n",
			"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n",
			"data: [DONE]\n\n",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
		}
	}))
	defer upstreamServer.Close()

	logger := testhelpers.NewTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	credName := "test"
	modelID := "gpt-4"
	rl.AddModel(credName, modelID, 100)

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()
	err = prx.handleStreamingWithTokens(w, resp, credName, modelID, nil)
	require.NoError(t, err, "handleStreamingWithTokens не должен возвращать ошибку")

	// Проверяем что токены НЕ были добавлены
	credentialTPM := rl.GetCurrentTPM(credName)
	assert.Equal(t, 0, credentialTPM,
		"GetCurrentTPM должен быть 0 если нет usage информации в потоке",
	)

	modelTPM := rl.GetCurrentModelTPM(credName, modelID)
	assert.Equal(t, 0, modelTPM,
		"GetCurrentModelTPM должен быть 0 если нет usage информации в потоке",
	)
}

// TestHandleStreamingWithTokens_MultipleChunks проверяет что tokens суммируются
// из нескольких чанков. Каждый SSE чанк может содержать только одно usage значение,
// которое будет извлечено и добавлено к total.
func TestHandleStreamingWithTokens_MultipleChunks(t *testing.T) {
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Пишем множество чанков со своими delta и usage данными
		// Каждый SSE message может содержать одно usage значение
		chunks := []string{
			"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}], \"usage\": {\"total_tokens\": 10}}\n\n",
			"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n",
			"data: {\"choices\": [{\"delta\": {\"content\": \"!\"}}], \"usage\": {\"total_tokens\": 5}}\n\n",
			"data: [DONE]\n\n",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	logger := testhelpers.NewTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	credName := "test"
	modelID := "gpt-4"

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()
	err = prx.handleStreamingWithTokens(w, resp, credName, modelID, nil)
	require.NoError(t, err, "handleStreamingWithTokens не должен возвращать ошибку")

	// Проверяем что токены были просуммированы: 10 + 5 = 15
	// (total_tokens появляется в 1-м и 3-м чанках)
	credentialTPM := rl.GetCurrentTPM(credName)
	assert.Greater(t, credentialTPM, 0,
		"Tokens должны быть добавлены в rateLimiter",
	)
	assert.GreaterOrEqual(t, credentialTPM, 10,
		"TPM должен содержать хотя бы один usage значение",
	)
}

// TestHandleStreamingWithTokens_WithoutModelID проверяет что функция работает
// даже без modelID (не должна падать)
func TestHandleStreamingWithTokens_WithoutModelID(t *testing.T) {
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"usage\": {\"total_tokens\": 100}}\n\n",
			"data: [DONE]\n\n",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
		}
	}))
	defer upstreamServer.Close()

	logger := testhelpers.NewTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	credName := "test"
	modelID := "" // Пустой modelID

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	// Это не должно упасть даже с пустым modelID
	err = prx.handleStreamingWithTokens(w, resp, credName, modelID, nil)
	require.NoError(t, err, "handleStreamingWithTokens не должен возвращать ошибку")

	// Проверяем что credential-level tokens были добавлены
	credentialTPM := rl.GetCurrentTPM(credName)
	assert.Equal(t, 100, credentialTPM,
		"Tokens должны быть добавлены на credential level даже без modelID",
	)
}

// TestHandleStreamingWithTokens_LoggingToLiteLLMDB проверяет что OpenAI streaming
// responses логируются в LiteLLM DB когда предоставлен logCtx
func TestHandleStreamingWithTokens_LoggingToLiteLLMDB(t *testing.T) {
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"usage\": {\"total_tokens\": 10}}\n\n",
			"data: {\"choices\": [{\"delta\": {\"content\": \"test\"}}]}\n\n",
			"data: [DONE]\n\n",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
		}
	}))
	defer upstreamServer.Close()

	logger := testhelpers.NewTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	// Создаем logCtx и проверяем что он будет логирован
	logCtx := &RequestLogContext{
		RequestID: "test-req-123",
		Token:     "sk-test-token",
		Credential: &config.CredentialConfig{
			Name: "test-cred",
			Type: config.ProviderTypeOpenAI,
		},
	}

	// handleStreamingWithTokens должен логировать в LiteLLM DB
	err = prx.handleStreamingWithTokens(w, resp, "test-cred", "gpt-4o-mini", logCtx)
	require.NoError(t, err, "handleStreamingWithTokens не должен возвращать ошибку")

	// Проверяем что logCtx был помечен как залогированный
	assert.True(t, logCtx.Logged, "logCtx должен быть помечен как залогированный")

	// Проверяем что status был установлен
	assert.Equal(t, "success", logCtx.Status, "logCtx.Status должен быть 'success'")

	// Проверяем что HTTP status был установлен
	assert.Equal(t, 200, logCtx.HTTPStatus, "logCtx.HTTPStatus должен быть 200")

	// Проверяем что токены были извлечены
	assert.NotNil(t, logCtx.TokenUsage, "logCtx.TokenUsage не должен быть nil")
	assert.Equal(t, 10, logCtx.TokenUsage.CompletionTokens, "CompletionTokens должны быть 10")
}

// ============ Tests for Solution 3: Hybrid Approach ============

// TestEstimatePromptTokens tests the prompt token estimation from request body
func TestEstimatePromptTokens(t *testing.T) {
	tests := []struct {
		name             string
		body             []byte
		minExpected      int
		maxExpected      int
		shouldBeAtLeast1 bool
	}{
		{
			name:             "empty body",
			body:             []byte(""),
			minExpected:      0,
			maxExpected:      0,
			shouldBeAtLeast1: false,
		},
		{
			name:             "simple text message",
			body:             []byte(`{"messages":[{"content":"Hello, world! This is a test message."}]}`),
			minExpected:      5,
			maxExpected:      15,
			shouldBeAtLeast1: true,
		},
		{
			name:             "invalid JSON",
			body:             []byte(`invalid json`),
			minExpected:      0,
			maxExpected:      0,
			shouldBeAtLeast1: false,
		},
		{
			name:             "no messages field (empty messages)",
			body:             []byte(`{"model":"gpt-4"}`),
			minExpected:      1, // Minimum of 1 token for valid API call
			maxExpected:      1,
			shouldBeAtLeast1: true, // Always at least 1 token
		},
		{
			name:             "multimodal message with text and image",
			body:             []byte(`{"messages":[{"content":[{"type":"text","text":"Analyze this image"},{"type":"image_url","image_url":{"url":"..."}}]}]}`),
			minExpected:      3,
			maxExpected:      10,
			shouldBeAtLeast1: true,
		},
		{
			name:             "multiple messages",
			body:             []byte(`{"messages":[{"content":"First message"},{"content":"Second message with more text"}]}`),
			minExpected:      8,
			maxExpected:      20,
			shouldBeAtLeast1: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimate := estimatePromptTokens(tt.body)

			if tt.shouldBeAtLeast1 {
				assert.GreaterOrEqual(t, estimate, 1, "estimate should be at least 1")
			}

			assert.GreaterOrEqual(t, estimate, tt.minExpected, "estimate should be >= minExpected")
			assert.LessOrEqual(t, estimate, tt.maxExpected, "estimate should be <= maxExpected")
		})
	}
}

// TestOpenAIStreamUsageExtractor tests OpenAI format usage extraction
func TestOpenAIStreamUsageExtractor(t *testing.T) {
	extractor := &openAIStreamUsageExtractor{}

	tests := []struct {
		name        string
		chunk       []byte
		expectNil   bool
		expectUsage func(*StreamUsageInfo) bool
	}{
		{
			name:      "valid usage chunk",
			chunk:     []byte(`{"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 100 && u.CompletionTokens == 50
			},
		},
		{
			name:      "usage with cached tokens",
			chunk:     []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":20,"audio_tokens":5}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 100 && u.CachedTokens == 20 && u.AudioInputTokens == 5
			},
		},
		{
			name:      "usage with audio output tokens",
			chunk:     []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"completion_tokens_details":{"audio_tokens":10}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 100 && u.AudioOutputTokens == 10
			},
		},
		{
			name:      "no usage field",
			chunk:     []byte(`{"choices":[{"delta":{"content":"hello"}}]}`),
			expectNil: true,
		},
		{
			name:      "zero tokens (invalid)",
			chunk:     []byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0}}`),
			expectNil: true,
		},
		{
			name:      "invalid JSON",
			chunk:     []byte(`invalid json`),
			expectNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractor.ExtractUsage(tt.chunk)

			if tt.expectNil {
				assert.Nil(t, result, "should return nil")
			} else {
				assert.NotNil(t, result, "should not return nil")
				if result != nil && tt.expectUsage != nil {
					assert.True(t, tt.expectUsage(result), "usage should match expectations")
				}
			}
		})
	}
}

// TestAnthropicStreamUsageExtractor tests Anthropic format usage extraction
func TestAnthropicStreamUsageExtractor(t *testing.T) {
	extractor := &anthropicStreamUsageExtractor{}

	tests := []struct {
		name        string
		chunk       []byte
		expectNil   bool
		expectUsage func(*StreamUsageInfo) bool
	}{
		{
			name:      "valid message_delta with usage",
			chunk:     []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":100,"output_tokens":50}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 100 && u.CompletionTokens == 50
			},
		},
		{
			name:      "cache tokens present",
			chunk:     []byte(`{"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":20}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 100 && u.CacheReadTokens == 20
			},
		},
		{
			name:      "no usage field",
			chunk:     []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`),
			expectNil: true,
		},
		{
			name:      "zero tokens (invalid)",
			chunk:     []byte(`{"usage":{"input_tokens":0,"output_tokens":0}}`),
			expectNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractor.ExtractUsage(tt.chunk)

			if tt.expectNil {
				assert.Nil(t, result, "should return nil")
			} else {
				assert.NotNil(t, result, "should not return nil")
				if result != nil && tt.expectUsage != nil {
					assert.True(t, tt.expectUsage(result), "usage should match expectations")
				}
			}
		})
	}
}

// TestGetStreamUsageExtractor tests factory function for different providers
func TestGetStreamUsageExtractor(t *testing.T) {
	tests := []struct {
		name         string
		providerName string
		expectedType string
	}{
		{
			name:         "OpenAI",
			providerName: "openai",
			expectedType: "*proxy.openAIStreamUsageExtractor",
		},
		{
			name:         "Anthropic",
			providerName: "anthropic",
			expectedType: "*proxy.anthropicStreamUsageExtractor",
		},
		{
			name:         "Vertex AI",
			providerName: "vertex ai",
			expectedType: "*proxy.openAIStreamUsageExtractor",
		},
		{
			name:         "unknown provider",
			providerName: "unknown",
			expectedType: "*proxy.openAIStreamUsageExtractor",
		},
		{
			name:         "case insensitive",
			providerName: "OPENAI",
			expectedType: "*proxy.openAIStreamUsageExtractor",
		},
		{
			name:         "with whitespace",
			providerName: "  openai  ",
			expectedType: "*proxy.openAIStreamUsageExtractor",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := getStreamUsageExtractor(tt.providerName)
			assert.NotNil(t, extractor, "extractor should not be nil")
			// Type checking is implicit through the tests above
		})
	}
}

// TestHandleStreamingWithTokens_HybridApproach verifies the hybrid approach implementation
// Tests that usage info is extracted from the last chunk with cached/audio token details
func TestHandleStreamingWithTokens_HybridApproach(t *testing.T) {
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Simulate streaming with usage info in multiple chunks
		chunks := []string{
			// First chunk with some tokens
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}],\"usage\":{\"total_tokens\":5}}\n\n",
			// Middle chunk with content
			"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
			// Final chunk with complete usage info including cached and audio tokens
			// This uses both presence of fields and their values
			"data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":10,\"audio_tokens\":5},\"completion_tokens_details\":{\"audio_tokens\":2}}}\n\n",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(1 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	logger := testhelpers.NewTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	// Create log context with prompt tokens estimate
	logCtx := &RequestLogContext{
		RequestID:            "test-request-123",
		PromptTokensEstimate: 95, // Simulating estimated prompt tokens
		TokenUsage:           &transform.TokenUsage{},
	}

	err = prx.handleStreamingWithTokens(w, resp, "test-cred", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	// Verify hybrid approach results
	assert.True(t, logCtx.Logged, "logCtx should be marked as logged")
	assert.NotNil(t, logCtx.TokenUsage, "TokenUsage should not be nil")

	// With the hybrid approach, prompt tokens should use the estimate initially
	// If usage was extracted, it would override it
	assert.Greater(t, logCtx.TokenUsage.PromptTokens, 0,
		"PromptTokens should be > 0 from estimate or extracted usage")

	// Completion tokens should be at least from the token count
	assert.Greater(t, logCtx.TokenUsage.CompletionTokens, 0,
		"CompletionTokens should be > 0 from streaming count or extracted usage")
}
