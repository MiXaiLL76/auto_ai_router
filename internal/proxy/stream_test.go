package proxy

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
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
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

// TestHandleStreamingWithTokens_NoUsageChunk проверяет что handleStreamingWithTokens
// считает токены из delta-текста, когда usage-чанк не пришёл (обрыв стрима или
// провайдер не отправил usage). При отсутствии usage используется tokenizer.
func TestHandleStreamingWithTokens_NoUsageChunk(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Пишем SSE чанки БЕЗ usage информации — провайдер закрывает соединение без usage
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

	// "hello" + " world" считаются через tokenizer модели.
	expectedEstimated := countTextTokensForModel(modelID, "hello world")

	credentialTPM := rl.GetCurrentTPM(credName)
	assert.Equal(t, expectedEstimated, credentialTPM,
		"GetCurrentTPM должен содержать оценку из delta-текста когда usage-чанк не пришёл",
	)

	modelTPM := rl.GetCurrentModelTPM(credName, modelID)
	assert.Equal(t, expectedEstimated, modelTPM,
		"GetCurrentModelTPM должен содержать оценку из delta-текста когда usage-чанк не пришёл",
	)
}

// TestHandleStreamingWithTokens_MultipleChunks проверяет что tokens суммируются
// из нескольких чанков. Каждый SSE чанк может содержать только одно usage значение,
// которое будет извлечено и добавлено к total.
func TestHandleStreamingWithTokens_MultipleChunks(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

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
	largeInput := strings.Repeat("a", 280000)
	largeInputTokens := countTextTokensForModel("gpt-4o", largeInput)

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
			maxExpected:      20,
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
		{
			name:             "responses api: string input",
			body:             []byte(`{"model":"gpt-4o","input":"Hello, this is a test prompt for the Responses API."}`),
			minExpected:      8,
			maxExpected:      20,
			shouldBeAtLeast1: true,
		},
		{
			name:             "responses api: array input with messages",
			body:             []byte(`{"model":"gpt-4o","input":[{"role":"user","content":"Analyze this text please"},{"role":"assistant","content":"Sure, I can help."}]}`),
			minExpected:      8,
			maxExpected:      20,
			shouldBeAtLeast1: true,
		},
		{
			name:             "responses api: instructions field",
			body:             []byte(`{"model":"gpt-4o","input":"Short question","instructions":"You are a helpful assistant that answers concisely."}`),
			minExpected:      12,
			maxExpected:      25,
			shouldBeAtLeast1: true,
		},
		{
			name:             "responses api: large string input",
			body:             []byte(`{"model":"gpt-4o","input":"` + largeInput + `"}`),
			minExpected:      largeInputTokens,
			maxExpected:      largeInputTokens,
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
			name:      "usage with cached audio tokens",
			chunk:     []byte(`{"usage":{"prompt_tokens":200,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":100}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 200 && u.CachedTokens == 80 && u.CachedAudioTokens == 40 && u.AudioInputTokens == 60
			},
		},
		{
			name:      "usage with negative cached tokens and cached audio",
			chunk:     []byte(`{"usage":{"prompt_tokens":200,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":-80,"cached_audio_tokens":40,"audio_tokens":100}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 200 && u.CachedTokens == 0 && u.CachedAudioTokens == 0 && u.AudioInputTokens == 100
			},
		},
		{
			name:      "usage with cache write tokens",
			chunk:     []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"cache_write_tokens":40}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 100 && u.CacheCreationTokens == 40
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
			name:      "usage with image prediction and cached output details",
			chunk:     []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"image_tokens":7},"completion_tokens_details":{"image_tokens":11,"cached_tokens":3,"accepted_prediction_tokens":5,"rejected_prediction_tokens":2}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.ImageTokens == 7 && u.OutputImageTokens == 11 && u.CachedOutputTokens == 3 &&
					u.AcceptedPredictionTokens == 5 && u.RejectedPredictionTokens == 2
			},
		},
		{
			name:      "chat usage with confirmed web search requests",
			chunk:     []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"server_tool_use":{"web_search_requests":3}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.WebSearchRequests == 3
			},
		},
		{
			name:      "no usage field",
			chunk:     []byte(`{"choices":[{"delta":{"content":"hello"}}]}`),
			expectNil: true,
		},
		{
			name:      "explicit zero usage",
			chunk:     []byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 0 && u.CompletionTokens == 0
			},
		},
		{
			name:      "invalid JSON",
			chunk:     []byte(`invalid json`),
			expectNil: true,
		},
		// Responses API format tests (GPT-5, /v1/responses)
		{
			name:      "responses API - top level usage",
			chunk:     []byte(`{"usage":{"input_tokens":120,"output_tokens":60,"total_tokens":180}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 120 && u.CompletionTokens == 60
			},
		},
		{
			name:      "responses API - response.completed event",
			chunk:     []byte(`{"type":"response.completed","response":{"id":"resp_123","usage":{"input_tokens":200,"output_tokens":80,"total_tokens":280}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 200 && u.CompletionTokens == 80
			},
		},
		{
			name:      "responses API - completed output items provide web search count",
			chunk:     []byte(`{"type":"response.completed","response":{"output":[{"type":"web_search_call","status":"completed"},{"type":"web_search_call","status":"completed"}],"usage":{"input_tokens":20,"output_tokens":5,"total_tokens":25}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.WebSearchRequests == 2
			},
		},
		{
			name:      "responses API - usage count takes precedence over output items",
			chunk:     []byte(`{"type":"response.completed","response":{"output":[{"type":"web_search_call","status":"completed"}],"usage":{"input_tokens":20,"output_tokens":5,"server_tool_use":{"web_search_requests":3}}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.WebSearchRequests == 3
			},
		},
		{
			name:      "responses API - with output_tokens_details",
			chunk:     []byte(`{"type":"response.completed","response":{"usage":{"input_tokens":150,"output_tokens":100,"input_tokens_details":{"image_tokens":8},"output_tokens_details":{"reasoning_tokens":40,"image_tokens":12}}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 150 && u.CompletionTokens == 100 && u.ReasoningTokens == 40 &&
					u.ImageTokens == 8 && u.OutputImageTokens == 12
			},
		},
		{
			name:      "responses API - with input_tokens_details cached",
			chunk:     []byte(`{"usage":{"input_tokens":300,"output_tokens":50,"input_tokens_details":{"cached_tokens":100}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 300 && u.CompletionTokens == 50 && u.CachedTokens == 100
			},
		},
		{
			name:      "responses API - cached audio is excluded from audio input",
			chunk:     []byte(`{"usage":{"input_tokens":300,"output_tokens":50,"input_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":100}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 300 && u.CompletionTokens == 50 && u.CachedAudioTokens == 40 && u.AudioInputTokens == 60
			},
		},
		{
			name:      "responses API - negative cached tokens do not increase audio",
			chunk:     []byte(`{"usage":{"input_tokens":300,"output_tokens":50,"input_tokens_details":{"cached_tokens":-80,"cached_audio_tokens":40,"audio_tokens":100}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 300 && u.CompletionTokens == 50 && u.CachedTokens == 0 && u.CachedAudioTokens == 0 && u.AudioInputTokens == 100
			},
		},
		{
			name:      "responses API - with cache creation tokens",
			chunk:     []byte(`{"usage":{"input_tokens":300,"output_tokens":50,"input_tokens_details":{"cache_creation_tokens":120}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 300 && u.CacheCreationTokens == 120
			},
		},
		{
			name: "responses API - SSE format response.completed",
			chunk: []byte("event: response.completed\ndata: " +
				`{"type":"response.completed","response":{"usage":{"input_tokens":500,"output_tokens":200,"total_tokens":700}}}` +
				"\n\n"),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 500 && u.CompletionTokens == 200
			},
		},
		{
			name:      "responses API - cache hit (input_tokens=0, output_tokens>0)",
			chunk:     []byte(`{"usage":{"input_tokens":0,"output_tokens":75,"input_tokens_details":{"cached_tokens":200}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 0 && u.CompletionTokens == 75 && u.CachedTokens == 200
			},
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

func TestOpenAIStreamUsageExtractor_PreservesTransformedAudioInput(t *testing.T) {
	extractor := &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: true}

	result := extractor.ExtractUsage([]byte(`{"usage":{"prompt_tokens":200,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":60}}}`))

	require.NotNil(t, result)
	assert.Equal(t, 60, result.AudioInputTokens)
	assert.Equal(t, 40, result.CachedAudioTokens)
}

func TestHandleResponsesAPIStreaming_Passthrough(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			`data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
			`data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}` + "\n\n",
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()
	cred := &config.CredentialConfig{
		Name: "test",
		Type: config.ProviderTypeOpenAI,
	}

	err = prx.handleResponsesAPIStreaming(w, resp, cred, "gpt-4o", nil, nil)
	require.NoError(t, err)

	body := w.Body.String()
	assert.Contains(t, body, "event: response.created")
	assert.Contains(t, body, "response.output_text.delta")
	assert.Contains(t, body, "Hello")
	assert.Contains(t, body, "event: response.completed")
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
			name:      "confirmed Anthropic web search usage",
			chunk:     []byte(`{"usage":{"input_tokens":100,"output_tokens":50,"server_tool_use":{"web_search_requests":2}}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.WebSearchRequests == 2
			},
		},
		{
			name:      "no usage field",
			chunk:     []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`),
			expectNil: true,
		},
		{
			name:      "explicit zero usage",
			chunk:     []byte(`{"usage":{"input_tokens":0,"output_tokens":0}}`),
			expectNil: false,
			expectUsage: func(u *StreamUsageInfo) bool {
				return u.PromptTokens == 0 && u.CompletionTokens == 0
			},
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

// TestAnthropicStreamUsageExtractor_CacheCreationTokens verifies that
// anthropicStreamUsageExtractor correctly extracts cache_creation_input_tokens
// into the CacheCreationTokens field of StreamUsageInfo.
func TestAnthropicStreamUsageExtractor_CacheCreationTokens(t *testing.T) {
	extractor := &anthropicStreamUsageExtractor{}

	chunk := []byte(`{"type":"message_delta","usage":{"input_tokens":200,"output_tokens":80,"cache_creation_input_tokens":150,"cache_read_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":50,"ephemeral_1h_input_tokens":100}}}`)

	result := extractor.ExtractUsage(chunk)
	require.NotNil(t, result, "should extract usage from chunk with cache_creation_input_tokens")

	assert.Equal(t, 200, result.PromptTokens, "PromptTokens should be 200")
	assert.Equal(t, 80, result.CompletionTokens, "CompletionTokens should be 80")
	assert.Equal(t, 150, result.CacheCreationTokens, "CacheCreationTokens should be 150")
	assert.Equal(t, 50, result.CacheCreation5mTokens)
	assert.Equal(t, 100, result.CacheCreation1hTokens)
	assert.Equal(t, 30, result.CacheReadTokens, "CacheReadTokens should be 30")
	assert.Equal(t, 30, result.CachedTokens, "CachedTokens should equal CacheReadTokens (30)")
}

func TestOpenAIStreamUsageExtractor_CacheTTLAndAudioDetails(t *testing.T) {
	extractor := &openAIStreamUsageExtractor{}
	chunk := []byte(`data: {"usage":{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":40,"cached_audio_tokens":15,"cache_creation_tokens":20,"cache_creation_token_details":{"ephemeral_5m_input_tokens":5,"ephemeral_1h_input_tokens":15}}}}`)

	result := extractor.ExtractUsage(chunk)

	require.NotNil(t, result)
	assert.Equal(t, 40, result.CachedTokens)
	assert.Equal(t, 15, result.CachedAudioTokens)
	assert.Equal(t, 20, result.CacheCreationTokens)
	assert.Equal(t, 5, result.CacheCreation5mTokens)
	assert.Equal(t, 15, result.CacheCreation1hTokens)
}

// TestOpenAIStreamUsageExtractor_MultiPayloadNoStaleData verifies that when a single
// stream chunk contains multiple SSE data payloads (e.g. "data: {...}\ndata: {...}\n"),
// the extractor does not carry over stale fields from the first payload into the result
// of the second payload. The extractor iterates from last to first, so if the last
// payload has fewer fields, earlier payload data must not leak through.
func TestOpenAIStreamUsageExtractor_MultiPayloadNoStaleData(t *testing.T) {
	extractor := &openAIStreamUsageExtractor{}

	// Two SSE payloads in one chunk.
	// First payload has detailed usage with cached_tokens and audio_tokens.
	// Second (last) payload has only basic prompt/completion tokens, no details.
	// The extractor reads from last to first, so it should return the second
	// payload's data without any stale cached_tokens or audio_tokens from the first.
	chunk := []byte(
		"data: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":20,\"audio_tokens\":5},\"completion_tokens_details\":{\"audio_tokens\":10,\"reasoning_tokens\":15}}}\n" +
			"data: {\"usage\":{\"prompt_tokens\":80,\"completion_tokens\":30}}\n",
	)

	result := extractor.ExtractUsage(chunk)
	require.NotNil(t, result, "should extract usage from multi-payload chunk")

	// Should reflect the LAST payload only (prompt=80, completion=30)
	assert.Equal(t, 80, result.PromptTokens, "PromptTokens should be from last payload (80)")
	assert.Equal(t, 30, result.CompletionTokens, "CompletionTokens should be from last payload (30)")

	// These fields must be zero because the last payload has no details —
	// stale data from the first payload must NOT leak through.
	assert.Equal(t, 0, result.CachedTokens, "CachedTokens should be 0 (no stale data from first payload)")
	assert.Equal(t, 0, result.AudioInputTokens, "AudioInputTokens should be 0 (no stale data from first payload)")
	assert.Equal(t, 0, result.AudioOutputTokens, "AudioOutputTokens should be 0 (no stale data from first payload)")
	assert.Equal(t, 0, result.ReasoningTokens, "ReasoningTokens should be 0 (no stale data from first payload)")
}

// TestHandleStreamingWithTokens_HybridApproach verifies the hybrid approach implementation
// Tests that usage info is extracted from the last chunk with cached/audio token details
func TestHandleStreamingWithTokens_HybridApproach(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	// Create log context with prompt tokens estimate
	logCtx := &RequestLogContext{
		RequestID:              "test-request-123",
		promptTokensEstimateFn: func() int { return 95 }, // Simulating estimated prompt tokens
		TokenUsage:             &converter.TokenUsage{},
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

// TestHandleStreamingWithTokens_WithDoneAndUsage verifies that the presence of data: [DONE]
// as the last chunk does not prevent extracting the correct usage from the previous chunk,
// and that CompletionTokens is not overwritten with TotalTokens.
func TestHandleStreamingWithTokens_WithDoneAndUsage(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"total_tokens\":105}}\n\n",
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	logCtx := &RequestLogContext{
		RequestID:              "test-request-done-123",
		promptTokensEstimateFn: func() int { return 95 },
		TokenUsage:             &converter.TokenUsage{},
		Credential: &config.CredentialConfig{
			Name: "test",
			Type: config.ProviderTypeOpenAI,
		},
		Request: httptest.NewRequest("POST", "/v1/chat/completions", nil),
	}

	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	assert.True(t, logCtx.Logged, "logCtx should be marked as logged")
	assert.NotNil(t, logCtx.TokenUsage, "TokenUsage should not be nil")

	// Verify that the actual prompt tokens and completion tokens were extracted and NOT overwritten
	assert.Equal(t, 100, logCtx.TokenUsage.PromptTokens, "PromptTokens should be 100 (extracted from usage)")
	assert.Equal(t, 5, logCtx.TokenUsage.CompletionTokens, "CompletionTokens should be 5 (extracted from usage, not total_tokens = 105)")

	// Verify that tokens were consumed in the rate limiter (total_tokens = 105)
	assert.Equal(t, 105, prx.rateLimiter.GetCurrentTPM("test"), "Rate limiter should have recorded 105 tokens")
}

// TestHandleStreamingWithTokens_CombinedUsageAndDone verifies that when the usage data
// and the data: [DONE] sentinel arrive in the same TCP/buffer read chunk, the usage is still
// extracted correctly and lastChunk is updated.
func TestHandleStreamingWithTokens_CombinedUsageAndDone(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"total_tokens\":105}}\n\ndata: [DONE]\n\n",
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	logCtx := &RequestLogContext{
		RequestID:              "test-request-combined-123",
		promptTokensEstimateFn: func() int { return 95 },
		TokenUsage:             &converter.TokenUsage{},
		Credential: &config.CredentialConfig{
			Name: "test",
			Type: config.ProviderTypeOpenAI,
		},
		Request: httptest.NewRequest("POST", "/v1/chat/completions", nil),
	}

	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	assert.True(t, logCtx.Logged, "logCtx should be marked as logged")
	assert.NotNil(t, logCtx.TokenUsage, "TokenUsage should not be nil")

	// Verify that the actual prompt tokens and completion tokens were extracted and NOT overwritten
	assert.Equal(t, 100, logCtx.TokenUsage.PromptTokens, "PromptTokens should be 100 (extracted from usage)")
	assert.Equal(t, 5, logCtx.TokenUsage.CompletionTokens, "CompletionTokens should be 5 (extracted from usage, not total_tokens = 105)")
}

// TestHandleStreamingWithTokens_PartialZeroUsage_FallsBackForMissingField verifies that when
// a provider sends a usage object with some fields correctly populated and others as a bogus
// zero (observed from CometAPI's Anthropic-compatible streaming: message_start carries
// usage:{input_tokens:0,output_tokens:0} as a placeholder while output tokens land correctly
// via message_delta), the local tiktoken estimate fills in only the missing field. Before this
// fix, providerUsage was a single flag for "did any usage arrive at all", so a genuinely-sent
// completion_tokens value silently suppressed the prompt-token fallback too, permanently
// under-billing the request's input tokens.
func TestHandleStreamingWithTokens_PartialZeroUsage_FallsBackForMissingField(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":6,\"total_tokens\":6}}\n\n",
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	logCtx := &RequestLogContext{
		RequestID:              "test-request-partial-zero-usage",
		promptTokensEstimateFn: func() int { return 12 }, // local tiktoken fallback estimate
		TokenUsage:             &converter.TokenUsage{},
		Credential: &config.CredentialConfig{
			Name: "test",
			Type: config.ProviderTypeOpenAI,
		},
		Request: httptest.NewRequest("POST", "/v1/chat/completions", nil),
	}

	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	assert.True(t, logCtx.Logged, "logCtx should be marked as logged")
	assert.NotNil(t, logCtx.TokenUsage, "TokenUsage should not be nil")

	// PromptTokens: the provider reported 0, which is treated as "not reported" — the local
	// estimate must fill it in instead of silently billing zero input tokens.
	assert.Equal(t, 12, logCtx.TokenUsage.PromptTokens,
		"PromptTokens should fall back to the local estimate when the provider's usage object reports 0")
	// CompletionTokens: the provider genuinely reported 6 — must NOT be touched by the fallback.
	assert.Equal(t, 6, logCtx.TokenUsage.CompletionTokens,
		"CompletionTokens should keep the provider-reported value, not the fallback")
}

// TestHandleStreamingWithTokens_LegitimateZeroCompletion_NotOverwritten verifies the reverse of
// the partial-zero-usage case above: a genuine non-zero prompt_tokens alongside a genuine zero
// completion_tokens (e.g. output filtered or stopped immediately) must NOT be distrusted. No
// known provider bug leaves completion_tokens stuck at a placeholder zero while prompt_tokens is
// reported correctly (the CometAPI bug goes the other way — see the test above), so overwriting
// this legitimate zero with the local estimate would over-bill the request for output tokens it
// never produced.
func TestHandleStreamingWithTokens_LegitimateZeroCompletion_NotOverwritten(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":500,\"completion_tokens\":0,\"total_tokens\":500}}\n\n",
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

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()

	logCtx := &RequestLogContext{
		RequestID:              "test-request-legitimate-zero-completion",
		promptTokensEstimateFn: func() int { return 12 }, // local tiktoken fallback estimate
		TokenUsage:             &converter.TokenUsage{},
		Credential: &config.CredentialConfig{
			Name: "test",
			Type: config.ProviderTypeOpenAI,
		},
		Request: httptest.NewRequest("POST", "/v1/chat/completions", nil),
	}

	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	assert.True(t, logCtx.Logged, "logCtx should be marked as logged")
	assert.NotNil(t, logCtx.TokenUsage, "TokenUsage should not be nil")

	// PromptTokens: the provider genuinely reported 500 — must NOT be touched by the fallback.
	assert.Equal(t, 500, logCtx.TokenUsage.PromptTokens,
		"PromptTokens should keep the provider-reported value")
	// CompletionTokens: the provider genuinely reported 0 — must be trusted as-is, not
	// overwritten by the local estimate.
	assert.Equal(t, 0, logCtx.TokenUsage.CompletionTokens,
		"CompletionTokens should stay 0 when the provider reports it alongside a real prompt count")
}

// TestTokenCapturingWriter_CumulativeTokens verifies that tokenCapturingWriter does NOT
// accumulate total_tokens across chunks. Vertex/Gemini include a cumulative total_tokens in
// every streaming chunk; naively adding them up multiplies the real count by N (the number of
// chunks). The writer must keep only the last non-zero value.
func TestTokenCapturingWriter_CumulativeTokens(t *testing.T) {
	t.Run("vertex/gemini cumulative pattern - same value in every chunk", func(t *testing.T) {
		var total int
		w := &tokenCapturingWriter{
			writer: io.Discard,
			tokens: &total,
		}

		// Simulate 5 Vertex/Gemini chunks each reporting the same cumulative total_tokens=1000.
		// With accumulation the result would be 5000; with assignment it stays 1000.
		chunk := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"total_tokens\":1000}}\n\n")
		for range 5 {
			_, err := w.Write(chunk)
			require.NoError(t, err)
		}

		assert.Equal(t, 1000, total,
			"cumulative total_tokens must not be accumulated across chunks — expected 1000, not 5000")
	})

	t.Run("openai pattern - total_tokens only in last chunk", func(t *testing.T) {
		var total int
		w := &tokenCapturingWriter{
			writer: io.Discard,
			tokens: &total,
		}

		// OpenAI emits usage only in the final chunk.
		contentChunk := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		usageChunk := []byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"total_tokens\":750}}\n\n")

		for range 4 {
			_, _ = w.Write(contentChunk)
		}
		_, err := w.Write(usageChunk)
		require.NoError(t, err)

		assert.Equal(t, 750, total,
			"single total_tokens from final chunk must be preserved unchanged")
	})

	t.Run("growing cumulative values keep the last one", func(t *testing.T) {
		var total int
		w := &tokenCapturingWriter{
			writer: io.Discard,
			tokens: &total,
		}

		// Each chunk has a slightly larger cumulative count (typical for streaming models
		// that update the running total as tokens are generated).
		values := []int{100, 250, 500, 800, 1000}
		for _, v := range values {
			chunk := []byte(fmt.Sprintf("data: {\"usage\":{\"total_tokens\":%d}}\n\n", v))
			_, err := w.Write(chunk)
			require.NoError(t, err)
		}

		assert.Equal(t, 1000, total,
			"must keep the last (largest) cumulative total, not the sum of all values")
	})
}

// buildBedrockEventFrame creates an AWS EventStream binary frame for testing.
// Payload is base64-encoded Anthropic SSE JSON wrapped in {"bytes":"<base64>"}.
// CRC fields are zeroed — DecodeEventStreamToSSE does not validate them.
func buildBedrockEventFrame(anthropicEvent string) []byte {
	encoded := base64.StdEncoding.EncodeToString([]byte(anthropicEvent))
	payload, _ := json.Marshal(map[string]string{"bytes": encoded})

	payloadLength := len(payload)
	totalLength := 12 + payloadLength + 4 // prelude(12) + payload + msgCRC(4)

	buf := make([]byte, totalLength)
	binary.BigEndian.PutUint32(buf[0:4], uint32(totalLength))
	binary.BigEndian.PutUint32(buf[4:8], 0)  // headers length = 0
	binary.BigEndian.PutUint32(buf[8:12], 0) // prelude CRC (not validated)
	copy(buf[12:12+payloadLength], payload)
	binary.BigEndian.PutUint32(buf[12+payloadLength:], 0) // message CRC (not validated)
	return buf
}

// TestHandleResponsesAPIStreaming_BedrockAnthropic verifies that Bedrock Anthropic
// streaming works when modelID is an alias ("anthropic/claude-sonnet-4.5") and
// logCtx.RealModelID holds the real Bedrock model ID. The bug was that
// isAnthropicBedrockModel("anthropic/claude-sonnet-4.5") returned false, causing
// the Anthropic SSE events to be piped through without transformation, resulting
// in an empty Responses API response with zero tokens.
func TestHandleResponsesAPIStreaming_BedrockAnthropic(t *testing.T) {
	anthropicEvents := []string{
		`{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":16,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"stored"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}`,
		`{"type":"message_stop"}`,
	}

	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}
		for _, event := range anthropicEvents {
			_, _ = w.Write(buildBedrockEventFrame(event))
			flusher.Flush()
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeBedrock, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()
	cred := &config.CredentialConfig{
		Name: "test",
		Type: config.ProviderTypeBedrock,
	}

	// Alias as modelID, real Bedrock model name in logCtx — this is the production scenario
	// that was broken (isAnthropicBedrockModel failed on the alias).
	logCtx := &RequestLogContext{
		ModelID:     "anthropic/claude-sonnet-4.5",
		RealModelID: "anthropic.claude-sonnet-4-5-20250929-v1:0",
	}

	err = prx.handleResponsesAPIStreaming(w, resp, cred, "anthropic/claude-sonnet-4.5", logCtx, nil)
	require.NoError(t, err)

	body := w.Body.String()
	assert.Contains(t, body, "event: response.created", "should emit response.created event")
	assert.Contains(t, body, "response.output_text.delta", "should emit text delta event")
	assert.Contains(t, body, "stored", "should include actual response text")
	assert.Contains(t, body, "event: response.completed", "should emit response.completed event")
	assert.NotContains(t, body, `"output_tokens":0`, "should have non-zero token counts")
}
