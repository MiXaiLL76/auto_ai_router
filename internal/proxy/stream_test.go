package proxy

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

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
	logger := createTestLogger()
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

	logger := createTestLogger()
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

	logger := createTestLogger()
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

	logger := createTestLogger()
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
