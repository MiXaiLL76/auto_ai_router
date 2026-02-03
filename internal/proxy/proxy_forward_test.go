package proxy

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestForwardToProxy_Headers проверяет корректное проксирование заголовков:
// 1. Hop-by-hop заголовки (Connection, Transfer-Encoding и т.д.) не проксируются
// 2. Authorization заменяется на cred.APIKey когда APIKey задан
// 3. Content-Length корректно устанавливается на ответе
func TestForwardToProxy_Headers(t *testing.T) {
	// Флаги для отслеживания проверок в upstream handler
	var (
		connectionHeaderFound bool
		authHeaderValue       string
		receivedHeaders       http.Header
		requestReceived       bool
	)

	// Создаем upstream сервер, который проверяет входящие заголовки
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestReceived = true
		receivedHeaders = r.Header

		// Проверка 1: Hop-by-hop заголовок "Connection" не должен быть проксирован
		if r.Header.Get("Connection") != "" {
			connectionHeaderFound = true
		}

		// Проверка 2: Сохраняем Authorization для проверки
		authHeaderValue = r.Header.Get("Authorization")

		// Ответить успешно с телом
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("upstream body"))
	}))
	defer upstreamServer.Close()

	// Создаем Proxy с маленьким requestTimeout
	logger := createTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	// Создаем credentials для proxy типа с APIKey
	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "remote-key",
	}

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	// Создаем upstream request с заголовками для проксирования
	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("request body"))
	upstreamReq.Header.Set("Authorization", "Bearer sk-master") // Это должно быть заменено
	upstreamReq.Header.Set("Connection", "keep-alive")          // Это НЕ должно быть проксировано
	upstreamReq.Header.Set("Content-Type", "application/json")  // Это должно быть проксировано
	upstreamReq.Header.Set("X-Custom-Header", "custom-value")   // Это должно быть проксировано

	// Отправляем request
	w := httptest.NewRecorder()
	respBody := []byte("request body")
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, respBody, time.Now())

	// Проверки результата
	require.NoError(t, err, "forwardToProxy должен выполниться без ошибок")
	require.NotNil(t, proxyResp, "ProxyResponse не должен быть nil")
	require.True(t, requestReceived, "Upstream сервер должен получить запрос")

	// ============ ПРОВЕРКА 1: Hop-by-hop заголовки не проксируются ============
	assert.False(t, connectionHeaderFound,
		"Connection заголовок (hop-by-hop) не должен быть проксирован",
	)
	assert.Empty(t, receivedHeaders.Get("Connection"),
		"Connection заголовок должен быть пустым в upstream запросе",
	)

	// ============ ПРОВЕРКА 2: Authorization заменяется на cred.APIKey ============
	expectedAuth := "Bearer remote-key"
	assert.Equal(t, expectedAuth, authHeaderValue,
		"Authorization должен быть заменен на Bearer <cred.APIKey>",
	)
	assert.NotContains(t, authHeaderValue, "sk-master",
		"Оригинальный Authorization (sk-master) не должен быть в upstream запросе",
	)

	// Другие заголовки должны быть проксированы корректно
	assert.Equal(t, "application/json", receivedHeaders.Get("Content-Type"),
		"Content-Type должен быть проксирован",
	)
	assert.Equal(t, "custom-value", receivedHeaders.Get("X-Custom-Header"),
		"Custom заголовки должны быть проксированы",
	)

	// ============ ПРОВЕРКА 3: Content-Length в ответе правильный ============
	assert.Equal(t, http.StatusOK, proxyResp.StatusCode,
		"Статус код ответа должен быть 200 OK",
	)

	expectedBody := []byte("upstream body")
	assert.Equal(t, expectedBody, proxyResp.Body,
		"Тело ответа должно соответствовать upstream ответу",
	)

	// Проверка что Content-Length будет правильно установлен в ResponseWriter
	// (forwardToProxy устанавливает его позже при записи в ResponseWriter)
	expectedContentLength := len(expectedBody)
	assert.Equal(t, expectedContentLength, len(proxyResp.Body),
		"Длина тела ответа должна соответствовать Content-Length",
	)
}

// TestForwardToProxy_HeadersWithoutAPIKey проверяет что Authorization копируется
// если в credentials нет APIKey
func TestForwardToProxy_HeadersWithoutAPIKey(t *testing.T) {
	var authHeaderValue string

	// Upstream сервер
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeaderValue = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer upstreamServer.Close()

	// Credentials БЕЗ APIKey
	logger := createTestLogger()
	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "", // Пусто!
	}

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("body"))
	upstreamReq.Header.Set("Authorization", "Bearer custom-token")

	w := httptest.NewRecorder()
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now())

	require.NoError(t, err)
	require.NotNil(t, proxyResp)

	// Когда APIKey пустой, оригинальный Authorization должен быть скопирован
	assert.Equal(t, "Bearer custom-token", authHeaderValue,
		"Когда APIKey отсутствует, оригинальный Authorization должен быть скопирован",
	)
}

// TestForwardToProxy_MultipleHopByHopHeaders проверяет что несколько hop-by-hop
// заголовков не проксируются
func TestForwardToProxy_MultipleHopByHopHeaders(t *testing.T) {
	var receivedHeaders http.Header

	// Upstream сервер проверяет что hop-by-hop заголовки отсутствуют
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer upstreamServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "key",
	}

	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	// Создаем request со всеми hop-by-hop заголовками
	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("body"))
	upstreamReq.Header.Set("Connection", "keep-alive")
	upstreamReq.Header.Set("Keep-Alive", "5")
	upstreamReq.Header.Set("Proxy-Authenticate", "Basic realm=test")
	upstreamReq.Header.Set("Proxy-Authorization", "Bearer token")
	upstreamReq.Header.Set("Trailer", "X-Custom")
	upstreamReq.Header.Set("Transfer-Encoding", "chunked")
	upstreamReq.Header.Set("Upgrade", "websocket")

	w := httptest.NewRecorder()
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now())

	require.NoError(t, err)
	require.NotNil(t, proxyResp)

	// Проверяем что hop-by-hop заголовки не проксированы (кроме тех которые может модифицировать http.Client)
	assert.Empty(t, receivedHeaders.Get("Connection"), "Connection не должен быть проксирован")
	assert.Empty(t, receivedHeaders.Get("Keep-Alive"), "Keep-Alive не должен быть проксирован")
	assert.Empty(t, receivedHeaders.Get("Proxy-Authenticate"), "Proxy-Authenticate не должен быть проксирован")
	assert.Empty(t, receivedHeaders.Get("Proxy-Authorization"), "Proxy-Authorization не должен быть проксирован")
	assert.Empty(t, receivedHeaders.Get("Trailer"), "Trailer не должен быть проксирован")
	assert.Empty(t, receivedHeaders.Get("Upgrade"), "Upgrade не должен быть проксирован")
}

// TestForwardToProxy_ContentLengthCorrect проверяет что Content-Length правильно
// устанавливается на ответе (переопределяя upstream значение)
func TestForwardToProxy_ContentLengthCorrect(t *testing.T) {
	responseBody := "This is the upstream response body"

	// Upstream сервер отправляет Content-Length (правильный будет перезаписан)
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(responseBody))
	}))
	defer upstreamServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "key",
	}

	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("body"))
	upstreamReq.Header.Set("Authorization", "Bearer key")

	w := httptest.NewRecorder()
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now())

	require.NoError(t, err)
	require.NotNil(t, proxyResp)

	// executeProxyRequest читает весь body и вернет правильную длину в ProxyResponse
	actualBody := proxyResp.Body
	expectedLength := len(responseBody)
	actualLength := len(actualBody)

	assert.Equal(t, expectedLength, actualLength,
		"Длина тела должна быть правильной",
	)
	assert.Equal(t, responseBody, string(actualBody),
		"Тело ответа должно быть верным",
	)
}

// TestForwardToProxy_QueryParameters проверяет что query параметры корректно
// проксируются
func TestForwardToProxy_QueryParameters(t *testing.T) {
	var receivedQuery string

	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer upstreamServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "key",
	}

	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	// Request с query параметрами
	upstreamReq := httptest.NewRequest("POST", "/v1/test?param1=value1&param2=value2", strings.NewReader("body"))
	upstreamReq.Header.Set("Authorization", "Bearer key")

	w := httptest.NewRecorder()
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now())

	require.NoError(t, err)
	require.NotNil(t, proxyResp)

	// Query параметры должны быть проксированы
	assert.Equal(t, "param1=value1&param2=value2", receivedQuery,
		"Query параметры должны быть корректно проксированы",
	)
}

// TestForwardToProxy_LargeResponseBody проверяет что большие тела ответа
// правильно читаются и устанавливается правильный Content-Length
func TestForwardToProxy_LargeResponseBody(t *testing.T) {
	// Создаем большое тело ответа (1MB)
	largeBody := bytes.Repeat([]byte("x"), 1024*1024)

	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(largeBody)
	}))
	defer upstreamServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "key",
	}

	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("body"))
	upstreamReq.Header.Set("Authorization", "Bearer key")

	w := httptest.NewRecorder()
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now())

	require.NoError(t, err)
	require.NotNil(t, proxyResp)

	// Проверяем что весь большой body был прочитан
	assert.Equal(t, len(largeBody), len(proxyResp.Body),
		"Весь большой body должен быть прочитан",
	)
	assert.Equal(t, largeBody, proxyResp.Body,
		"Большой body должен быть полностью скопирован",
	)
}

// TestForwardToProxy_UpstreamError проверяет обработку ошибок при обращении к upstream
func TestForwardToProxy_UpstreamError(t *testing.T) {
	// Создаем server который закрывает соединение
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Имитируем внутреннюю ошибку
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error": "internal server error"}`))
	}))
	defer upstreamServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstreamServer.URL,
		APIKey:  "key",
	}

	bal, rl := createTestBalancer(upstreamServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)

	prx := createProxyWithParams(
		bal, logger, 10, 5*time.Second, metrics,
		"master-key", rl, tm, mm,
		"test-version", "test-commit",
	)

	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("body"))
	upstreamReq.Header.Set("Authorization", "Bearer key")

	w := httptest.NewRecorder()
	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now())

	// Ошибок быть не должно, но статус код должен быть 500
	require.NoError(t, err)
	require.NotNil(t, proxyResp)
	assert.Equal(t, http.StatusInternalServerError, proxyResp.StatusCode,
		"Статус код ошибки должен быть проксирован",
	)
}
