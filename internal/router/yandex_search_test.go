package router

import (
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/require"
)

func TestYandexSearchPathUsesDedicatedHandler(t *testing.T) {
	router := New(nil, nil, &config.MonitoringConfig{}, slog.Default(), &config.Config{})
	router.SetYandexSearchHandler(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		require.Equal(t, "/v1/search/yandex", request.URL.Path)
		w.WriteHeader(http.StatusOK)
	}))

	request := httptest.NewRequest(http.MethodPost, "/v1/search/yandex", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusOK, response.Code)
}

func TestYandexSearchPathRemainsClosedWhenDisabled(t *testing.T) {
	router := New(nil, nil, &config.MonitoringConfig{}, slog.Default(), &config.Config{})
	request := httptest.NewRequest(http.MethodPost, "/v1/search/yandex", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusNotFound, response.Code)
}

func TestYandexSearchPathRejectsNonPost(t *testing.T) {
	router := New(nil, nil, &config.MonitoringConfig{}, slog.Default(), &config.Config{})
	router.SetYandexSearchHandler(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		t.Fatal("handler must not be invoked for a disallowed method")
	}))

	request := httptest.NewRequest(http.MethodGet, "/v1/search/yandex", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusMethodNotAllowed, response.Code)
	require.Equal(t, http.MethodPost, response.Header().Get("Allow"))
}
