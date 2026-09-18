package router

import (
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/require"
)

func TestVideoPathsUseDedicatedHandler(t *testing.T) {
	router := New(nil, nil, &config.MonitoringConfig{}, slog.Default(), &config.Config{})
	router.SetVideoHandler(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		require.Equal(t, "/v1/videos", request.URL.Path)
		w.WriteHeader(http.StatusAccepted)
	}))

	request := httptest.NewRequest(http.MethodPost, "/v1/videos", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusAccepted, response.Code)
}

func TestVideoPathsRemainClosedWhenDisabled(t *testing.T) {
	router := New(nil, nil, &config.MonitoringConfig{}, slog.Default(), &config.Config{})
	request := httptest.NewRequest(http.MethodPost, "/v1/videos", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusNotFound, response.Code)
}
