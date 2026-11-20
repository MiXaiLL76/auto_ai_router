package yandexsearch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClientSearchInjectsFolderIDAndStripsClientHeaders(t *testing.T) {
	var gotAuth, gotForwardedFor string
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotForwardedFor = r.Header.Get("X-Forwarded-For")
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		assert.Equal(t, "/v2/web/search", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"rawData":"c2VhcmNoIHJlc3VsdHM="}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		BaseURL:  server.URL,
		APIKey:   "server-held-key",
		FolderID: "b1gserverfolder",
	})

	clientBody := []byte(`{"query":{"searchType":"SEARCH_TYPE_RU","queryText":"test"},"folderId":"client-supplied-and-ignored"}`)
	status, respBody, err := client.Search(context.Background(), clientBody)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, status)
	assert.Equal(t, "Api-Key server-held-key", gotAuth)
	assert.Empty(t, gotForwardedFor, "client headers must never reach Yandex")
	assert.Equal(t, "b1gserverfolder", gotBody["folderId"], "server folderId must override any client-supplied value")
	assert.Equal(t, "test", gotBody["query"].(map[string]any)["queryText"], "non-folderId fields must pass through unchanged")

	var decoded map[string]string
	require.NoError(t, json.Unmarshal(respBody, &decoded))
	assert.Equal(t, "c2VhcmNoIHJlc3VsdHM=", decoded["rawData"])
}

func TestClientSearchAddsFolderIDWhenMissing(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "key", FolderID: "injected-folder"})
	_, _, err := client.Search(context.Background(), []byte(`{"query":{"queryText":"x"}}`))
	require.NoError(t, err)
	assert.Equal(t, "injected-folder", gotBody["folderId"])
}

func TestClientSearchRejectsNonObjectBody(t *testing.T) {
	client := NewClient(ClientConfig{BaseURL: "https://example.invalid", APIKey: "key", FolderID: "folder"})
	_, _, err := client.Search(context.Background(), []byte(`not json`))
	require.ErrorIs(t, err, ErrInvalidBody)

	_, _, err = client.Search(context.Background(), []byte(`[1,2,3]`))
	require.ErrorIs(t, err, ErrInvalidBody)
}

func TestClientSearchPassesThroughNon2xxStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"message":"rate limited"}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "key", FolderID: "folder"})
	status, body, err := client.Search(context.Background(), []byte(`{"query":{}}`))
	require.NoError(t, err, "a non-2xx upstream response is a valid passthrough result, not a client error")
	assert.Equal(t, http.StatusTooManyRequests, status)
	assert.Contains(t, string(body), "rate limited")
}
