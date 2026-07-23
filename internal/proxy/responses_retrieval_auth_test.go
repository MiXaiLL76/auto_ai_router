package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type retrievalAuthResponseStore struct {
	ownerHash  string
	getCalls   int
	masterGets int
}

func (s *retrievalAuthResponseStore) SaveResponse(context.Context, string, *responses.Response, map[string]string, int, json.RawMessage, string) error {
	return nil
}

func (s *retrievalAuthResponseStore) GetResponse(_ context.Context, _ string, apiKeyHash string) (*responses.Response, error) {
	s.getCalls++
	if s.ownerHash != "" && apiKeyHash != s.ownerHash {
		return nil, fmt.Errorf("unauthorized")
	}
	return &responses.Response{ID: "resp-owned", Object: "response"}, nil
}

func (s *retrievalAuthResponseStore) GetEntry(context.Context, string, string) (*responsestore.StoredEntry, error) {
	return nil, fmt.Errorf("not found")
}

func (s *retrievalAuthResponseStore) GetResponseByID(context.Context, string) (*responses.Response, error) {
	s.masterGets++
	return &responses.Response{ID: "resp-owned", Object: "response"}, nil
}

func (s *retrievalAuthResponseStore) CleanupExpired(context.Context) error { return nil }
func (s *retrievalAuthResponseStore) Close() error                         { return nil }

func TestHandleGetResponseAcceptsXAPIKeyThroughSharedAuth(t *testing.T) {
	db := &clientAuthTestDB{
		tokens: map[string]*dbmodels.TokenInfo{
			"llm-key": {
				Token: "llm-key-hash",
			},
		},
	}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	store := &retrievalAuthResponseStore{}
	prx.responseStore = store

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/resp-owned", nil)
	req.Header.Set("x-api-key", "llm-key")
	recorder := httptest.NewRecorder()

	prx.HandleGetResponse(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, 1, store.getCalls)
	assert.Contains(t, recorder.Body.String(), "resp-owned")
}

func TestHandleGetResponsePreservesTenantOwnershipAndMasterBypass(t *testing.T) {
	db := &clientAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"owner-key": {
			Token: "owner-key-hash",
		},
		"other-key": {
			Token: "other-key-hash",
		},
	}}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	store := &retrievalAuthResponseStore{ownerHash: "owner-key-hash"}
	prx.responseStore = store

	tests := []struct {
		name       string
		token      string
		wantStatus int
	}{
		{name: "owner", token: "owner-key", wantStatus: http.StatusOK},
		{name: "different tenant is masked", token: "other-key", wantStatus: http.StatusNotFound},
		{name: "master bypasses ownership", token: "master-key", wantStatus: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			beforeGets := store.getCalls
			beforeMaster := store.masterGets
			req := httptest.NewRequest(http.MethodGet, "/v1/responses/resp-owned", nil)
			req.Header.Set("Authorization", "Bearer "+tt.token)
			recorder := httptest.NewRecorder()

			prx.HandleGetResponse(recorder, req)

			require.Equal(t, tt.wantStatus, recorder.Code, recorder.Body.String())
			switch tt.token {
			case "master-key":
				assert.Equal(t, beforeGets, store.getCalls)
				assert.Equal(t, beforeMaster+1, store.masterGets)
			default:
				assert.Equal(t, beforeGets+1, store.getCalls)
				assert.Equal(t, beforeMaster, store.masterGets)
			}
		})
	}
}
