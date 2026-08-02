package router

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMessagesErrorWriter_DecodesGzipBeforeParsing verifies that when the
// upstream error body arrives gzip-compressed (because the real client sent
// Accept-Encoding: gzip and it's honored end to end), the buffered error
// writer still decompresses before parsing the JSON error message, instead
// of returning garbled compressed bytes as the message.
func TestMessagesErrorWriter_DecodesGzipBeforeParsing(t *testing.T) {
	rec := httptest.NewRecorder()
	w := &messagesErrorWriter{ResponseWriter: rec}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	_, err := gz.Write([]byte(`{"error":{"message":"upstream exploded"}}`))
	require.NoError(t, err)
	require.NoError(t, gz.Close())

	w.Header().Set("Content-Encoding", "gzip")
	w.WriteHeader(500)
	_, err = w.Write(buf.Bytes())
	require.NoError(t, err)
	w.finalize()

	var resp struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "upstream exploded", resp.Error.Message)
	assert.Empty(t, rec.Header().Get("Content-Encoding"), "compressed error body must not be forwarded as-is")
}
