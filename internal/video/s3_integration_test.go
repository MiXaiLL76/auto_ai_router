package video

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestS3StoreRoundTrip(t *testing.T) {
	accessKey := os.Getenv("AIR_VIDEO_TEST_S3_ACCESS_KEY")
	secretKey := os.Getenv("AIR_VIDEO_TEST_S3_SECRET_KEY")
	if accessKey == "" || secretKey == "" {
		t.Skip("AIR_VIDEO_TEST_S3 credentials are not set")
	}
	store, err := NewS3Store(S3Config{
		Endpoint:  "https://s3.twcstorage.ru",
		Region:    "ru-1",
		Bucket:    "a73def9f-143e-4c0e-a7c8-eb36cae1a4be",
		Prefix:    "air-video-test",
		AccessKey: accessKey,
		SecretKey: secretKey,
	})
	require.NoError(t, err)
	payload := append([]byte{137, 80, 78, 71, 13, 10, 26, 10}, []byte("air-video-s3-test")...)
	sum := sha256.Sum256(payload)
	upload := &Upload{
		ID: "upl_" + uuid.NewString(), OrganizationID: "integration-test",
		Purpose: "video_input_image", MIME: "image/png", SizeBytes: int64(len(payload)),
		SHA256: hex.EncodeToString(sum[:]), ObjectKey: "integration-test/" + uuid.NewString(),
		State: "created", ExpiresAt: time.Now().Add(time.Minute),
	}
	t.Cleanup(func() { _ = store.DeleteUpload(t.Context(), upload) })
	require.NoError(t, store.PutUpload(t.Context(), upload, "image/png", bytes.NewReader(payload)))
	require.NoError(t, store.VerifyUpload(t.Context(), upload))
	object, err := store.OpenUpload(t.Context(), upload)
	require.NoError(t, err)
	actual, err := io.ReadAll(object.Body)
	require.NoError(t, err)
	require.NoError(t, object.Body.Close())
	require.Equal(t, payload, actual)
	require.NoError(t, store.DeleteUpload(t.Context(), upload))
}
