package spendlog

import (
	"context"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/require"
)

func TestShutdownCancelsAdmittedSynchronousAccountingOperation(t *testing.T) {
	cfg := models.DefaultConfig()
	cfg.Logger = testhelpers.NewTestLogger()
	logger := NewLogger(nil, cfg)

	require.True(t, logger.beginOperation())
	opCtx, cancelOperation := logger.synchronousOperationContext(context.Background())
	defer cancelOperation()
	released := make(chan struct{})
	go func() {
		<-opCtx.Done()
		logger.enqueueWG.Done()
		close(released)
	}()

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), time.Second)
	defer cancelShutdown()
	require.NoError(t, logger.Shutdown(shutdownCtx))
	require.ErrorIs(t, opCtx.Err(), context.Canceled)
	select {
	case <-released:
	case <-time.After(time.Second):
		t.Fatal("synchronous accounting operation did not release its shutdown lifecycle ticket")
	}
}
