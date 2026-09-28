package models

import (
	"log/slog"
	"os"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestSupportsVision_StaticWinsAndFalseWins(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	yes, no := true, false
	manager := New(logger, 100, []config.ModelRPMConfig{
		// One deployment of the name cannot see images, so the name cannot either.
		{Name: "glm", Credential: "a", SupportsVision: &yes},
		{Name: "glm", Credential: "b", SupportsVision: &no},
		{Name: "qwen", Credential: "a", SupportsVision: &yes},
		// false first, then true: false still wins.
		{Name: "gpt-oss", Credential: "a", SupportsVision: &no},
		{Name: "gpt-oss", Credential: "b", SupportsVision: &yes},
	})

	supported, known := manager.SupportsVision("gpt-oss")
	assert.True(t, known)
	assert.False(t, supported)

	supported, known = manager.SupportsVision("glm")
	assert.True(t, known)
	assert.False(t, supported)
	_, known = manager.SupportsVision("unknown")
	assert.False(t, known, "no declaration means unknown, not false")

	manager.UpdateDBModels([]config.ModelRPMConfig{
		{Name: "qwen", Credential: "a", SupportsVision: &no}, // static config wins
		{Name: "db-only", Credential: "a", SupportsVision: &no},
	}, nil, nil)

	supported, known = manager.SupportsVision("qwen")
	assert.True(t, known)
	assert.True(t, supported, "config.yaml overrides model_info.supports_vision")
	supported, known = manager.SupportsVision("db-only")
	assert.True(t, known)
	assert.False(t, supported)

	manager.UpdateDBModels(nil, nil, nil)
	_, known = manager.SupportsVision("db-only")
	assert.False(t, known, "a removed DB model drops its flag")
	_, known = manager.SupportsVision("glm")
	assert.True(t, known, "static flags survive DB syncs")
}
