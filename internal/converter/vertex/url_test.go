package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

func TestBuildGeminiURL(t *testing.T) {
	cred := &config.CredentialConfig{
		BaseURL: "https://generativelanguage.googleapis.com",
	}

	t.Run("non_streaming", func(t *testing.T) {
		url := BuildGeminiURL(cred, "gemini-2.5-flash", false)
		assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", url)
	})

	t.Run("streaming", func(t *testing.T) {
		url := BuildGeminiURL(cred, "gemini-2.5-flash", true)
		assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", url)
	})

	t.Run("trailing_slash", func(t *testing.T) {
		credSlash := &config.CredentialConfig{
			BaseURL: "https://generativelanguage.googleapis.com/",
		}
		url := BuildGeminiURL(credSlash, "gemini-3-pro", false)
		assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro:generateContent", url)
	})
}

func TestBuildVertexURL(t *testing.T) {
	t.Run("regional_gemini", func(t *testing.T) {
		cred := &config.CredentialConfig{
			ProjectID: "my-project",
			Location:  "us-central1",
		}
		url := BuildVertexURL(cred, "gemini-2.5-flash", false)
		assert.Equal(t, "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent", url)
	})

	t.Run("regional_streaming", func(t *testing.T) {
		cred := &config.CredentialConfig{
			ProjectID: "my-project",
			Location:  "europe-west1",
		}
		url := BuildVertexURL(cred, "gemini-2.5-pro", true)
		assert.Contains(t, url, "streamGenerateContent?alt=sse")
		assert.Contains(t, url, "europe-west1-aiplatform.googleapis.com")
	})

	t.Run("global_location", func(t *testing.T) {
		cred := &config.CredentialConfig{
			ProjectID: "my-project",
			Location:  "global",
		}
		url := BuildVertexURL(cred, "gemini-2.5-flash", false)
		assert.Equal(t, "https://aiplatform.googleapis.com/v1beta1/projects/my-project/locations/global/publishers/google/models/gemini-2.5-flash:generateContent", url)
	})

	t.Run("claude_model_anthropic_publisher", func(t *testing.T) {
		cred := &config.CredentialConfig{
			ProjectID: "my-project",
			Location:  "us-east5",
		}
		url := BuildVertexURL(cred, "claude-sonnet-4-20250514", false)
		assert.Contains(t, url, "/publishers/anthropic/")
	})
}
