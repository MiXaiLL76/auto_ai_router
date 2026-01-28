"""
OpenAI image generation tests for auto_ai_router
"""

import pytest
import base64
import tempfile
from pathlib import Path


class TestOpenAIImageGeneration:
    """OpenAI DALL-E image generation tests"""

    @pytest.mark.parametrize("model,size,quality", [
        ("gpt-image-1", "1024x1024", "medium"),
        ("gpt-image-1", "1024x1536", "high"),
        ("gpt-image-1-mini", "1024x1024", "low"),
        ("gpt-image-1-mini", "1536x1024", "auto"),
    ])
    def test_basic_generation(self, openai_client, model, size, quality, timeout_long):
        """Test basic image generation with different parameters"""
        response = openai_client.images.generate(
            model=model,
            prompt="A cute robot painting on a canvas in an art studio",
            size=size,
            quality=quality,
            n=1
        )

        assert len(response.data) == 1
        image_data = response.data[0]
        assert image_data.url or image_data.b64_json

    @pytest.mark.parametrize("n_images", [1, 2])
    def test_multiple_images(self, openai_client, n_images, timeout_long):
        """Test generating multiple images"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="A simple geometric pattern",
            size="1024x1024",
            n=n_images
        )

        assert len(response.data) == n_images
        for image_data in response.data:
            assert image_data.url or image_data.b64_json

    def test_quality_parameter(self, openai_client, timeout_long):
        """Test different quality parameters"""
        for quality in ["low", "medium", "high", "auto"]:
            response = openai_client.images.generate(
                model="gpt-image-1",
                prompt="A futuristic cityscape at sunset",
                size="1024x1024",
                quality=quality,
                n=1
            )

            assert len(response.data) == 1
            assert response.data[0].url or response.data[0].b64_json

    def test_basic_response(self, openai_client, timeout_long):
        """Test basic response format"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="A simple test image",
            size="1024x1024",
            n=1
        )

        assert len(response.data) == 1
        image_data = response.data[0]
        # Should have either URL or b64_json (depends on provider default)
        assert image_data.url or image_data.b64_json

    def test_save_generated_image(self, openai_client, timeout_long):
        """Test saving generated image to file"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="A test image for saving",
            size="1024x1024",
            n=1
        )

        assert len(response.data) == 1
        image_data = response.data[0]

        # Handle both URL and b64_json responses
        if image_data.b64_json:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image_bytes = base64.b64decode(image_data.b64_json)
                tmp_file.write(image_bytes)
                tmp_path = Path(tmp_file.name)

            # Verify file was created and has content
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 1000

            # Cleanup
            tmp_path.unlink()
        elif image_data.url:
            # Just verify URL is valid
            assert image_data.url.startswith("http")

    @pytest.mark.parametrize("prompt", [
        "A red apple on a white background",
        "Abstract art with blue and gold colors",
        "A minimalist logo design"
    ])
    def test_different_prompts(self, openai_client, prompt, timeout_long):
        """Test image generation with different prompts"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        assert len(response.data) == 1
        assert response.data[0].url or response.data[0].b64_json
