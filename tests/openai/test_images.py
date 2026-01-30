"""
OpenAI image generation tests
Tests image generation API with different parameters
"""

import pytest


class TestOpenAIImageGeneration:
    """Test image generation functionality"""

    @pytest.mark.parametrize("size", ["1024x1024"])
    def test_basic_image_generation(self, openai_client, size):
        """Test basic image generation"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="A serene landscape with mountains",
            size=size,
            n=1
        )

        assert response is not None
        assert hasattr(response, 'data')
        assert len(response.data) >= 1
        assert hasattr(response.data[0], 'url') or hasattr(response.data[0], 'b64_json')

    def test_single_image(self, openai_client):
        """Test single image generation"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="A robot",
            size="1024x1024",
            n=1
        )

        assert len(response.data) == 1
        image = response.data[0]
        assert image.url or image.b64_json

    def test_multiple_images(self, openai_client):
        """Test multiple image generation"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="A sunset",
            size="1024x1024",
            n=2
        )

        assert len(response.data) >= 2
        for image in response.data:
            assert image.url or image.b64_json

    @pytest.mark.parametrize("prompt", [
        "A red apple",
        "Blue ocean waves",
        "Futuristic city"
    ])
    def test_different_prompts(self, openai_client, prompt):
        """Test generation with different prompts"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        assert len(response.data) >= 1
        assert response.data[0].url or response.data[0].b64_json

    def test_response_structure(self, openai_client):
        """Test response structure"""
        response = openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt="Test",
            size="1024x1024",
            n=1
        )

        assert hasattr(response, 'data')
        assert hasattr(response, 'created')
        image_data = response.data[0]
        assert image_data.url or image_data.b64_json
