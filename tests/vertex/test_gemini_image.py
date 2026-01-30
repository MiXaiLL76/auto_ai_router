"""
Vertex AI Image Generation tests
Tests Imagen models for image generation
"""

import pytest
from test_helpers import TestModels, ResponseValidator


class TestVertexImageGeneration:
    """Test Vertex Imagen image generation"""

    @pytest.mark.parametrize("model", TestModels.VERTEX_IMAGE_MODELS)
    def test_basic_image_generation(self, openai_client, model):
        """Test basic image generation with Imagen"""
        try:
            response = openai_client.images.generate(
                model=model,
                prompt="A beautiful landscape with mountains",
                n=1,
                size="512x512"
            )

            assert response is not None
            assert hasattr(response, 'data')
            assert len(response.data) >= 1
            # May have URL or b64_json
            image = response.data[0]
            assert image.url or image.b64_json
        except Exception as e:
            # May be rate limited (429) - that's OK for now
            if "429" in str(e) or "rate" in str(e).lower():
                pytest.skip(f"Image API rate limited: {e}")
            raise

    @pytest.mark.parametrize("model", TestModels.VERTEX_IMAGE_MODELS)
    def test_simple_prompt(self, openai_client, model):
        """Test image generation with simple prompt"""
        try:
            response = openai_client.images.generate(
                model=model,
                prompt="Cat",
                n=1,
                size="512x512"
            )

            assert len(response.data) >= 1
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                pytest.skip(f"Image API rate limited: {e}")
            raise

    @pytest.mark.parametrize("model", TestModels.VERTEX_IMAGE_MODELS)
    def test_detailed_prompt(self, openai_client, model):
        """Test with detailed prompt"""
        try:
            response = openai_client.images.generate(
                model=model,
                prompt="Futuristic city with neon lights, flying cars, tall buildings, night time",
                n=1,
                size="512x512"
            )

            assert len(response.data) >= 1
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                pytest.skip(f"Image API rate limited: {e}")
            raise

    @pytest.mark.parametrize("model", TestModels.VERTEX_IMAGE_MODELS)
    def test_style_variations(self, openai_client, model):
        """Test different style prompts"""
        prompts = [
            "Oil painting of forest",
            "Digital art of space",
        ]

        for prompt in prompts:
            try:
                response = openai_client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size="512x512"
                )

                assert len(response.data) >= 1
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    pytest.skip(f"Image API rate limited: {e}")
                raise
