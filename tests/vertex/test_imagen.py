"""
Vertex AI Imagen tests for auto_ai_router
"""

import pytest
import requests
import base64
import tempfile
from pathlib import Path


class TestImagenGeneration:
    """Imagen image generation tests"""

    @pytest.mark.parametrize("model,size,quality", [
        ("imagen-3.0-fast-generate-001", "1024x1024", "standard"),
        ("imagen-4.0-fast-generate-001", "1024x1024", "hd"),
        ("imagen-4.0-fast-generate-001", "512x512", "standard"),
    ])
    def test_basic_generation(self, openai_client, model, size, quality, timeout_long):
        """Test basic image generation with different parameters"""
        response = openai_client.images.generate(
            model=model,
            prompt="A cute robot painting a colorful landscape on a canvas, studio lighting, high quality digital art",
            size=size,
            quality=quality,
            n=1
        )

        assert len(response.data) == 1
        image_data = response.data[0]
        assert image_data.b64_json or image_data.url

        if image_data.b64_json:
            # Verify it's valid base64 image data
            image_bytes = base64.b64decode(image_data.b64_json)
            assert len(image_bytes) > 1000  # Should be substantial image data

    @pytest.mark.parametrize("n_images", [1, 2])
    def test_multiple_images(self, openai_client, n_images, timeout_long):
        """Test generating multiple images"""
        response = openai_client.images.generate(
            model="imagen-3.0-fast-generate-001",
            prompt="A serene Japanese garden with cherry blossoms and a small pond, digital art style",
            size="1024x1024",
            n=n_images
        )

        assert len(response.data) == n_images
        for image_data in response.data:
            assert image_data.b64_json or image_data.url

    @pytest.mark.parametrize("prompt", [
        "Фотореалистичный оранжевый цветок лилии на зеленом фоне",
        "A beautiful sunset over mountains, photorealistic, high quality",
        "Abstract geometric patterns in blue and gold colors"
    ])
    def test_different_prompts(self, openai_client, prompt, timeout_long):
        """Test image generation with different prompts"""
        response = openai_client.images.generate(
            model="imagen-3.0-fast-generate-001",
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        assert len(response.data) == 1
        assert response.data[0].b64_json or response.data[0].url


class TestImagenAdvanced:
    """Advanced Imagen functionality tests"""

    def test_advanced_parameters_requests(self, master_key, base_url, timeout_long):
        """Test advanced parameters via requests"""
        payload = {
            "model": "imagen-4.0-fast-generate-001",
            "prompt": "A beautiful sunset over mountains, photorealistic, high quality",
            "n": 2,
            "size": "1024x1024",
            "quality": "hd",
            "style": "natural",
            "response_format": "b64_json",
            "user": "test_user_images"
        }

        response = requests.post(
            f"{base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()
        assert "data" in result
        assert len(result["data"]) == 2

        for img_data in result["data"]:
            assert "b64_json" in img_data
            # Verify valid base64
            image_bytes = base64.b64decode(img_data["b64_json"])
            assert len(image_bytes) > 1000

    def test_save_generated_image(self, openai_client, timeout_long):
        """Test saving generated image to file"""
        response = openai_client.images.generate(
            model="imagen-3.0-fast-generate-001",
            prompt="A simple flower image for testing",
            size="512x512",
            n=1
        )

        assert len(response.data) == 1
        image_data = response.data[0]

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

    @pytest.mark.parametrize("response_format", ["b64_json", "url"])
    def test_response_formats(self, master_key, base_url, response_format, timeout_long):
        """Test different response formats"""
        payload = {
            "model": "imagen-3.0-fast-generate-001",
            "prompt": "Simple test image",
            "n": 1,
            "size": "512x512",
            "response_format": response_format
        }

        response = requests.post(
            f"{base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["data"]) == 1

        img_data = result["data"][0]
        if response_format == "b64_json":
            assert "b64_json" in img_data
            assert img_data["b64_json"]
        else:
            assert "url" in img_data
            assert img_data["url"]
