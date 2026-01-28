"""
Gemini image generation and editing tests for auto_ai_router
"""

import pytest
import requests
import json
import base64
import mimetypes
import tempfile
from pathlib import Path


class TestGeminiImageGeneration:
    """Gemini image generation via chat completions"""

    @pytest.mark.parametrize("model", [
        "gemini-2.5-flash-image",
        "gemini-2.5-pro-image"
    ])
    def test_image_generation_fixed(self, master_key, base_url, model, timeout_long):
        """Test Gemini image generation with updated vertex.go"""
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Сгенерируй фотореалистичный цветок оранжевой лилии на зеленом фоне"}
            ],
            "max_tokens": 1,
            "extra_body": {
                "modalities": ["image"],
                "generation_config": {
                    "temperature": 0.4
                }
            }
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()

        # Extract images from response
        images = result.get("choices", [{}])[0].get("message", {}).get("images", [])
        assert len(images) > 0

        for img in images:
            assert "b64_json" in img
            # Verify valid base64 image data
            image_bytes = base64.b64decode(img["b64_json"])
            assert len(image_bytes) > 1000

    @pytest.mark.parametrize("prompt,temperature", [
        ("Generate a realistic red rose", 0.3),
        ("Create a futuristic robot", 0.7),
        ("Paint a mountain landscape", 0.5)
    ])
    def test_different_generation_prompts(self, master_key, base_url, prompt, temperature, timeout_long):
        """Test image generation with different prompts and temperatures"""
        payload = {
            "model": "gemini-2.5-flash-image",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1,
            "extra_body": {
                "modalities": ["image"],
                "generation_config": {
                    "temperature": temperature
                }
            }
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()
        images = result.get("choices", [{}])[0].get("message", {}).get("images", [])
        assert len(images) > 0


class TestGeminiImageEditing:
    """Gemini image editing via chat completions"""

    @staticmethod
    def _create_test_image():
        """Create a simple test image for editing tests"""
        # Create a simple colored square as test image
        from PIL import Image
        import io

        # Create a simple 256x256 red square
        img = Image.new('RGB', (256, 256), color='red')

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes.getvalue()

    @staticmethod
    def _to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Convert image bytes to data URL"""
        b64_data = base64.b64encode(image_bytes).decode()
        return f"data:{mime_type};base64,{b64_data}"

    def test_image_editing_with_test_image(self, master_key, base_url, timeout_long):
        """Test image editing with a programmatically created test image"""
        # Create test image
        test_image_bytes = self._create_test_image()
        data_url = self._to_data_url(test_image_bytes)

        payload = {
            "model": "gemini-2.5-flash-image",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Add a yellow circle in the center of this image"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            "max_tokens": 1,
            "extra_body": {
                "modalities": ["image"],
                "generation_config": {
                    "temperature": 0.4
                }
            }
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()
        images = result.get("choices", [{}])[0].get("message", {}).get("images", [])
        assert len(images) > 0

        for img in images:
            assert "b64_json" in img
            image_bytes = base64.b64decode(img["b64_json"])
            assert len(image_bytes) > 1000

    @pytest.mark.parametrize("edit_instruction", [
        "Make the image brighter",
        "Add a blue border around the image",
        "Change the background to green"
    ])
    def test_different_editing_instructions(self, master_key, base_url, edit_instruction, timeout_long):
        """Test image editing with different instructions"""
        # Create test image
        test_image_bytes = self._create_test_image()
        data_url = self._to_data_url(test_image_bytes)

        payload = {
            "model": "gemini-2.5-flash-image",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": edit_instruction},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            "max_tokens": 1,
            "extra_body": {
                "modalities": ["image"],
                "generation_config": {
                    "temperature": 0.4
                }
            }
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()
        images = result.get("choices", [{}])[0].get("message", {}).get("images", [])
        assert len(images) > 0

    def test_save_edited_image(self, master_key, base_url, timeout_long):
        """Test saving edited image to file"""
        # Create test image
        test_image_bytes = self._create_test_image()
        data_url = self._to_data_url(test_image_bytes)

        payload = {
            "model": "gemini-2.5-flash-image",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Add a white circle in the center"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            "max_tokens": 1,
            "extra_body": {
                "modalities": ["image"],
                "generation_config": {
                    "temperature": 0.4
                }
            }
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=timeout_long
        )

        assert response.status_code == 200
        result = response.json()
        images = result.get("choices", [{}])[0].get("message", {}).get("images", [])
        assert len(images) > 0

        # Save first image to temporary file
        img = images[0]
        image_bytes = base64.b64decode(img["b64_json"])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = Path(tmp_file.name)

        # Verify file was created and has content
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 1000

        # Cleanup
        tmp_path.unlink()
