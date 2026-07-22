"""
Gemini Native Image Generation / Editing tests.
Tests gemini-*-image-* models for text-to-image and image-to-image editing.
"""

import base64
import io
import os
import struct
import zlib

import pytest

from test_helpers import TestModels

# Path to the real flower photo used in integration tests
_FLOWER_JPG = os.path.join(os.path.dirname(__file__), "flower.jpg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid_png(r: int, g: int, b: int, size: int = 64) -> bytes:
    """Generate solid-color PNG as bytes."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + bytes([r, g, b]) * size
    idat = zlib.compress(raw_row * size)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _png_bytes(r: int, g: int, b: int) -> io.BytesIO:
    """Return BytesIO PNG for use as image file argument."""
    return io.BytesIO(_make_solid_png(r, g, b))


def _skip_on_error(e, model: str) -> None:
    """Skip only on rate limit / quota / model not available, fail otherwise."""
    err = str(e).lower()
    if any(x in err for x in ("429", "quota", "rate_limit", "ratelimit", "resource_exhausted", "throttling")):
        pytest.skip(f"Rate limit / quota exceeded for {model}: {e}")
    raise e


def _assert_image_item(item) -> None:
    """Assert response item contains url or b64_json."""
    assert item.url or item.b64_json, (
        "Image response item has neither url nor b64_json"
    )


def _assert_b64_valid(b64: str) -> None:
    """Assert b64_json decodes to a valid PNG or JPEG."""
    raw = base64.b64decode(b64)
    assert len(raw) > 500, f"Decoded image too small ({len(raw)} bytes)"
    assert raw[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1"), (
        "Decoded data doesn't look like a valid PNG or JPEG"
    )


def _image_dimensions(b64: str) -> tuple[int, int]:
    raw = base64.b64decode(b64)
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", raw[16:24])

    if raw.startswith(b"\xff\xd8"):
        offset = 2
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        while offset + 4 <= len(raw):
            if raw[offset] != 0xFF:
                offset += 1
                continue
            while offset < len(raw) and raw[offset] == 0xFF:
                offset += 1
            if offset >= len(raw):
                break
            marker = raw[offset]
            offset += 1
            if marker in sof_markers:
                height = int.from_bytes(raw[offset + 3:offset + 5], "big")
                width = int.from_bytes(raw[offset + 5:offset + 7], "big")
                return width, height
            if marker == 0xD9 or marker == 0xDA:
                break
            segment_length = int.from_bytes(raw[offset:offset + 2], "big")
            if segment_length < 2:
                break
            offset += segment_length

    raise AssertionError("Unable to determine generated image dimensions")


# ---------------------------------------------------------------------------
# Text-to-Image Generation
# ---------------------------------------------------------------------------

class TestGeminiImageGeneration:
    """Text-to-image generation with Gemini native image models."""

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_basic_generation(self, openai_client, model):
        """Basic image generation returns at least one image."""
        try:
            resp = openai_client.images.generate(
                model=model,
                prompt="A beautiful mountain landscape at sunset",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp is not None
        assert len(resp.data) >= 1
        _assert_image_item(resp.data[0])

    @pytest.mark.parametrize(
        ("model", "size", "expected_dimensions"),
        [
            ("gemini-2.5-flash-image", "896x1152", (896, 1152)),
            ("gemini-3-pro-image-preview", "1792x2400", (1792, 2400)),
            ("gemini-3.1-flash-image-preview", "1024x1024", (1024, 1024)),
            ("gemini-3.1-flash-image-preview", "3:4", (896, 1200)),
            ("gemini-3.1-flash-image-preview", "1792x2400", (1792, 2400)),
            ("gemini-3.1-flash-image-preview", "896x1152", (928, 1152)),
            ("gemini-3.1-flash-image-preview", "640x1024", (848, 1264)),
        ],
    )
    def test_generation_respects_size_profile(
        self, openai_client, model, size, expected_dimensions
    ):
        """Size is mapped through the selected Gemini model's output grid."""
        try:
            resp = openai_client.images.generate(
                model=model,
                prompt="A mountain landscape at sunset",
                response_format="b64_json",
                n=1,
                size=size,
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp.data, "Response data is empty"
        b64 = resp.data[0].b64_json
        assert b64, "Model did not return b64_json"
        assert _image_dimensions(b64) == expected_dimensions

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_generation_count(self, openai_client, model):
        """n=1 returns exactly one image."""
        try:
            resp = openai_client.images.generate(
                model=model,
                prompt="A blue mountain lake at sunrise",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert len(resp.data) == 1, f"Expected 1 image, got {len(resp.data)}"

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_b64_decodable(self, openai_client, model):
        """b64_json response decodes to a valid PNG or JPEG."""
        try:
            resp = openai_client.images.generate(
                model=model,
                prompt="A red cube on a white background, 3D render",
                response_format="b64_json",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp.data, "Response data is empty"
        b64 = resp.data[0].b64_json
        if not b64:
            pytest.skip("Model returned url instead of b64_json")
        _assert_b64_valid(b64)

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_different_prompts_different_results(self, openai_client, model):
        """Different prompts produce different b64 results (no stale cache)."""
        try:
            resp1 = openai_client.images.generate(
                model=model,
                prompt="A red apple on a wooden table",
                response_format="b64_json",
                n=1,
                size="1024x1024",
            )
            resp2 = openai_client.images.generate(
                model=model,
                prompt="A blue ocean wave at night",
                response_format="b64_json",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        b64_1 = resp1.data[0].b64_json or ""
        b64_2 = resp2.data[0].b64_json or ""
        if b64_1 and b64_2:
            assert b64_1 != b64_2, (
                "Two different prompts returned identical b64_json — possible caching issue"
            )

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_detailed_prompt(self, openai_client, model):
        """Detailed prompt with style description generates a valid image."""
        try:
            resp = openai_client.images.generate(
                model=model,
                prompt="Futuristic city with neon lights, flying cars, tall skyscrapers, night time, cyberpunk style",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert len(resp.data) >= 1
        _assert_image_item(resp.data[0])


# ---------------------------------------------------------------------------
# Image Editing — single image
# ---------------------------------------------------------------------------

class TestGeminiImageEdit:
    """Image editing with Gemini native image models (single image + prompt)."""

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_reachable(self, openai_client, model):
        """images.edit() with one image returns a result."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=_png_bytes(0, 128, 0),
                prompt="Add a sunset sky background",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp is not None
        assert len(resp.data) >= 1
        _assert_image_item(resp.data[0])

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_returns_image(self, openai_client, model):
        """Response contains at least one image in data."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=_png_bytes(255, 0, 0),
                prompt="Make the background blue",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert len(resp.data) >= 1, f"Expected at least 1 image, got {len(resp.data)}"
        _assert_image_item(resp.data[0])

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_b64_decodable(self, openai_client, model):
        """If response contains b64_json, it decodes to a valid PNG/JPEG."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=_png_bytes(0, 100, 0),
                prompt="Add snow on top",
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp.data, "Response data is empty"
        b64 = resp.data[0].b64_json
        if not b64:
            pytest.skip("Model returned url instead of b64_json")
        _assert_b64_valid(b64)


# ---------------------------------------------------------------------------
# Image Editing — multiple images
# ---------------------------------------------------------------------------

class TestGeminiImageEditMulti:
    """Editing with multiple input images (image-to-image composition)."""

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_multi_edit_returns_image(self, openai_client, model):
        """images.edit() with two images returns at least one image."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=[
                    _png_bytes(100, 200, 100),
                    _png_bytes(200, 50, 50),
                ],
                prompt=(
                    "Combine these two images: place the object from the "
                    "second image into the scene of the first image."
                ),
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert len(resp.data) >= 1, f"Expected at least 1 image, got {len(resp.data)}"
        _assert_image_item(resp.data[0])

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_multi_edit_result_differs_from_input(self, openai_client, model):
        """Edit result is not identical to either input image."""
        input_png_1 = _make_solid_png(255, 0, 0)
        input_png_2 = _make_solid_png(0, 0, 255)
        input_b64_1 = base64.b64encode(input_png_1).decode()
        input_b64_2 = base64.b64encode(input_png_2).decode()

        try:
            resp = openai_client.images.edit(
                model=model,
                image=[
                    io.BytesIO(input_png_1),
                    io.BytesIO(input_png_2),
                ],
                prompt="Merge these two images into one artistic composition",
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp.data, "Response data is empty"
        result_b64 = resp.data[0].b64_json
        if not result_b64:
            pytest.skip("Model returned url instead of b64_json, can't compare")
        assert result_b64 != input_b64_1, "Result is identical to first input image"
        assert result_b64 != input_b64_2, "Result is identical to second input image"

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_multi_edit_composition(self, openai_client, model):
        """Composition prompt with two semantic images returns a result."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=[
                    _png_bytes(255, 200, 180),
                    _png_bytes(0, 0, 200),
                ],
                prompt=(
                    "First image is a person. Second image is clothing. "
                    "Dress the person in the clothing from the second image. "
                    "Preserve the person's face, pose, and proportions."
                ),
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert len(resp.data) >= 1
        _assert_image_item(resp.data[0])


# ---------------------------------------------------------------------------
# Real-photo editing — flower.jpg
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flower_jpg_bytes():
    """Load flower.jpg once per module; skip if file is missing."""
    if not os.path.exists(_FLOWER_JPG):
        pytest.skip(f"Test photo not found: {_FLOWER_JPG}")
    with open(_FLOWER_JPG, "rb") as f:
        return f.read()


class TestGeminiImageEditWithFlower:
    """Integration tests that use the real flower.jpg photo."""

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_flower_basic(self, openai_client, model, flower_jpg_bytes):
        """images.edit() on a real JPEG returns at least one image."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=io.BytesIO(flower_jpg_bytes),
                prompt="Добавь шмеля на цветок",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp is not None
        assert len(resp.data) >= 1
        _assert_image_item(resp.data[0])

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_flower_b64_decodable(self, openai_client, model, flower_jpg_bytes):
        """Edit of flower.jpg returns valid PNG or JPEG in b64_json."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=io.BytesIO(flower_jpg_bytes),
                prompt="Добавь шмеля на цветок",
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp.data, "Response data is empty"
        b64 = resp.data[0].b64_json
        if not b64:
            pytest.skip("Model returned url instead of b64_json")
        _assert_b64_valid(b64)

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_flower_usage_format(self, openai_client, model, flower_jpg_bytes):
        """Usage field uses images API format: input_tokens / output_tokens (not prompt_tokens)."""
        try:
            resp = openai_client.images.edit(
                model=model,
                image=io.BytesIO(flower_jpg_bytes),
                prompt="Добавь шмеля на цветок",
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp is not None

        usage = getattr(resp, "usage", None)
        if usage is None:
            pytest.skip("Model did not return usage metadata")

        # Must have the images-API fields, not the chat-completions fields
        assert hasattr(usage, "input_tokens"), (
            f"usage.input_tokens missing — got fields: {vars(usage)}"
        )
        assert hasattr(usage, "output_tokens"), (
            f"usage.output_tokens missing — got fields: {vars(usage)}"
        )
        assert hasattr(usage, "input_tokens_details"), (
            f"usage.input_tokens_details missing — got fields: {vars(usage)}"
        )

        assert usage.input_tokens > 0, "input_tokens should be positive"
        assert usage.output_tokens > 0, "output_tokens should be positive"

        details = usage.input_tokens_details
        assert details is not None
        image_toks = getattr(details, "image_tokens", None)
        assert image_toks is not None and image_toks > 0, (
            f"input_tokens_details.image_tokens should be > 0 for a real photo input, got {image_toks}"
        )

    @pytest.mark.parametrize("model", TestModels.GEMINI_IMAGE_MODELS)
    def test_edit_flower_result_differs_from_input(self, openai_client, model, flower_jpg_bytes):
        """Edit result is not byte-identical to the input photo."""
        input_b64 = base64.b64encode(flower_jpg_bytes).decode()

        try:
            resp = openai_client.images.edit(
                model=model,
                image=io.BytesIO(flower_jpg_bytes),
                prompt="Добавь шмеля на цветок",
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )
        except Exception as e:
            _skip_on_error(e, model)

        assert resp.data, "Response data is empty"
        result_b64 = resp.data[0].b64_json
        if not result_b64:
            pytest.skip("Model returned url instead of b64_json, can't compare bytes")

        assert result_b64 != input_b64, (
            "Edit result is byte-identical to the input — model may have returned the original unchanged"
        )
