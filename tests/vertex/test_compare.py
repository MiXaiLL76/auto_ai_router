"""
Compare OpenAI API (router) vs GenAI API (direct Vertex AI).
Verifies conversion: text, streaming, images, tool_calls, structured output.

pytest tests/vertex/test_compare.py -v -s
"""

import json
import os
import pytest
from test_helpers import TestModels, StreamingValidator
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

TOKEN_TOLERANCE_PCT = 10.0
TOKEN_TOLERANCE_ABS = 5
MAX_TOKENS = 512


def _gt_tokens(resp):
    """(text_tokens, thoughts, prompt) from GenAI response."""
    m = resp.usage_metadata
    thoughts = getattr(m, "thoughts_token_count", None) or 0
    return (m.candidates_token_count or 0, thoughts, m.prompt_token_count or 0)


def _dt_text_tokens(usage):
    """(text_tokens, reasoning) from OpenAI usage."""
    reasoning = 0
    if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
        reasoning = usage.completion_tokens_details.reasoning_tokens
    return usage.completion_tokens - reasoning, reasoning


def _assert_tokens_close(gt, dt, label):
    """Assert within 10% or 5 absolute difference."""
    if gt == 0 and dt == 0:
        return
    diff = abs(gt - dt)
    pct = diff / max(gt, dt) * 100 if max(gt, dt) > 0 else 0
    msg = f"{label}: GT={gt}, DT={dt}, diff={diff} ({pct:.1f}%)"
    print(f"  {msg}")
    assert pct <= TOKEN_TOLERANCE_PCT or diff <= TOKEN_TOLERANCE_ABS, msg


def _has_unicode(text, ranges):
    """Check text contains chars in unicode ranges [(lo, hi), ...]."""
    return any(lo <= ord(c) <= hi for c in text for lo, hi in ranges)


class TestCompareText:
    """Text requests: tokens, prompt_tokens, content adequacy."""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_text(self, openai_client, genai_client, model):
        """Basic text — token counts and content relevance."""
        prompt = "Explain what a neural network is in 2-3 sentences."

        gt = genai_client.chats.create(model=model).send_message(message=prompt, config={"max_output_tokens": MAX_TOKENS, "seed" : 42})
        dt = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            seed=42,
            max_tokens=MAX_TOKENS)

        gt_txt, gt_th, gt_pr = _gt_tokens(gt)
        dt_txt, dt_re = _dt_text_tokens(dt.usage)

        # Content adequacy
        for text, name in [(gt.text, "GT"), (dt.choices[0].message.content, "DT")]:
            assert len(text) > 20, f"{name} too short"
            assert any(k in text.lower() for k in ["neural", "network", "neuron", "layer"]), \
                f"{name} off-topic: {text[:80]}"

        # Prompt tokens must match (identical input)
        assert gt_pr == dt.usage.prompt_tokens, \
            f"Prompt tokens mismatch: GT={gt_pr}, DT={dt.usage.prompt_tokens}"

        print(f"\n  GT: {gt_txt} text + {gt_th} thoughts | DT: {dt_txt} text + {dt_re} reasoning")
        _assert_tokens_close(gt_txt, dt_txt, "Text completion")

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_multilingual(self, openai_client, genai_client, model):
        """Multilingual — non-Latin scripts survive conversion."""
        prompt = "Say hello in Chinese, Arabic, and Russian. One line per language."

        gt = genai_client.chats.create(model=model).send_message(
            message=prompt,
            config={"max_output_tokens": MAX_TOKENS, "seed": 42})
        dt = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS, seed=42)

        gt_text = gt.text
        dt_text = dt.choices[0].message.content

        print(f"\n=== MULTILINGUAL DEBUG ===")
        print(f"GT text: {gt_text}")
        print(f"DT text: {dt_text}")
        print(f"Texts equal: {gt_text == dt_text}")
        print(f"GT usage: {gt.usage_metadata}")
        print(f"DT usage: {dt.usage}")

        for text, name in [(gt_text, "GT"), (dt_text, "DT")]:
            assert _has_unicode(text, [(0x4e00, 0x9fff)]), f"{name}: no Chinese chars"
            assert _has_unicode(text, [(0x0600, 0x06ff)]), f"{name}: no Arabic chars"
            assert _has_unicode(text, [(0x0400, 0x04ff)]), f"{name}: no Cyrillic chars"

        gt_txt, _, _ = _gt_tokens(gt)
        dt_txt, _ = _dt_text_tokens(dt.usage)

        print(f"\n  GT: {gt_txt} text tokens | DT: {dt_txt} text tokens")
        print(f"  GT full: {gt_text[:100]}")
        print(f"  DT full: {dt_text[:100]}")

        _assert_tokens_close(gt_txt, dt_txt, "Multilingual")


class TestCompareStreaming:
    """Streaming conversion: content and chunks."""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_streaming(self, openai_client, model):
        """Streaming vs non-streaming produce coherent content."""
        prompt = "List the 8 planets in our solar system."

        regular = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS, seed=42)

        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS, stream=True, seed=42)
        stream_text, chunks = StreamingValidator.collect_streaming_content(stream)

        assert chunks > 0, "No streaming chunks"
        assert len(stream_text) > 20, f"Streaming too short: {stream_text[:50]}"

        planets = ["mercury", "venus", "earth", "mars", "jupiter"]
        for text, name in [(regular.choices[0].message.content, "regular"), (stream_text, "stream")]:
            assert any(p in text.lower() for p in planets), f"{name} off-topic: {text[:80]}"

        print(f"\n  Regular: {len(regular.choices[0].message.content)} chars")
        print(f"  Stream: {len(stream_text)} chars, {chunks} chunks")


class TestCompareImages:
    """Vision/image request conversion."""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_image(self, openai_client, genai_client, model):
        """Image input — prompt tokens and content recognition."""
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
        text = "What painting is this? Answer in one sentence."

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        # Use local image file to avoid 403 download errors
        local_image_path = os.path.join(os.path.dirname(__file__), "Van_Gogh_Starry_Night.jpg")
        if not os.path.exists(local_image_path):
            pytest.skip(f"Local image not found at {local_image_path}")

        img = Image.open(local_image_path)

        print(f"\n=== IMAGE TEST DEBUG ===")
        print(f"Model: {model}")
        print(f"Image URL (for router): {url}")
        print(f"Local image path: {local_image_path}")

        print(f"\n[GT] Calling GenAI directly...")
        gt = genai_client.models.generate_content(
            model=model, contents=[text, img],
            config={"max_output_tokens": MAX_TOKENS, "seed": 42})
        print(f"[GT] Response: {gt.text[:100]}")

        print(f"\n[DT] Calling OpenAI client (router)...")
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}}
            ]}],
            "max_tokens": MAX_TOKENS,
            "seed": 42
        }
        print(f"[DT] Request payload: {json.dumps(request_payload, indent=2)}")
        dt = openai_client.chat.completions.create(**request_payload)
        print(f"[DT] Response: {dt.choices[0].message.content[:100]}")

        keywords = ["van gogh", "starry night", "gogh"]
        for resp, name in [(gt.text, "GT"), (dt.choices[0].message.content, "DT")]:
            assert any(k in resp.lower() for k in keywords), \
                f"{name} didn't recognize painting: {resp[:80]}"

        gt_txt, _, gt_pr = _gt_tokens(gt)
        dt_txt, _ = _dt_text_tokens(dt.usage)
        _assert_tokens_close(gt_pr, dt.usage.prompt_tokens, "Image prompt")
        _assert_tokens_close(gt_txt, dt_txt, "Image completion")

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_image_base64(self, openai_client, genai_client, model):
        """Image as base64 (data URL) — embedded inline instead of URL reference."""
        text = "What painting is this? Answer in one sentence."

        try:
            from PIL import Image
            import base64
        except ImportError:
            pytest.skip("PIL/base64 not available")

        # Use local image file
        local_image_path = os.path.join(os.path.dirname(__file__), "Van_Gogh_Starry_Night.jpg")
        if not os.path.exists(local_image_path):
            pytest.skip(f"Local image not found at {local_image_path}")

        # Load image and convert to base64 data URL
        with open(local_image_path, 'rb') as f:
            image_data = f.read()
        b64_image = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64_image}"

        img = Image.open(local_image_path)

        print(f"\n=== IMAGE BASE64 TEST DEBUG ===")
        print(f"Model: {model}")
        print(f"Image size: {len(image_data)} bytes")
        print(f"Data URL length: {len(data_url)} (first 100 chars: {data_url[:100]}...)")

        print(f"\n[GT] Calling GenAI directly...")
        gt = genai_client.models.generate_content(
            model=model, contents=[text, img],
            config={"max_output_tokens": MAX_TOKENS, "seed": 42})
        print(f"[GT] Response: {gt.text[:100]}")

        print(f"\n[DT] Calling OpenAI client with base64 data URL...")
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}],
            "max_tokens": MAX_TOKENS,
            "seed": 42
        }
        print(f"[DT] Request with data URL (truncated): {request_payload['messages'][0]['content'][1]['image_url']['url'][:100]}...")
        dt = openai_client.chat.completions.create(**request_payload)
        print(f"[DT] Response: {dt.choices[0].message.content[:100]}")

        # Verify both recognize the painting
        keywords = ["van gogh", "starry night", "gogh"]
        for resp, name in [(gt.text, "GT"), (dt.choices[0].message.content, "DT")]:
            assert any(k in resp.lower() for k in keywords), \
                f"{name} didn't recognize painting: {resp[:80]}"

        # Compare tokens
        gt_txt, _, gt_pr = _gt_tokens(gt)
        dt_txt, _ = _dt_text_tokens(dt.usage)
        _assert_tokens_close(gt_pr, dt.usage.prompt_tokens, "Base64 image prompt")
        _assert_tokens_close(gt_txt, dt_txt, "Base64 image completion")


class TestCompareImageGeneration:
    """Chat-based image generation request handling."""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_image_generation_instruction(self, openai_client, genai_client, model):
        """Image generation instruction — model interprets request and token counts match."""
        prompt = "Describe how to create a digital painting of a sunset over mountains. Include color palette and techniques."

        print(f"\n=== IMAGE GENERATION INSTRUCTION TEST ===")
        print(f"Model: {model}")

        print(f"\n[GT] Calling GenAI directly...")
        gt = genai_client.chats.create(model=model).send_message(
            message=prompt,
            config={"max_output_tokens": MAX_TOKENS, "seed": 42})
        print(f"[GT] Response length: {len(gt.text)} chars")
        print(f"[GT] Response preview: {gt.text[:100]}")

        print(f"\n[DT] Calling OpenAI client (router)...")
        dt = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS, seed=42)
        print(f"[DT] Response length: {len(dt.choices[0].message.content)} chars")
        print(f"[DT] Response preview: {dt.choices[0].message.content[:100]}")

        # Verify both have substantial responses
        for text, name in [(gt.text, "GT"), (dt.choices[0].message.content, "DT")]:
            assert len(text) > 50, f"{name} response too short"

        # Compare tokens
        gt_txt, _, gt_pr = _gt_tokens(gt)
        dt_txt, _ = _dt_text_tokens(dt.usage)

        print(f"\n  GT: {gt_txt} text tokens, {gt_pr} prompt tokens")
        print(f"  DT: {dt_txt} text tokens, {dt.usage.prompt_tokens} prompt tokens")

        _assert_tokens_close(gt_pr, dt.usage.prompt_tokens, "Generation instruction prompt")
        _assert_tokens_close(gt_txt, dt_txt, "Generation instruction completion")

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_creative_generation(self, openai_client, genai_client, model):
        """Creative generation — longer form content with consistent token counting."""
        prompt = "Write a short creative story (3-4 sentences) about an ancient artifact discovered in a library."

        print(f"\n=== CREATIVE GENERATION TEST ===")
        print(f"Model: {model}")

        print(f"\n[GT] Calling GenAI...")
        gt = genai_client.chats.create(model=model).send_message(
            message=prompt,
            config={"max_output_tokens": MAX_TOKENS, "seed": 42})

        print(f"\n[DT] Calling OpenAI client (router)...")
        dt = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS, seed=42)

        # Verify both responses are creative and substantive
        keywords = ["ancient", "artifact", "library", "discovered"]
        for text, name in [(gt.text, "GT"), (dt.choices[0].message.content, "DT")]:
            assert any(k.lower() in text.lower() for k in keywords), \
                f"{name} doesn't match prompt theme"
            assert len(text) > 80, f"{name} response too short for creative task"

        # Token comparison
        gt_txt, _, gt_pr = _gt_tokens(gt)
        dt_txt, _ = _dt_text_tokens(dt.usage)

        print(f"\n  GT: {gt_pr} prompt, {gt_txt} text tokens")
        print(f"  DT: {dt.usage.prompt_tokens} prompt, {dt_txt} text tokens")

        _assert_tokens_close(gt_pr, dt.usage.prompt_tokens, "Creative prompt")
        _assert_tokens_close(gt_txt, dt_txt, "Creative completion")


class TestCompareImageGenerationModels:
    """Image generation (Imagen) model comparison."""

    def test_image_generation_imagen3(self, openai_client, genai_client):
        """Image generation — Imagen 3.0 fast, compare GenAI vs Router with token counts."""
        model = "imagen-3.0-fast-generate-001"
        prompt = "A serene mountain lake at sunset with pine trees, digital art style"

        print(f"\n=== IMAGE GENERATION (IMAGEN 3.0) TEST ===")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")

        # Test GenAI direct call
        print(f"\n[GT] Calling GenAI image generation...")
        gt = genai_client.models.generate_images(
            model=model,
            prompt=prompt,
            config={"number_of_images": 1, "output_mime_type": "image/jpeg"}
        )
        print(f"[GT] Generated {len(gt.generated_images)} image(s)")

        gt_tokens = 0
        if hasattr(gt, 'usage_metadata') and gt.usage_metadata:
            gt_tokens = gt.usage_metadata.prompt_token_count or 0
            print(f"[GT] Prompt tokens: {gt_tokens}")

        # Test OpenAI/Router call
        print(f"\n[DT] Calling OpenAI images.generate (via router)...")
        dt = openai_client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        print(f"[DT] Generated {len(dt.data)} image(s)")

        dt_tokens = 0
        if hasattr(dt, 'usage') and dt.usage:
            dt_tokens = dt.usage.prompt_tokens or 0
            print(f"[DT] Prompt tokens: {dt_tokens}")

        # Verify images were generated
        assert len(gt.generated_images) > 0, "GT: No images generated"
        assert len(dt.data) > 0, "DT: No images generated"

        # Compare image data
        if dt.data[0].b64_json:
            dt_image_size = len(dt.data[0].b64_json)
            print(f"\n[DT] Image size (B64): {dt_image_size} chars")

        # Compare tokens if available
        if gt_tokens > 0 and dt_tokens > 0:
            diff = abs(gt_tokens - dt_tokens)
            pct = (diff / max(gt_tokens, dt_tokens)) * 100
            print(f"\n  Tokens: GT={gt_tokens}, DT={dt_tokens}, diff={diff} ({pct:.1f}%)")
            _assert_tokens_close(gt_tokens, dt_tokens, "Image generation prompt")
        else:
            print(f"\n  Tokens: GT={gt_tokens}, DT={dt_tokens} (limited availability)")

        print(f"\n✓ Image generation test passed")
        print(f"  GT: {len(gt.generated_images)} image(s)")
        print(f"  DT: {len(dt.data)} image(s)")

    def test_image_generation_multiple(self, openai_client, genai_client):
        """Image generation — multiple images (n=3) comparison."""
        model = "imagen-3.0-fast-generate-001"
        prompt = "A sunset over mountains"
        n_images = 3

        print(f"\n=== IMAGE GENERATION (MULTIPLE - n={n_images}) TEST ===")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")

        # Test GenAI direct call
        print(f"\n[GT] Calling GenAI image generation (n={n_images})...")
        gt = genai_client.models.generate_images(
            model=model,
            prompt=prompt,
            config={
                "number_of_images": n_images,
                "output_mime_type": "image/jpeg"
            }
        )
        gt_count = len(gt.generated_images)
        print(f"[GT] Generated {gt_count} image(s)")

        # Test OpenAI/Router call
        print(f"\n[DT] Calling OpenAI images.generate (n={n_images} via router)...")
        dt = openai_client.images.generate(
            model=model,
            prompt=prompt,
            n=n_images,
            size="1024x1024"
        )
        dt_count = len(dt.data)
        print(f"[DT] Generated {dt_count} image(s)")

        # Verify counts match
        assert gt_count == n_images, f"GT: Expected {n_images} images, got {gt_count}"
        assert dt_count == n_images, f"DT: Expected {n_images} images, got {dt_count}"
        assert gt_count == dt_count, f"Image count mismatch: GT={gt_count}, DT={dt_count}"

        # Verify each image has data
        for i, img in enumerate(dt.data):
            assert img.b64_json or img.url, f"Image {i+1} has no data"

        print(f"\n✓ Multiple image generation test passed")
        print(f"  Both generated exactly {n_images} images")

class TestCompareToolCall:
    """Tool call conversion."""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_tool_call(self, openai_client, genai_client, model):
        """Tool call — function name and arguments pass through correctly."""
        prompt = "What is the weather in Tokyo right now?"

        tool_openai = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }

        from google.genai import types
        tool_genai = types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get current weather for a city",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"city": types.Schema(type="STRING", description="City name")},
                    required=["city"]
                )
            )
        ])

        gt = genai_client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS, tools=[tool_genai], seed=42))
        dt = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool_openai], max_tokens=MAX_TOKENS, seed=42)

        # GT: extract function call
        gt_fc = None
        if gt.candidates and gt.candidates[0].content:
            for part in gt.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    gt_fc = part.function_call
                    break
        assert gt_fc, f"GT: no tool call. Text: {getattr(gt, 'text', 'N/A')}"

        # DT: extract tool call
        dt_tcs = dt.choices[0].message.tool_calls
        assert dt_tcs, f"DT: no tool call. Text: {dt.choices[0].message.content}"
        dt_tc = dt_tcs[0]

        # Function name
        assert gt_fc.name == "get_weather", f"GT wrong func: {gt_fc.name}"
        assert dt_tc.function.name == "get_weather", f"DT wrong func: {dt_tc.function.name}"

        # Arguments
        gt_args = dict(gt_fc.args) if gt_fc.args else {}
        dt_args = json.loads(dt_tc.function.arguments) if dt_tc.function.arguments else {}

        assert "city" in gt_args, f"GT: no 'city' in {gt_args}"
        assert "city" in dt_args, f"DT: no 'city' in {dt_args}"
        assert "tokyo" in gt_args["city"].lower(), f"GT city={gt_args['city']}"
        assert "tokyo" in dt_args["city"].lower(), f"DT city={dt_args['city']}"

        print(f"\n  GT: {gt_fc.name}({gt_args})")
        print(f"  DT: {dt_tc.function.name}({dt_args})")


class TestCompareStructuredOutput:
    """Structured output (JSON schema) conversion."""

    @pytest.mark.parametrize("model", TestModels.VERTEX_MODELS)
    def test_json_schema(self, openai_client, genai_client, model):
        """JSON schema — structure and content correctness."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "population": {"type": "integer"},
                "country": {"type": "string"}
            },
            "required": ["name", "population", "country"]
        }
        prompt = "Return data about Tokyo as JSON with name, population, country."

        # GT: GenAI with response_schema
        gt = genai_client.chats.create(model=model).send_message(
            message=prompt,
            config={
                "max_output_tokens": MAX_TOKENS,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "seed": 42,
            })

        # DT: Router with response_format
        dt = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "city_info", "schema": schema}
            },
            seed=42
            )

        gt_text_raw = gt.text.strip()
        dt_text_raw = dt.choices[0].message.content.strip()

        gt_data = json.loads(gt_text_raw)
        dt_data = json.loads(dt_text_raw)

        print(f"\n=== JSON SCHEMA DEBUG ===")
        print(f"GT raw text: {gt_text_raw}")
        print(f"DT raw text: {dt_text_raw}")
        print(f"Texts equal: {gt_text_raw == dt_text_raw}")
        print(f"GT usage_metadata: {gt.usage_metadata}")
        print(f"GT candidates_token_count: {gt.usage_metadata.candidates_token_count if gt.usage_metadata else 'N/A'}")
        print(f"GT prompt_token_count: {gt.usage_metadata.prompt_token_count if gt.usage_metadata else 'N/A'}")
        print(f"DT usage: {dt.usage}")
        print(f"DT completion_tokens: {dt.usage.completion_tokens}")
        print(f"DT prompt_tokens: {dt.usage.prompt_tokens}")
        print(f"DT completion_tokens_details: {dt.usage.completion_tokens_details}")
        print(f"DT prompt_tokens_details: {dt.usage.prompt_tokens_details}")

        # Schema structure validation
        for data, name in [(gt_data, "GT"), (dt_data, "DT")]:
            assert isinstance(data.get("name"), str), f"{name}: 'name' not string in {data}"
            assert isinstance(data.get("population"), (int, float)), \
                f"{name}: 'population' not number in {data}"
            assert isinstance(data.get("country"), str), f"{name}: 'country' not string in {data}"

        # Content correctness
        for data, name in [(gt_data, "GT"), (dt_data, "DT")]:
            assert "tokyo" in data["name"].lower(), f"{name}: name={data['name']}"
            assert "japan" in data["country"].lower(), f"{name}: country={data['country']}"
            assert data["population"] > 1_000_000, \
                f"{name}: population too low: {data['population']}"

        print(f"\n  GT: {json.dumps(gt_data, ensure_ascii=False)}")
        print(f"  DT: {json.dumps(dt_data, ensure_ascii=False)}")

        gt_txt, _, _ = _gt_tokens(gt)
        dt_txt, _ = _dt_text_tokens(dt.usage)

        print(f"  GT: {gt_txt} text tokens | DT: {dt_txt} text tokens")

        _assert_tokens_close(gt_txt, dt_txt, "JSON schema")
