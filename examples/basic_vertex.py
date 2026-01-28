#!/usr/bin/env python3
"""
Vertex AI examples for auto_ai_router
Tests text generation and image generation with Gemini models
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get master key from environment or use default
master_key = os.getenv("ROUTER_MASTER_KEY", "sk-your-master-key-here")
base_url = os.getenv("ROUTER_BASE_URL", "http://localhost:8080/v1")

# Configure OpenAI client to use the router
client = OpenAI(
    api_key=master_key,
    base_url=base_url,
    max_retries=0  # Disable retries to avoid multiple requests
)

def test_text_generation():
    """Test basic text generation with Gemini"""
    print("=== Testing Text Generation with Gemini ===")

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a creative writing assistant."},
                {"role": "user", "content": "Write a short poem about artificial intelligence in 4 lines."}
            ],
            temperature=0.8,
            max_tokens=200
        )

        print(f"Model: {response.model}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"\nPoem:\n{response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"Text generation error: {e}")
        return False

def test_code_generation():
    """Test code generation with Gemini"""
    print("\n=== Testing Code Generation with Gemini ===")

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a Python programming expert."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers using recursion with memoization."}
            ],
            temperature=0.3,
            max_tokens=300
        )

        print(f"Model: {response.model}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"\nCode:\n{response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"Code generation error: {e}")
        return False

def test_conversation():
    """Test multi-turn conversation with Gemini"""
    print("\n=== Testing Multi-turn Conversation ===")

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that remembers context."},
                {"role": "user", "content": "I'm planning a trip to Japan. What are the must-visit places?"},
                {"role": "assistant", "content": "For Japan, I'd recommend Tokyo for modern culture, Kyoto for traditional temples, Osaka for food, and Mount Fuji for natural beauty."},
                {"role": "user", "content": "What's the best time to visit the places you mentioned?"}
            ],
            temperature=0.7,
            max_tokens=250
        )

        print(f"Model: {response.model}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"\nResponse:\n{response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"Conversation error: {e}")
        return False

def test_image_generation():
    """Test image generation with Imagen"""
    print("\n=== Testing Image Generation with Imagen ===")

    try:
        response = client.images.generate(
            model="imagen-3.0-fast-generate-001",
            prompt="A serene Japanese garden with cherry blossoms and a small pond, digital art style",
            size="1024x1024",
            quality="standard",
            n=1
        )

        print(f"Generated {len(response.data)} image(s)")
        for i, image in enumerate(response.data):
            if image.b64_json:
                print(f"Image {i+1}: Base64 data received ({len(image.b64_json)} characters)")
                # Optionally save the image
                # with open(f"generated_image_{i+1}.png", "wb") as f:
                #     f.write(base64.b64decode(image.b64_json))
            elif image.url:
                print(f"Image {i+1}: URL - {image.url}")

        return True

    except Exception as e:
        print(f"Image generation error: {e}")
        return False

def test_streaming():
    """Test streaming response with Gemini"""
    print("\n=== Testing Streaming Response ===")

    try:
        stream = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Tell me a very short story about a robot learning to paint."}
            ],
            temperature=0.8,
            max_tokens=150,
            stream=True
        )

        print("Streaming story:")
        full_content = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content

        print(f"\n\nFull story length: {len(full_content)} characters")
        return len(full_content) > 0

    except Exception as e:
        print(f"Streaming error: {e}")
        return False

def main():
    print("Testing Vertex AI integration through auto_ai_router")
    print("=" * 60)

    results = []

    # Test different capabilities
    results.append(("Text Generation", test_text_generation()))
    results.append(("Code Generation", test_code_generation()))
    results.append(("Conversation", test_conversation()))
    results.append(("Image Generation", test_image_generation()))
    results.append(("Streaming", test_streaming()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
