#!/usr/bin/env python3
"""
Test script for image generation through chat endpoint using tools
Tests generating images using chat API with tools: [{"image_generation": {}}]
"""

import base64
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# Initialize OpenAI client pointing to local proxy
client = OpenAI(
    api_key="sk-your-master-key-here",
    base_url="http://localhost:8080/v1",
)

prompt = "Generate a beautiful landscape image of mountains and a lake at sunset"

try:
    print("Sending chat request with image generation (tools method)...")
    print("Model: gemini-2.5-flash-image")
    print(f"Prompt: {prompt}")
    print()

    response = client.chat.completions.create(
        model="gemini-2.5-flash-image",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1,  # We don't need text response
        extra_body={
            "tools": [{"image_generation": {}}],  # Enable image generation
            "generation_config": {
                "response_mime_type": "image/png",  # "image/png" or "image/jpeg"
                "temperature": 0.4,
            }
        },
    )

    print(f"Response received!")
    print(f"Status: 200")
    print()

    for msg in response.choices:
        print(f"Message: {msg.message.to_dict().keys()}")

    # Extract images from response
    imgs = (response.choices or [{}])[0].message.images if hasattr(response.choices[0].message, "images") else []

    if not imgs:
        # Fallback: check if images are in different format
        if hasattr(response.choices[0].message, "content"):
            print(f"Content: {response.choices[0].message.content}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    saved = []
    for i, img in enumerate(imgs, 1):
        try:
            b64 = None
            if hasattr(img, "b64_json") and img.b64_json:
                b64 = img.b64_json
            elif "b64_json" in img:
                b64 = img["b64_json"]
            elif hasattr(img, "url") and img.url and img.url.startswith("data:image/"):
                b64 = img.url.split(",", 1)[1]

            if b64:
                print(f"Processing image {i}...")
                if isinstance(b64, str):
                    raw = base64.b64decode(b64)
                else:
                    raw = b64

                fp = output_dir / f"gemini_{int(time.time())}_{i}.png"
                with open(str(fp), "wb") as f:
                    f.write(raw)
                saved.append(str(fp))
                print(f"✓ Saved image {i} to {fp}")
                print(f"  File size: {len(raw)} bytes")
            else:
                print(f"⚠️ Image {i} has no valid data {b64=}")
        except Exception as e:
            print(f"Error processing image {i}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not saved:
        print("⚠️ No images in response. Full response:")
    else:
        print()
        print("Image generation completed successfully!")
        print(f"Saved images: {saved}")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
