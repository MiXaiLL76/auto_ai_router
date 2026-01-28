#!/usr/bin/env python3
"""
Simple image generation example with Vertex AI Imagen
Generates an image and saves it to file
"""

from openai import OpenAI
import os
import base64
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
    max_retries=0
)

def generate_and_save_image():
    """Generate an image and save it to file"""
    print("Generating image with Vertex AI Imagen...")

    try:
        response = client.images.generate(
            model="imagen-3.0-fast-generate-001",
            prompt="A cute robot painting a colorful landscape on a canvas, studio lighting, high quality digital art",
            size="1024x1024",
            n=1
        )

        if response.data and len(response.data) > 0:
            image_data = response.data[0]

            if image_data.b64_json:
                # Decode base64 and save to file
                image_bytes = base64.b64decode(image_data.b64_json)
                filename = "generated_robot_painting.png"

                with open(filename, "wb") as f:
                    f.write(image_bytes)

                print(f"✅ Image saved as {filename}")
                print(f"📊 Image size: {len(image_bytes)} bytes")
                return True
            else:
                print("❌ No image data received")
                return False
        else:
            print("❌ No images generated")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = generate_and_save_image()
    if success:
        print("\n🎉 Image generation completed successfully!")
    else:
        print("\n💥 Image generation failed!")
