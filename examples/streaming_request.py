#!/usr/bin/env python3
"""
Streaming request example for auto_ai_router
Tests Server-Sent Events (SSE) streaming functionality
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
    base_url=base_url
)

def main():
    print("Sending streaming chat completion request...")
    print("Assistant: ", end="", flush=True)

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short poem about programming."}
            ],
            temperature=0.8,
            max_tokens=200,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print("\n\n=== Stream Complete ===")
        print(f"Total characters received: {len(full_response)}")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
