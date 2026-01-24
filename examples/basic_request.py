#!/usr/bin/env python3
"""
Basic non-streaming request example for auto_ai_router
Tests a simple chat completion request
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
    print("Sending basic chat completion request...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=0.7,
            max_tokens=100
        )

        print("\n=== Response ===")
        print(f"Model: {response.model}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"\nAssistant: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
