#!/usr/bin/env python3
"""
Simple chat example with gpt-oss-120b model
Tests basic question-answer interaction through the proxy
"""

import sys
from openai import OpenAI

# Initialize OpenAI client pointing to local proxy
client = OpenAI(
    api_key="sk-your-master-key-here",
    base_url="http://localhost:8080/v1",
)

# Simple question for the model
question = "What are the main benefits of using a proxy server in a distributed system?"

try:
    print("Sending chat request to gpt-oss-120b...")
    print(f"Question: {question}")
    print()
    print("Answer:")
    print("-" * 50)

    # Create streaming response
    stream = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=512,
        temperature=0.7,
        stream=True,
    )

    # Stream the response chunks
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end="", flush=True)

    print()
    print("-" * 50)

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
