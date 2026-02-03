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

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=512,
        temperature=0.7,
    )

    print("Response received!")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.completion_tokens} completion tokens")
    print()
    print("Answer:")
    print("-" * 50)

    if response.choices and len(response.choices) > 0:
        print(response.choices[0].message.content)
    else:
        print("No response received")
        sys.exit(1)

    print("-" * 50)

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
