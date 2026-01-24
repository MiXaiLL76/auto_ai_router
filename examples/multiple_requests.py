#!/usr/bin/env python3
"""
Multiple requests example for auto_ai_router
Tests round-robin balancing and rate limiting
"""

from openai import OpenAI
import os
import time
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

def make_request(question_num):
    """Make a single request and return timing info"""
    questions = [
        "What is 2+2?",
        "Name a color.",
        "What day comes after Monday?",
        "What is the opposite of hot?",
        "Name a fruit.",
        "What is the capital of Japan?",
        "What animal says 'meow'?",
        "What color is the sky?",
        "How many legs does a dog have?",
        "What is 10-5?"
    ]

    question = questions[question_num % len(questions)]

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.5,
            max_tokens=50
        )

        duration = time.time() - start_time
        answer = response.choices[0].message.content.strip()

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "duration": duration,
            "tokens": response.usage.total_tokens
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "duration": duration
        }

def main():
    num_requests = 10
    print(f"Sending {num_requests} requests to test load balancing...\n")

    results = []
    for i in range(num_requests):
        print(f"Request {i+1}/{num_requests}...", end=" ", flush=True)
        result = make_request(i)
        results.append(result)

        if result["success"]:
            print(f"✓ ({result['duration']:.2f}s, {result['tokens']} tokens)")
        else:
            print(f"✗ Error: {result['error']}")

        # Small delay to avoid overwhelming the router
        time.sleep(0.5)

    # Print summary
    print("\n=== Summary ===")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        avg_tokens = sum(r["tokens"] for r in successful) / len(successful)
        print(f"Average duration: {avg_duration:.2f}s")
        print(f"Average tokens: {avg_tokens:.0f}")

    if failed:
        print("\nFailed requests:")
        for r in failed:
            print(f"  - {r['question']}: {r['error']}")

if __name__ == "__main__":
    main()
