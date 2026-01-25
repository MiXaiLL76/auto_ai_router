#!/usr/bin/env python3
"""
Basic embeddings example for auto_ai_router
Demonstrates text-embedding-3-small model usage for single and batch requests
"""

from openai import OpenAI
import os
import numpy as np
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


def single_embedding():
    """Example 1: Create embedding for a single text"""
    print("=== Single Text Embedding ===")

    text = "The quick brown fox jumps over the lazy dog"

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        embedding = response.data[0].embedding

        print(f"Text: {text}")
        print(f"Model: {response.model}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Total tokens used: {response.usage.total_tokens}")

        return embedding

    except Exception as e:
        print(f"Error: {e}")
        return None


def batch_embeddings():
    """Example 2: Create embeddings for multiple texts in one request"""
    print("\n=== Batch Text Embeddings ===")

    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "The weather is nice today",
        "Embeddings convert text to numerical vectors"
    ]

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        print(f"Number of texts: {len(texts)}")
        print(f"Model: {response.model}")
        print(f"Total tokens used: {response.usage.total_tokens}")
        print(f"\nResults:")

        embeddings = []
        for i, data in enumerate(response.data):
            embedding = data.embedding
            embeddings.append(embedding)
            print(f"  {i+1}. '{texts[i][:50]}...' -> {len(embedding)} dimensions")

        return embeddings

    except Exception as e:
        print(f"Error: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    print("Testing text-embedding-3-small model via auto_ai_router\n")

    # Example 1: Single embedding
    embedding1 = single_embedding()

    # Example 2: Batch embeddings
    embeddings = batch_embeddings()

    # Bonus: Show similarity between texts
    if embeddings and len(embeddings) >= 4:
        print("\n=== Similarity Examples ===")
        sim1 = cosine_similarity(embeddings[0], embeddings[3])  # ML vs Embeddings
        sim2 = cosine_similarity(embeddings[0], embeddings[2])  # ML vs Weather

        print(f"Similarity (ML & Embeddings): {sim1:.4f}")
        print(f"Similarity (ML & Weather): {sim2:.4f}")
        print(f"\nAs expected, semantically related texts have higher similarity!")


if __name__ == "__main__":
    main()
