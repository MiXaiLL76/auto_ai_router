"""
Common test helpers and fixtures to reduce code duplication
"""

import numpy as np
from typing import Any, List, Dict


class TestModels:
    """Standard test models for different providers"""

    # Anthropic models
    ANTHROPIC_MODELS = [
        "claude-opus-4-1",
    ]

    # OpenAI models
    OPENAI_MODELS = [
        "gpt-4o-mini",
    ]

    # Google Vertex AI models
    VERTEX_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    # Vertex AI Image models
    VERTEX_IMAGE_MODELS = [
        "imagen-4.0-fast-generate-001",
        "imagen-3.0-fast-generate-001",
    ]

    # Embedding models
    EMBEDDING_MODELS = [
        "text-embedding-3-small",
    ]

    # Image generation models
    IMAGE_MODELS = [
        "gpt-image-1-mini",
    ]


class ResponseValidator:
    """Validate API responses for consistency"""

    @staticmethod
    def validate_chat_response(response: Any) -> None:
        """Validate standard chat completion response"""
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None

    @staticmethod
    def validate_usage(response: Any) -> None:
        """Validate usage statistics in response"""
        assert hasattr(response, 'usage')
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    @staticmethod
    def validate_streaming_content(full_content: str, chunk_count: int) -> None:
        """Validate streaming response collected content"""
        assert chunk_count > 0, "Should receive multiple chunks"
        assert len(full_content) > 0, "Should have collected content"

    @staticmethod
    def validate_embedding_response(response: Any, expected_count: int = 1) -> None:
        """Validate embedding response"""
        assert response is not None
        assert hasattr(response, 'data')
        assert len(response.data) == expected_count
        assert len(response.data[0].embedding) > 0
        assert hasattr(response, 'usage')
        assert response.usage.total_tokens > 0


class ContentValidator:
    """Validate response content for semantic meaning"""

    @staticmethod
    def assert_contains_any(content: str, keywords: List[str], case_sensitive: bool = False) -> None:
        """Assert content contains at least one of the keywords"""
        check_content = content if case_sensitive else content.lower()
        keywords = keywords if case_sensitive else [k.lower() for k in keywords]
        assert any(kw in check_content for kw in keywords), (
            f"Expected at least one of {keywords} in content: {content}"
        )

    @staticmethod
    def assert_numeric_response(content: str, number: int | float) -> None:
        """Assert response contains the expected number"""
        assert str(number) in content or str(int(number)) in content, (
            f"Expected '{number}' in response: {content}"
        )


class ToolDefinitions:
    """Common tool definitions for testing"""

    @staticmethod
    def get_weather_tool() -> Dict[str, Any]:
        """Simple weather tool definition"""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }

    @staticmethod
    def get_search_flights_tool() -> Dict[str, Any]:
        """Flight search tool definition"""
        return {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search for flights between two cities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_city": {
                            "type": "string",
                            "description": "Departure city"
                        },
                        "to_city": {
                            "type": "string",
                            "description": "Arrival city"
                        },
                        "date": {
                            "type": "string",
                            "description": "Travel date (YYYY-MM-DD)"
                        },
                        "passengers": {
                            "type": "integer",
                            "description": "Number of passengers",
                            "default": 1
                        }
                    },
                    "required": ["from_city", "to_city", "date"]
                }
            }
        }

    @staticmethod
    def get_calculation_tool() -> Dict[str, Any]:
        """Calculator tool definition"""
        return {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

    @staticmethod
    def get_complex_tool_with_nested_params() -> Dict[str, Any]:
        """Complex tool with nested parameters for testing"""
        return {
            "type": "function",
            "function": {
                "name": "book_hotel",
                "description": "Book a hotel room",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "dates": {
                            "type": "object",
                            "properties": {
                                "check_in": {"type": "string"},
                                "check_out": {"type": "string"}
                            },
                            "required": ["check_in", "check_out"]
                        },
                        "room_preferences": {
                            "type": "object",
                            "properties": {
                                "beds": {"type": "integer"},
                                "smoking": {"type": "boolean"}
                            }
                        }
                    },
                    "required": ["location", "dates"]
                }
            }
        }


class VectorMath:
    """Vector operations for embedding tests"""

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def assert_similarity_order(
        embeddings: List[List[float]],
        similar_pairs: List[tuple],
        dissimilar_pairs: List[tuple]
    ) -> None:
        """Assert that similar pairs have higher similarity than dissimilar pairs"""
        for (i1, i2) in similar_pairs:
            for (j1, j2) in dissimilar_pairs:
                sim_similar = VectorMath.cosine_similarity(embeddings[i1], embeddings[i2])
                sim_dissimilar = VectorMath.cosine_similarity(embeddings[j1], embeddings[j2])
                assert sim_similar > sim_dissimilar, (
                    f"Expected similarity({i1},{i2})={sim_similar} > "
                    f"similarity({j1},{j2})={sim_dissimilar}"
                )


class ImageTestData:
    """Test images for multimodal tests"""

    @staticmethod
    def get_van_gogh_url() -> str:
        """URL of famous Van Gogh painting for testing"""
        return (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
            "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
            "1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
        )

    @staticmethod
    def get_red_pixel_base64() -> str:
        """Simple 1x1 red pixel PNG as base64"""
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    @staticmethod
    def build_image_url_message(url: str, text: str) -> Dict[str, Any]:
        """Build a message with image URL"""
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {"url": url}
                }
            ]
        }

    @staticmethod
    def build_base64_image_message(base64_data: str, text: str) -> Dict[str, Any]:
        """Build a message with base64 image"""
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_data}"}
                }
            ]
        }


class StreamingValidator:
    """Validate streaming responses"""

    @staticmethod
    def collect_streaming_content(stream: Any) -> tuple[str, int]:
        """Collect content from streaming response

        Returns:
            (full_content, chunk_count)
        """
        full_content = ""
        chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        return full_content, chunk_count

    @staticmethod
    def assert_valid_streaming_response(full_content: str, chunk_count: int,
                                       min_length: int = 1) -> None:
        """Assert streaming response is valid"""
        assert chunk_count > 0, "Should receive multiple chunks"
        assert len(full_content) >= min_length, "Should have collected content"
