"""
Shared pytest fixtures and configuration for auto_ai_router tests
"""

import os
import sys
import pytest
from openai import OpenAI

# Add tests directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from test_helpers import TestModels, ResponseValidator, ContentValidator


@pytest.fixture
def master_key():
    """Get master key from environment"""
    return os.getenv("ROUTER_MASTER_KEY", "sk-your-master-key-here")


@pytest.fixture
def base_url():
    """Get base URL from environment"""
    return os.getenv("ROUTER_BASE_URL", "http://localhost:8080/v1")


@pytest.fixture
def openai_client(master_key, base_url):
    """Create OpenAI client configured for the router"""
    return OpenAI(
        api_key=master_key,
        base_url=base_url,
        max_retries=0
    )


@pytest.fixture
def timeout_short():
    """Short timeout for quick tests"""
    return 30


@pytest.fixture
def timeout_long():
    """Long timeout for image generation tests"""
    return 180


@pytest.fixture
def test_models():
    """Get standard test models"""
    return TestModels()


@pytest.fixture
def response_validator():
    """Get response validator"""
    return ResponseValidator()


@pytest.fixture
def content_validator():
    """Get content validator"""
    return ContentValidator()
