"""
Shared pytest fixtures and configuration for auto_ai_router tests
"""

import os
import sys
import pytest
from openai import OpenAI

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

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


@pytest.fixture
def genai_client():
    """Create Google Generative AI client for direct Vertex API testing

    Configures genai to use Vertex AI with credentials from environment:
    - VERTEX_PROJECT_ID: Google Cloud project ID
    - VERTEX_LOCATION: Region (default: us-central1)
    - VERTEX_CREDENTIALS_FILE: Path to service account JSON key
    """
    if not GENAI_AVAILABLE:
        pytest.skip("google.genai not installed")

    # Get credentials from environment
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "global")
    credentials_file = os.getenv("VERTEX_CREDENTIALS_FILE")

    if not project_id:
        pytest.skip("VERTEX_PROJECT_ID environment variable not set")

    if credentials_file and not os.path.exists(credentials_file):
        pytest.skip(f"Credentials file not found: {credentials_file}")

    try:
        if credentials_file:
            # Set credentials for google.genai
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

        # Initialize client - google.genai will use credentials automatically
        client = genai.Client(
            api_key=os.getenv("GENAI_API_KEY") or None,
            vertexai=True
        )
        return client
    except Exception as e:
        pytest.skip(f"Failed to initialize genai client: {e}")
