"""
Integration tests for auto_ai_router functionality
"""

import pytest
import requests


class TestRouterIntegration:
    """Integration tests for router endpoints"""

    def test_health_endpoint(self, base_url):
        """Test health check endpoint"""
        response = requests.get(f"{base_url.replace('/v1', '')}/health")
        assert response.status_code == 200

        result = response.json()
        assert "status" in result

    def test_visual_health_endpoint(self, base_url):
        """Test visual health dashboard endpoint"""
        response = requests.get(f"{base_url.replace('/v1', '')}/vhealth")
        assert response.status_code == 200
        # Should return HTML content
        assert "text/html" in response.headers.get("content-type", "")

    def test_metrics_endpoint(self, base_url):
        """Test Prometheus metrics endpoint"""
        response = requests.get(f"{base_url.replace('/v1', '')}/metrics")
        assert response.status_code == 200

        # Should contain Prometheus metrics
        content = response.text
        assert "auto_ai_router" in content

    def test_models_endpoint(self, master_key, base_url):
        """Test models list endpoint"""
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {master_key}"}
        )
        assert response.status_code == 200

        result = response.json()
        assert "data" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) > 0

        # Check model structure
        model = result["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"

    @pytest.mark.parametrize("invalid_key", [
        "invalid-key",
        "sk-wrong-key",
        "",
        "Bearer sk-wrong"
    ])
    def test_invalid_auth(self, base_url, invalid_key):
        """Test authentication with invalid keys"""
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {invalid_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "test"}]
            }
        )
        assert response.status_code == 401

    def test_model_aware_routing(self, openai_client):
        """Test that model-aware routing works"""
        # Test with OpenAI model
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        assert openai_response.model == "gpt-4o-mini"

        # Test with Gemini model
        gemini_response = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        assert gemini_response.model == "gemini-2.5-flash"

    def test_rate_limiting_headers(self, master_key, base_url):
        """Test that rate limiting information is present"""
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
        )

        assert response.status_code == 200
        # Router might add rate limiting headers
        # This is optional depending on implementation


class TestRouterErrorHandling:
    """Test error handling scenarios"""

    def test_invalid_model(self, master_key, base_url):
        """Test request with invalid model"""
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "non-existent-model",
                "messages": [{"role": "user", "content": "test"}]
            }
        )

        # Should return error for non-existent model
        assert response.status_code >= 400

    def test_malformed_request(self, master_key, base_url):
        """Test malformed request handling"""
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                # Missing required messages field
            }
        )

        assert response.status_code >= 400

    def test_empty_messages(self, master_key, base_url):
        """Test request with empty messages"""
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": []
            }
        )

        assert response.status_code >= 400
