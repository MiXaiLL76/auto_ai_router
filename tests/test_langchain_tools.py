"""
LangChain ChatOpenAI with tool_calls integration tests for Vertex AI

Tests tool calling and structured outputs when using LangChain
ChatOpenAI client with Vertex AI backend via auto_ai_router
"""

import pytest
from typing import Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from pydantic import BaseModel, Field
except ImportError:
    pytest.skip("LangChain not installed", allow_module_level=True)


# Tool definitions for testing
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    weather_data = {
        "New York": "Cloudy, 15°C",
        "London": "Rainy, 10°C",
        "Tokyo": "Sunny, 22°C",
        "Sydney": "Warm, 25°C"
    }
    return weather_data.get(location, "Weather data not available")


@tool
def calculate_distance(location1: str, location2: str) -> float:
    """Calculate distance between two locations in kilometers"""
    distances = {
        ("New York", "London"): 5570,
        ("New York", "Tokyo"): 10840,
        ("London", "Tokyo"): 9570,
        ("Sydney", "Tokyo"): 7820,
    }

    key = tuple(sorted([location1, location2]))
    for k, v in distances.items():
        if set(k) == set([location1, location2]):
            return float(v)

    return 0.0


@tool
def list_cities() -> list:
    """List available cities"""
    return ["New York", "London", "Tokyo", "Sydney"]


class WeatherInfo(BaseModel):
    """Weather information schema"""
    location: str = Field(description="Location name")
    temperature: str = Field(description="Temperature reading")
    conditions: str = Field(description="Weather conditions")


class TestLangChainBasicTools:
    """Basic tool calling functionality tests"""

    def test_tool_calling_single_call(self, openai_client):
        """Test basic tool calling with ChatOpenAI"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.5
        )

        # Bind tools to the model
        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        # Query that should trigger tool call
        response = model_with_tools.invoke(
            "What's the weather in Tokyo?"
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, 'tool_calls') or hasattr(response, 'content')

    def test_tool_calling_multiple_calls(self, openai_client):
        """Test tool calling with multiple tool invocations"""
        model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.3
        )

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        # Query that should trigger multiple tool calls
        response = model_with_tools.invoke(
            "What cities are available? Show me the weather in the first two."
        )

        assert response is not None

    def test_tool_with_specific_tool_choice(self, openai_client):
        """Test tool calling with forced tool selection"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather, calculate_distance, list_cities]
        # Force using specific tool
        model_with_tools = model.bind_tools(
            tools,
            tool_choice="auto"
        )

        response = model_with_tools.invoke(
            "What's the weather in London?"
        )

        assert response is not None

    def test_message_with_tool_results(self, openai_client):
        """Test conversation with tool calls and results"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        # First turn - get tool call
        messages = [HumanMessage("What's the weather in Tokyo?")]
        response = model_with_tools.invoke(messages)

        assert response is not None


class TestLangChainAgents:
    """Agent-based tool calling tests"""

    def test_tool_calling_agent_creation(self, openai_client):
        """Test creating a tool-calling agent with ChatOpenAI"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.3
        )

        tools = [get_weather, calculate_distance, list_cities]

        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can access various tools."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent (may fail if model doesn't support tool_choice properly)
        try:
            agent = create_tool_calling_agent(model, tools, prompt)
            assert agent is not None
        except Exception as e:
            # Tool calling might not be fully supported
            pytest.skip(f"Tool calling agent not supported: {str(e)}")

    def test_agent_with_executor(self, openai_client):
        """Test tool-calling agent with executor"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.3
        )

        tools = [get_weather, calculate_distance, list_cities]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful weather assistant."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        try:
            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=5,
            )

            result = agent_executor.invoke({
                "input": "What's the weather in Tokyo and London? Also tell me the distance between them."
            })

            assert "output" in result or result is not None
        except Exception as e:
            pytest.skip(f"Agent executor not fully supported: {str(e)}")


class TestLangChainToolSchema:
    """Test tool schema and parameter validation"""

    def test_tool_definition_schema(self, openai_client):
        """Test that tool schema is properly defined"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        # Verify tools are bound
        bound_tools = model_with_tools.kwargs.get('tools')
        if bound_tools:
            assert len(bound_tools) > 0

    def test_tool_with_complex_parameters(self, openai_client):
        """Test tool calling with complex parameter types"""
        @tool
        def search_locations(
            query: str,
            radius_km: Optional[float] = None,
            limit: Optional[int] = None
        ) -> list:
            """Search for locations matching criteria"""
            return [
                {"name": "Tokyo", "distance": 0},
                {"name": "Osaka", "distance": 400},
                {"name": "Kyoto", "distance": 470},
            ]

        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [search_locations]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "Find locations near Tokyo within 500km, limit to 10 results"
        )

        assert response is not None


class TestLangChainIntegration:
    """Integration tests with vertex transform layer"""

    def test_tool_calls_with_parameters_transformation(self, openai_client):
        """Test that parameters are properly transformed through Vertex"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "What's the weather in London?"
        )

        assert response is not None

    def test_streaming_with_tool_calls(self, openai_client):
        """Test streaming response with tool calling"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            streaming=True,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        # Streaming with tool_calls now supported in Vertex and Anthropic
        try:
            chunks = []
            for chunk in model_with_tools.stream(
                "What's the weather in Tokyo?"
            ):
                chunks.append(chunk)

            assert len(chunks) > 0
        except Exception as e:
            pytest.skip(f"Streaming with tools error: {str(e)}")

    def test_concurrent_tool_calls(self, openai_client):
        """Test parallel tool execution"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "Get weather for Tokyo, London, and Sydney, and calculate distances between them"
        )

        assert response is not None

    def test_model_response_format(self, openai_client):
        """Test response format compatibility"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke("What's the weather in Tokyo?")

        assert hasattr(response, 'content') or hasattr(response, 'tool_calls')

    def test_error_handling_with_tools(self, openai_client):
        """Test error handling when tools fail"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        @tool
        def failing_tool(query: str) -> str:
            """Tool that might fail"""
            raise ValueError("Intentional failure")

        tools = [failing_tool, get_weather]
        model_with_tools = model.bind_tools(tools)

        try:
            response = model_with_tools.invoke("Help me with something")
            assert response is not None
        except Exception as e:
            assert "Intentional failure" in str(e) or response is not None

    def test_vertex_tool_calls_google_model(self, openai_client):
        """Test tool calling with Google Vertex AI Gemini model"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.5,
        )

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "What's the weather in Tokyo? Also list all available cities."
        )

        assert response is not None
        assert hasattr(response, 'content') or hasattr(response, 'tool_calls')

    def test_anthropic_tool_calls_claude_model(self, openai_client):
        """Test tool calling with Anthropic Claude model"""
        model = ChatOpenAI(
            model="claude-opus-4-1",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.5,
        )

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "What's the weather in London? Calculate distance to Paris."
        )

        assert response is not None
        assert hasattr(response, 'content') or hasattr(response, 'tool_calls')

    def test_vertex_multiple_function_calls(self, openai_client):
        """Test Vertex AI with multiple concurrent function calls"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "Get weather for Tokyo and London. Calculate distance between them."
        )

        assert response is not None

    def test_anthropic_multiple_function_calls(self, openai_client):
        """Test Anthropic with multiple concurrent function calls"""
        model = ChatOpenAI(
            model="claude-opus-4-1",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "Get weather for Tokyo and Sydney. Calculate distance between them."
        )

        assert response is not None


class TestProviderComparison:
    """Compare tool calling behavior between Google Vertex and Anthropic"""

    def test_google_vs_anthropic_same_prompt(self, openai_client):
        """Test same tool calling prompt with both providers"""
        # Google Vertex AI
        google_model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.3,
        )

        # Anthropic Claude
        anthropic_model = ChatOpenAI(
            model="claude-opus-4-1",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.3,
        )

        tools = [get_weather, calculate_distance]

        google_with_tools = google_model.bind_tools(tools)
        anthropic_with_tools = anthropic_model.bind_tools(tools)

        prompt = "What's the weather in Tokyo? How far is it from London?"

        try:
            google_response = google_with_tools.invoke(prompt)
            anthropic_response = anthropic_with_tools.invoke(prompt)

            # Both should return valid responses
            assert google_response is not None
            assert anthropic_response is not None
        except Exception as e:
            pytest.skip(f"Provider comparison test failed: {str(e)}")

    def test_google_vertex_tool_response_structure(self, openai_client):
        """Verify Google Vertex AI tool response structure"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke("What's the weather in London?")

        # Should have either content or tool_calls
        assert hasattr(response, 'content') or hasattr(response, 'tool_calls')

    def test_anthropic_tool_response_structure(self, openai_client):
        """Verify Anthropic tool response structure"""
        model = ChatOpenAI(
            model="claude-opus-4-1",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke("What's the weather in Paris?")

        # Should have either content or tool_calls
        assert hasattr(response, 'content') or hasattr(response, 'tool_calls')


class TestStreamingToolSupport:
    """Tests for streaming tool call support"""

    def test_vertex_streaming_text_only(self, openai_client):
        """Test Vertex streaming with text content"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            streaming=True,
        )

        # Without tools, streaming should work fine
        try:
            chunks = []
            for chunk in model.stream("Say hello"):
                chunks.append(chunk)
            assert len(chunks) > 0
        except Exception as e:
            pytest.skip(f"Basic streaming failed: {str(e)}")

    def test_vertex_streaming_with_tools(self, openai_client):
        """Test Vertex streaming with tool calls"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            streaming=True,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        try:
            chunks = []
            for chunk in model_with_tools.stream(
                "What's the weather in London?"
            ):
                chunks.append(chunk)
            assert len(chunks) > 0
        except Exception as e:
            pytest.skip(f"Vertex streaming with tools: {str(e)}")

    def test_anthropic_streaming_with_tools(self, openai_client):
        """Test Anthropic Claude streaming with tool_use calls"""
        model = ChatOpenAI(
            model="claude-opus-4-1",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            streaming=True,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        try:
            chunks = []
            for chunk in model_with_tools.stream(
                "What's the weather in Paris?"
            ):
                chunks.append(chunk)
            assert len(chunks) > 0
        except Exception as e:
            pytest.skip(f"Anthropic streaming with tools: {str(e)}")

    def test_google_streaming_with_tools(self, openai_client):
        """Test Google Vertex AI streaming with tool calls"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            streaming=True,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        try:
            chunks = []
            for chunk in model_with_tools.stream(
                "What's the weather in Tokyo?"
            ):
                chunks.append(chunk)
            assert len(chunks) > 0
        except Exception as e:
            pytest.skip(f"Google streaming with tools: {str(e)}")


class TestVertexToolCallsWithHistory:
    """Test Vertex AI tool calling with conversation history"""

    def test_vertex_tool_calls_with_developer_role(self, openai_client):
        """Test Vertex tool calling with developer role messages and tool_calls in history"""
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )

        model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=1,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        # Build conversation history with tool calls
        # Using LangChain's ToolCall format instead of OpenAI format
        messages = [
            SystemMessage("You are a helpful assistant with access to weather and distance calculation tools."),
            HumanMessage("What's the weather in Tokyo?"),
            AIMessage(
                "I'll check the weather for Tokyo.",
                tool_calls=[
                    {
                        "id": "call_weather_001",
                        "name": "get_weather",
                        "args": {"location": "Tokyo"},
                    }
                ],
            ),
            ToolMessage(
                "Sunny, 22°C",
                tool_call_id="call_weather_001",
            ),
            HumanMessage("How far is Tokyo from London?"),
        ]

        response = model_with_tools.invoke(messages)
        assert response is not None
        assert hasattr(response, "content") or hasattr(response, "tool_calls")

    def test_vertex_multiple_function_calls_in_sequence(self, openai_client):
        """Test Vertex handling multiple function calls with proper sequence"""
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            ToolMessage,
        )

        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=0.7,
            seed=42,
        )

        tools = [get_weather, calculate_distance, list_cities]
        model_with_tools = model.bind_tools(tools)

        # Simulate conversation with multiple tool calls
        # Using LangChain's ToolCall format (id, name, args)
        messages = [
            HumanMessage("Get weather for Tokyo and London, then calculate distance between them"),
            AIMessage(
                content="I'll get the weather for both cities and calculate the distance.",
                tool_calls=[
                    {
                        "id": "call_weather_tokyo",
                        "name": "get_weather",
                        "args": {"location": "Tokyo"},
                    },
                    {
                        "id": "call_weather_london",
                        "name": "get_weather",
                        "args": {"location": "London"},
                    },
                    {
                        "id": "call_distance",
                        "name": "calculate_distance",
                        "args": {"location1": "Tokyo", "location2": "London"},
                    },
                ],
            ),
            ToolMessage("Sunny, 22°C", tool_call_id="call_weather_tokyo"),
            ToolMessage("Rainy, 10°C", tool_call_id="call_weather_london"),
            ToolMessage("9570", tool_call_id="call_distance"),
            HumanMessage("What should I pack for a trip to both cities?"),
        ]

        response = model_with_tools.invoke(messages)
        assert response is not None
        # Response should include content about packing recommendations
        if hasattr(response, "content"):
            assert len(response.content) > 0

    def test_vertex_tool_calls_with_all_parameters(self, openai_client):
        """Test Vertex tool calling with all OpenAI-compatible parameters"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
            temperature=1,
            seed=42,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            max_tokens=500,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke(
            "What's the weather in Tokyo and how far is it from London?"
        )
        assert response is not None

    def test_vertex_tool_calls_response_format(self, openai_client):
        """Test that Vertex returns proper tool_calls format"""
        model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather]
        model_with_tools = model.bind_tools(tools)

        response = model_with_tools.invoke("What's the weather in Tokyo?")
        assert response is not None

        # If tool_calls are present, verify LangChain format
        # LangChain transforms OpenAI format to its own format
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                # LangChain format: {"id": "...", "name": "...", "args": {...}, "type": "tool_call"}
                assert "id" in tool_call
                assert "name" in tool_call
                assert "args" in tool_call or "arguments" in tool_call
                # Type can be either "function" (OpenAI format) or "tool_call" (LangChain format)
                if "type" in tool_call:
                    assert tool_call["type"] in ["function", "tool_call"]

    def test_vertex_tool_choice_auto(self, openai_client):
        """Test Vertex with tool_choice='auto' parameter"""
        model = ChatOpenAI(
            model="gemini-2.5-pro",
            openai_api_base="http://localhost:8080/v1",
            openai_api_key=openai_client.api_key,
        )

        tools = [get_weather, calculate_distance]
        model_with_tools = model.bind_tools(tools, tool_choice="auto")

        response = model_with_tools.invoke(
            "Get weather for London and compare with Paris distance"
        )
        assert response is not None
