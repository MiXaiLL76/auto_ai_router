"""
Anthropic Claude tool/function calling tests
Tests OpenAI -> Anthropic tool definition conversion and handling
"""

import pytest
from test_helpers import (
    TestModels, ResponseValidator, ToolDefinitions
)


class TestAnthropicBasicToolCalling:
    """Test basic tool calling functionality"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_simple_tool_definition(self, openai_client, model):
        """Test sending simple tool definition"""
        tools = [ToolDefinitions.get_weather_tool()]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)
        ResponseValidator.validate_usage(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_multiple_tools(self, openai_client, model):
        """Test API with multiple tool definitions"""
        tools = [
            ToolDefinitions.get_weather_tool(),
            ToolDefinitions.get_calculation_tool(),
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather like and what is 5+3?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_multiple_parameters(self, openai_client, model):
        """Test tool with multiple required and optional parameters"""
        tools = [ToolDefinitions.get_search_flights_tool()]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Find flights from New York to London for 3 people on 2024-06-15"
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_choice_required(self, openai_client, model):
        """Test forcing tool usage with tool_choice='required'"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Please add 5 and 3"}
            ],
            tools=tools,
            tool_choice="required",
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_choice_auto(self, openai_client, model):
        """Test tool_choice='auto' (default behavior)"""
        tools = [ToolDefinitions.get_weather_tool()]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Just say hello"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicComplexToolDefinitions:
    """Test complex tool parameter structures"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_nested_object_parameters(self, openai_client, model):
        """Test tool with nested object parameters"""
        tools = [ToolDefinitions.get_complex_tool_with_nested_params()]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Book me a hotel in Paris from 2024-06-01 to 2024-06-05"
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_enum_values(self, openai_client, model):
        """Test tool parameter with enum values"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": "Set an alarm",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time": {"type": "string"},
                            "frequency": {
                                "type": "string",
                                "enum": ["once", "daily", "weekly", "monthly"]
                            }
                        },
                        "required": ["time", "frequency"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Set a daily alarm for 7:30 AM"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_array_parameters(self, openai_client, model):
        """Test tool with array parameters"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email to multiple recipients",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipients": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Email addresses"
                            },
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        },
                        "required": ["recipients", "subject", "body"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Send an email to alice@example.com and bob@example.com with subject 'Meeting' and body 'Let's meet tomorrow'"
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicToolsAdvanced:
    """Test advanced tool calling scenarios"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_default_values(self, openai_client, model):
        """Test tool parameters with default values"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Search for artificial intelligence"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_description_details(self, openai_client, model):
        """Test tool with detailed descriptions and examples"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_discount",
                    "description": "Calculate discount amount based on original price and discount percentage. Example: price=100, discount=10 returns 10",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "original_price": {
                                "type": "number",
                                "description": "The original price before discount"
                            },
                            "discount_percent": {
                                "type": "number",
                                "description": "The discount percentage (0-100)"
                            }
                        },
                        "required": ["original_price", "discount_percent"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Calculate 20% discount on $50"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=150
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_optional_parameters(self, openai_client, model):
        """Test tool where only some parameters are required"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book_table",
                    "description": "Book a restaurant table",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant": {"type": "string"},
                            "date": {"type": "string"},
                            "time": {"type": "string"},
                            "party_size": {"type": "integer"},
                            "special_requests": {"type": "string"}
                        },
                        "required": ["restaurant", "date", "time", "party_size"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Book a table at Mario's for 4 people on 2024-06-15 at 19:00 with special dietary needs"
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_multiple_tools_different_categories(self, openai_client, model):
        """Test with multiple tools from different categories"""
        tools = [
            ToolDefinitions.get_weather_tool(),
            ToolDefinitions.get_search_flights_tool(),
            ToolDefinitions.get_calculation_tool(),
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris, find flights to London, and calculate 15% of 1000"
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=300
        )

        ResponseValidator.validate_chat_response(response)


class TestAnthropicToolsEdgeCases:
    """Test edge cases in tool calling"""

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_very_long_description(self, openai_client, model):
        """Test tool with lengthy description"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "complex_task",
                    "description": "This is a very detailed tool that performs complex analysis. "
                                  "It takes multiple parameters and returns comprehensive results. "
                                  "The tool is designed to handle edge cases and provide accurate "
                                  "information based on the input parameters provided by the user. "
                                  "It supports multiple data types and formats.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"}
                        },
                        "required": ["input"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Use the tool with input 'test'"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_single_tool_in_list(self, openai_client, model):
        """Test with single tool in tools list"""
        tools = [ToolDefinitions.get_weather_tool()]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)

    @pytest.mark.parametrize("model", TestModels.ANTHROPIC_MODELS)
    def test_tool_with_no_required_parameters(self, openai_client, model):
        """Test tool where no parameters are required"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone (optional)"
                            }
                        },
                        "required": []
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the current time?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100
        )

        ResponseValidator.validate_chat_response(response)
