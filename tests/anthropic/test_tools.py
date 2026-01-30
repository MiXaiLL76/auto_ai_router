"""
Tool/Function calling tests for OpenAI -> Anthropic bridge
"""

import json
import pytest


class TestToolCalling:
    """Test tool/function calling capabilities"""

    def test_simple_tool_definition(self, openai_client):
        """Test sending tool definitions to the API"""
        tools = [
            {
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
        ]

        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        assert response.choices[0].message is not None
        assert response.usage.prompt_tokens > 0

    def test_tool_with_multiple_parameters(self, openai_client):
        """Test tool with multiple required and optional parameters"""
        tools = [
            {
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
        ]

        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Find flights from New York to London for 3 people on 2024-06-15"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        assert response.choices[0].message is not None

    def test_multiple_tools(self, openai_client):
        """Test API with multiple tool definitions"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time in a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "What's the weather like and what time is it in Tokyo?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        assert response.choices[0].message is not None

    def test_tool_choice_required(self, openai_client):
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
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Please add 5 and 3"}
            ],
            tools=tools,
            tool_choice="required",
            max_tokens=100
        )

        assert response.choices[0].message is not None

    def test_nested_object_parameters(self, openai_client):
        """Test tool with nested object parameters"""
        tools = [
            {
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
        ]

        response = openai_client.chat.completions.create(
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Book me a hotel in Paris from 2024-06-01 to 2024-06-05"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        assert response.choices[0].message is not None

    def test_tool_with_enum_values(self, openai_client):
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
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Set a daily alarm for 7:30 AM"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=150
        )

        assert response.choices[0].message is not None

    def test_tool_with_array_parameters(self, openai_client):
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
            model="claude-opus-4-1",
            messages=[
                {"role": "user", "content": "Send an email to alice@example.com and bob@example.com with subject 'Meeting' and body 'Let's meet tomorrow'"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200
        )

        assert response.choices[0].message is not None
