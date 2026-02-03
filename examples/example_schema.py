#!/usr/bin/env python3
"""
Test script for schema evaluation endpoint using structured JSON responses
Tests the /v1/chat/completions endpoint with JSON Schema validation
"""

import json
import sys
from openai import OpenAI

# Initialize OpenAI client pointing to local proxy
client = OpenAI(
    api_key="sk-your-master-key-here",
    base_url="http://localhost:8080/v1",
)

# Sample instruction for evaluation
instruction = """You are an AI evaluator for assessing response quality based on a JSON criteria.
Evaluate one criterion based on the input JSON data.

## Rules
- Objectivity: no assumptions
- Accuracy: follow the judge fields strictly
- Context: consider ALL messages in the dialogue
- Conservatism: score=true only if pass_condition is met AND fail_condition is not met

## Input data
dialogue: array of messages with role and text fields
judge: object with judge_id, instruction, pass_condition, fail_condition, mandatory, mandatory_condition

## Algorithm
1. Check applicability:
   - If mandatory=true → judge is applicable
   - If mandatory=false → judge is applicable ONLY if mandatory_condition is explicitly satisfied
   - If mandatory=false and mandatory_condition is not satisfied → NOT_APPLICABLE

2. Evaluate - only if judge is applicable:
   - Study the entire dialogue
   - Correlate with instruction, pass_condition and fail_condition
   - Set true ONLY if pass_condition is met AND fail_condition is not met
   - In all other cases → false

## Output format
Return ONLY one JSON object without explanations:

{"judge_id":"SAMPLE_SCOPE", "status":"NOT_APPLICABLE", "score": null}
{"judge_id":"SAMPLE_EFFICIENCY", "status":"EVALUATED", "score": true}
{"judge_id":"SAMPLE_QUALITY", "status":"EVALUATED", "score": false}

## Judge
{
  "judge_id": "SCOPE_001",
  "mandatory": false,
  "instruction": "Compare all explicitly mentioned subtasks from user with those addressed by assistant",
  "pass_condition": "Assistant addressed each explicitly mentioned subtask or indicated impossibility",
  "fail_condition": "Assistant ignores at least one explicitly mentioned subtask",
  "mandatory_condition": "Required if user explicitly listed two or more subtasks using conjunctions or action verbs"
}

## Sample Dialogue
{
  "dialogue": [
    {
      "role": "user",
      "text": "Help me find a suitable solution for my business problem and compare it with alternatives"
    },
    {
      "role": "assistant",
      "text": "I can help you with this. Let me propose several solutions...\\n\\n**Solution 1:** First approach - good for quick implementation\\n**Solution 2:** Second approach - reliable and stable\\n**Solution 3:** Third approach - excellent balance of features\\n\\nEach solution has its own advantages for different use cases."
    },
    {
      "role": "user",
      "text": "Can you also compare with the existing solutions available?"
    },
    {
      "role": "assistant",
      "text": "Unfortunately, I don't have access to a complete database of all existing solutions, so I cannot provide a precise comparison. The solutions I proposed are conceptually similar to existing alternatives, but this is a general comparison."
    }
  ]
}"""

# Prepare the request payload
payload = {
    "messages": [
        {
            "content": [
                {
                    "text": instruction,
                    "type": "text"
                }
            ],
            "role": "user"
        }
    ],
    "model": "claude-opus-4-1",
    "response_format": {
        "json_schema": {
            "name": "EvaluationSchema",
            "schema": {
                "$defs": {
                    "ScoreValue": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "title": "ScoreValue"
                    }
                },
                "properties": {
                    "score": {
                        "$ref": "#/$defs/ScoreValue"
                    },
                    "status": {
                        "type": "string",
                        "title": "Status"
                    },
                    "evaluation_reasoning": {
                        "type": "string",
                        "title": "Evaluation Reasoning",
                        "description": "Brief explanation that justifies your score if the judge is applicable."
                    },
                    "applicability_reasoning": {
                        "type": "string",
                        "title": "Applicability Reasoning",
                        "description": "Brief explanation that justifies your status, why the judge is or is not applicable."
                    }
                },
                "additionalProperties": False,
                "type": "object",
                "required": ["applicability_reasoning", "evaluation_reasoning", "status", "score"],
                "title": "EvaluationSchema"
            },
            "strict": True
        },
        "type": "json_schema"
    }
}

try:
    print("Sending chat completion request with JSON Schema validation...")
    print("Model: claude-opus-4-1")
    print("Response format: JSON Schema (strict)")
    print()

    response = client.chat.completions.create(**payload)

    print("Response received!")
    print("Status: Success")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.completion_tokens} completion tokens")
    print()

    # Extract and display the response content
    if response.choices and len(response.choices) > 0:
        message = response.choices[0].message
        print("Response content:")
        print("-" * 50)

        # Try to parse as JSON
        try:
            if isinstance(message.content, str):
                response_data = json.loads(message.content)
            else:
                response_data = message.content

            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(message.content)

        print("-" * 50)
        print()
        print("Schema validation completed successfully!")
    else:
        print("No response choices available")
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
