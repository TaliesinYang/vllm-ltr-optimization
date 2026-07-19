import json

from ltr_training.tier2 import _toolace_tools, build_request


def test_build_request_includes_full_history_and_generation_contract() -> None:
    row = {
        "tool_schema": " [ ] ",
        "history": [
            ["human", "First question"],
            ["gpt", "First answer"],
            ["human", "Follow-up question"],
            ["gpt", "Follow-up answer"],
        ],
        "prompt": "Final question",
    }

    request = build_request(row, model="Qwen/Qwen3.5-9B")

    assert request == {
        "model": "Qwen/Qwen3.5-9B",
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up question"},
            {"role": "assistant", "content": "Follow-up answer"},
            {"role": "user", "content": "Final question"},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_toolace_tools_passes_through_openai_wrapped_tools() -> None:
    wrapped = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "calendar.create-event",
                    "description": "Create an event",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                    },
                },
            }
        ]
    )

    tools = _toolace_tools(wrapped)

    assert tools == json.loads(wrapped)
