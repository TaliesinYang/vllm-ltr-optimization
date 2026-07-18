import json

from ltr_training.label_input import LabelInput, canonical_schema_hash
from ltr_training.ood_conversion import (
    convert_bfcl_row,
    convert_toolathlon_row,
    sample_label_inputs,
    summarize_conversion,
)


BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
TOOLATHLON_REVISION = "6194034105bc27fa438447172be0e7b4e35396e4"


def test_label_input_is_separate_and_request_id_matches_sample_id() -> None:
    item = LabelInput(
        sample_id="bfcl:simple_1:0000",
        request_id="bfcl:simple_1:0000",
        prompt="hello",
        tool_schema='[{"name":"lookup"}]',
        history=(),
        session_id="simple_1",
        task_id="simple_1",
        source="bfcl",
        source_revision=BFCL_REVISION,
        category="simple",
    )

    assert item.request_id == item.sample_id
    assert "output_length" not in item.to_dict()


def test_bfcl_converter_keeps_only_first_assistant_invocation() -> None:
    row = {
        "id": "multi_turn_base_1",
        "question": [
            [{"role": "user", "content": "first request"}],
            [{"role": "user", "content": "second request"}],
        ],
        "path": ["Files.lookup"],
        "involved_classes": ["Files"],
    }
    function_index = {
        "Files.lookup": {
            "name": "Files.lookup",
            "description": "lookup",
            "parameters": {"properties": {}, "type": "dict"},
        }
    }

    item = convert_bfcl_row(
        row,
        revision=BFCL_REVISION,
        category="multi_turn_base",
        function_index=function_index,
    )

    assert item.prompt == "first request"
    assert item.history == ()
    assert json.loads(item.tool_schema)[0]["function"]["parameters"]["type"] == "object"
    assert item.task_id == "multi_turn_base_1"


def test_bfcl_multi_turn_uses_all_available_class_tools_not_answer_path_only() -> None:
    row = {
        "id": "multi_turn_base_2",
        "question": [[{"role": "user", "content": "first request"}]],
        "path": ["GorillaFileSystem.find"],
        "involved_classes": ["GorillaFileSystem"],
    }
    function_index = {
        "GorillaFileSystem": [
            {"name": "find", "parameters": {"type": "dict", "properties": {}}},
            {"name": "ls", "parameters": {"type": "dict", "properties": {}}},
        ]
    }

    item = convert_bfcl_row(
        row,
        revision=BFCL_REVISION,
        category="multi_turn_base",
        function_index=function_index,
    )

    assert {tool["function"]["name"] for tool in json.loads(item.tool_schema)} == {
        "find",
        "ls",
    }


def test_toolathlon_converter_deserializes_strings_and_stops_at_first_assistant() -> None:
    row = {
        "modelname_run": "model_1",
        "task_name": "book-trip",
        "request_id": "native-request",
        "config": json.dumps({"task_str": "first request"}),
        "messages": json.dumps(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second request"},
            ]
        ),
        "tool_calls": json.dumps(
            {"tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]}
        ),
    }

    item = convert_toolathlon_row(
        row,
        revision=TOOLATHLON_REVISION,
        category="model_1",
    )

    assert item.prompt == "first request"
    assert item.history == (("system", "system"),)
    assert item.task_id == "book-trip"
    assert json.loads(item.tool_schema)[0]["function"]["name"] == "search"
    assert "second request" not in json.dumps(item.to_dict())


def test_sampling_and_manifest_report_rows_tasks_inputs_and_two_disjoint_levels() -> None:
    items = [
        LabelInput(
            sample_id=f"bfcl:task-{index}:0000",
            request_id=f"bfcl:task-{index}:0000",
            prompt="same" if index < 2 else f"prompt-{index}",
            tool_schema='[{"name":"lookup"}]',
            history=(),
            session_id=f"task-{index}",
            task_id=f"task-{index // 2}",
            source="bfcl",
            source_revision=BFCL_REVISION,
            category="a" if index % 2 == 0 else "b",
        )
        for index in range(6)
    ]

    sampled = sample_label_inputs(items, sample_size=4, seed=17)
    manifest = summarize_conversion(sampled, sampling_seed=17)

    assert len(sampled) == 4
    assert manifest["row_count"] == 4
    assert manifest["unique_task_count"] <= 4
    assert manifest["unique_input_hash_count"] <= 4
    assert manifest["domain_separation"]["source_identity"] == "disjoint"
    assert manifest["domain_separation"]["schema_hash"] == "measured"
    assert canonical_schema_hash(sampled[0].tool_schema)
