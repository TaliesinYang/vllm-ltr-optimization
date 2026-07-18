from ltr_training.chat_anchor import build_sharegpt_anchor_workload, convert_sharegpt_row


def test_sharegpt_anchor_stops_before_first_assistant_and_uses_proxy() -> None:
    row = {
        "id": "chat-1",
        "conversations": [
            {"from": "system", "value": "system context"},
            {"from": "human", "value": "first prompt"},
            {"from": "gpt", "value": "first answer"},
            {"from": "human", "value": "later prompt"},
        ],
    }

    item = convert_sharegpt_row(row, revision="fixture")
    workload, manifest = build_sharegpt_anchor_workload(
        [item], lengths={item.sample_id: 12}, per_token_ms=2.5, seed=17
    )

    assert item.prompt == "first prompt"
    assert "later prompt" not in workload[0]["prompt"]
    assert workload[0]["baseline_service_ms"] == 30.0
    assert workload[0]["category"] == "anchor/sharegpt"
    assert workload[0]["max_tokens"] == 4096
    assert manifest["row_count"] == 1
