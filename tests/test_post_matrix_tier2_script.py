from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vllm_replay_server_allows_eight_sequences_at_point_nine_memory() -> None:
    script = (ROOT / "scripts" / "run_post_matrix_tier2.sh").read_text()

    assert "--max-num-seqs 8" in script
    assert "--gpu-memory-utilization 0.90" in script


def test_vllm_replay_server_uses_the_verified_local_snapshot_offline() -> None:
    script = (ROOT / "scripts" / "run_post_matrix_tier2.sh").read_text()

    assert "export HF_HUB_CACHE=/hy-tmp/hf" in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert "export TRANSFORMERS_OFFLINE=1" in script
