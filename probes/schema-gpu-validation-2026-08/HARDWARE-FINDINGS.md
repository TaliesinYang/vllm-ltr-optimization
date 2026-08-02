# GPU platform requirements for this experiment — measured 2026-08-01

No policy measurements were produced. What follows is the hardware/software
compatibility result from a full attempt on CloudLab, plus one capacity number.
Recorded because it constrains every later run and belongs in the paper's setup
section and limitations.

## The rig that was tried

CloudLab Wisconsin `c4130` (the only GPU node type with free capacity at the
time: d7525/A30, d8545/A100 and Clemson nvidiagh/GH200 were all 0 free).

| | |
|---|---|
| GPU | 4× Tesla V100-SXM2 16GB, **compute capability 7.0** (Volta) |
| Host | 2× Xeon E5-2667 (32 threads), 125 GB RAM, Ubuntu 22.04 |
| Driver | nvidia 580.173.02 (installed; the STD image ships none) |

## Result: Volta cannot run this experiment

Three independent blocks, in the order they were hit.

**1. Current vLLM has no Volta kernels.** `vllm 0.26.0` pulls
`torch 2.11.0+cu130`; CUDA 13 dropped Volta. Confirmed directly:

```
arch_list = ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
CUDA error: no kernel image is available for execution on the device
```

**2. The V1 engine refuses sm_70 outright.** Downgrading to `vllm 0.9.2` +
`torch 2.7.0+cu126` gives a working torch (`arch_list` includes `sm_70`, a real
fp16 matmul succeeds), but:

```
NotImplementedError: VLLM_USE_V1=1 is not supported with Compute Capability < 8.0
```

**3. V0's prefix-cache kernel does not compile on sm_70.** Falling back to
`VLLM_USE_V1=0` starts a server (XFormers backend) and serves requests — until a
request hits the prefix cache, at which point Triton fails to build the
prefix-prefill kernel:

```
vllm/attention/ops/prefix_prefill.py:850  in context_attention_fwd
  _fwd_kernel[grid](...)
triton/backends/nvidia/compiler.py       in make_llir
RuntimeError: PassManager::run failed
→ HTTP 500 on every request
```

The failing kernel is `forward_prefix`, which is *only* reached when prefix
caching is active. The one feature under study is the one that does not build.

**Requirement, therefore: compute capability ≥ 8.0 (Ampere or newer).** That is
also what vLLM V1 demands, so it is not merely a convenience.

Incidental consequences for the plan's model choice: Volta also rules out AWQ
(vLLM's AWQ kernels need sm_75+) and FP8 KV, so the `Qwen3-8B-AWQ` /
`Qwen3-4B-AWQ` configuration and the FP8-KV capacity sensitivity experiment are
both impossible on this class of hardware.

## The one usable measurement: single-GPU capacity

From a server that did start (Qwen3-4B, float16, TP=1, `--gpu-memory-utilization
0.90`, 16 GB V100):

```
KV cache capacity            30,080 tokens
max_model_len 32768          rejected (32768 > 30080)
Maximum concurrency at 28672 tokens/request:  1.13x
Model weights                7.56 GiB
```

So a single 16 GB card holds roughly **one** full-length agent request. The
170-tool workload (~55 K tokens) does not fit at all on one card, and the
concurrency sweep needs tensor parallelism or a larger card regardless of
architecture. This is architecture-independent arithmetic — it applies equally to
any 16 GB GPU.

## Scripts

`phase1/replay.py`, `phase1/run_phase1.sh`, `phase2/coldstart.py`,
`phase2/run_phase2.sh` are complete and were exercised end to end against a live
server; they stop only at the Triton failure above. Two fixes came out of that
run and are already applied:

- the server needs `--enable-auto-tool-choice --tool-call-parser hermes`, because
  the captured requests carry tools and vLLM defaults `tool_choice` to `auto`;
- both clients now surface the HTTP error *body*, not just the status code — the
  bare `HTTP 400` cost a whole arm before the real message was visible.
