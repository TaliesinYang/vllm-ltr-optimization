# PARS implementation + train + test plan (workflow-generated, code-true 2026-06-22)

All anchors verified against the live code. Every line number, the detached `if listMLE`/second-chain quirk (trainer.py:125-132), the rankNet pair template, the else call-path (line 157), eval squeeze (line 189), configs, and `save_pretrained` are confirmed. Writing the plan.

---

# vllm-ltr Capstone: listMLE Baseline → PARS → Generalization-Gap Plan

代码已逐行核对（trainer.py / rankNet.py / listMLE.py / prefill_predictor.py / configs / train.sh）。所有 file:line 锚点真实存在。

**最小路径 (MIN PATH)**: ① 1 条命令 → ② 写 3 个改动 + 1 配置 + 1 损失文件 → ② 1 条命令 → ③ 1 个 ~50 行 eval 脚本 → 填 Table E1。跳过 vLLM 原生 BERT serving、cross-model、OPT/T5 backbone 消融（除非有时间）。

---

## ① Train the listMLE baseline NOW (while GPU is live)

这是教授要的"自己 train 一遍"基线。用 repo 自带的 350M 配置，复刻 README 的 lmsys/LTR/70b Tau≈0.62，run-id 标 `OURS`。

```bash
cd /hy-tmp/vllm-ltr/train   # GPU 上；本地镜像 /Users/alex/develop/vllm-ltr/train

# --tokenizer 必须指向本地 Llama-3 tokenizer 目录（8B/70B 同词表），否则会触发 gated HF 下载
LLAMA3_TOK=/hy-tmp/models/Meta-Llama-3-70B   # 改成实际本地路径

python trainer.py \
  --config configs/config_prefill_opt_350m.txt \
  --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
  --job-dir MODEL \
  --run-id opt-350m-llama3-70b-lmsys-listMLE-b32-OURS \
  --batch-size 32 \
  --label-group-size 10 \
  --loss listMLE \
  --tokenizer "$LLAMA3_TOK"
```

- **关键**: `--loss listMLE` 命中 trainer.py:125 那个**独立 `if`**（不是第二条链），走 else 调用路径 (trainer.py:157)，整 batch=一个 slate `[1, batch_size]`。config 必须是 **rank** 型 (`num_labels=1, mtype=rank`)，已确认 `config_prefill_opt_350m.txt` 就是。
- **每个 epoch 打印** `Kendall's Tau: ...`（trainer.py:195）= lmsys 同分布 10% 留出集上的 Tau。最后一个 epoch 的值就是 Table E1 的 **Tau(in-dist)**，无需额外跑。默认 `--epoch 5`、`--lr 2e-5`、Adam、无 scheduler（trainer.py:27-28,122）。
- **产物落地**（trainer.py:201-216, `save_pretrained` 默认 safetensors）:
  - `MODEL/results/<run-id>/finetuned/model.safetensors`  ← **要保存的 artifact**（fp16）
  - `MODEL/results/<run-id>/usage_config.json`  ← reload 时用，里面 `model.path` 已自动填成 finetuned 目录
- **预计耗时（估算，未实测）**: c20000:30000 = 1 万条 → 90% ≈ 9000 训练样本，batch 32 ≈ 281 step/epoch ×5。OPT-350M 在 4090 上约 **15–30 min**。125M (`config_prefill_opt.txt`) 更快 ~8–15 min。

> 时间紧可同时先跑 125M 版（`--config configs/config_prefill_opt.txt`，README Tau=0.64@8b）当快速 sanity，再跑 350M 当正式基线。

---

## ② Implement PARS (offline, no GPU needed — write while ① trains)

PARS = pairwise margin loss + delta-filter + BERT backbone。三处改动，全部已对照真实代码。

### (a) 新增 pairwise margin 损失函数

**新文件** `/Users/alex/develop/vllm-ltr/train/allrank/models/losses/marginRanking.py`（mirror rankNet.py:47-95 的 pair 构造，把 BCE 换成 hinge；并内置 delta-filter）:

```python
# PARS pairwise margin ranking loss (Tao et al. 2025, arXiv:2510.03243)
from itertools import product
import torch
from allrank.data.dataset_loading import PADDED_Y_VALUE


def marginRanking(y_pred, y_true, true_lengths=None, margin=1.0, delta=0.0,
                  padded_value_indicator=PADDED_Y_VALUE):
    """
    L = mean_{kept pairs} max(0, margin - (s_high - s_low)).
    HIGHER y_true (= shorter-generation bucket) must get the HIGHER score,
    keeping the SAME direction as listMLE (listMLE.py:37 sorts y_true descending),
    so the trained predictor stays plug-compatible with the existing scheduler.
    :param y_pred:       model scores, shape [1, slate_length]
    :param y_true:       length-bucket labels, shape [1, slate_length]
    :param true_lengths: raw generated token lengths [1, slate_length] for the delta filter
    :param margin:       hinge margin (PARS fixes 1.0)
    :param delta:        min relative length diff to keep a pair (PARS: 0.2 Llama/GPT-4)
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    pairs = list(product(range(y_true.shape[1]), repeat=2))   # mirrors rankNet.py:64
    pairs_true = y_true[:, pairs]
    pairs_pred = y_pred[:, pairs]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = pairs_pred[:, :, 0] - pairs_pred[:, :, 1]     # s_high - s_low

    keep = (true_diffs > 0) & (~torch.isinf(true_diffs))       # mirrors rankNet.py:76

    # PARS delta-filter (Eq.1): |L_A - L_B| / max(L_A, L_B) >= delta
    if delta > 0.0 and true_lengths is not None:
        lengths = true_lengths.to(y_pred.device).float()
        pl = lengths[:, pairs]
        l_hi, l_lo = pl[:, :, 0], pl[:, :, 1]
        rel_diff = (l_hi - l_lo).abs() / torch.clamp(torch.maximum(l_hi, l_lo), min=1.0)
        keep = keep & (rel_diff >= delta)

    pred_diffs = pred_diffs[keep]
    if pred_diffs.numel() == 0:
        return y_pred.new_zeros(1, requires_grad=True).squeeze()

    return torch.clamp(margin - pred_diffs, min=0.0).mean()    # = max(0, -y(s_A-s_B)+margin), y=+1
```

**为什么方向是 high-label→high-score 而不是 PARS 原文的 "higher score = longer"**: 本仓库 `__len2label__`（trainer.py:50-52）给**更短**的 generation **更高** label，listMLE 按 label 降序排（listMLE.py:37）。要让新 predictor 和已训练的 listMLE 分数、下游 scheduler 单调一致，margin loss 必须强制 high-label→high-score。我们保留仓库的 label 方向，只换损失公式——这点要在 deliverable 里说明（与 PARS 原文 y 定义的等价重述，非偏离）。

### (b) trainer.py 三处编辑（让 `--loss marginRanking` 生效 + 把原始长度传进损失做 delta-filter）

**1. import**（trainer.py:13 之后）:
```python
from allrank.models.losses.marginRanking import marginRanking
```

**2. argparse**（trainer.py:35 之后）:
```python
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--delta",  type=float, default=0.2)
```

**3. loss 调用点**（trainer.py:152-157，替换整段）。`origin_len` 是循环变量 (trainer.py:138)，DataLoader collate 成 `[batch]` 的 LongTensor，是真实 token 长度——正好喂给 delta-filter：
```python
                if args.loss == 'crossentropy':
                    assert labels.max().item() < predictor.model.num_labels
                    logits = outputs.view(-1, predictor.model.num_labels)
                    loss = loss_func(logits, labels.view(logits.size(0)))
                elif args.loss == 'marginRanking':
                    lengths = origin_len.reshape(1, -1)        # raw gen lengths -> [1, batch]
                    loss = marginRanking(outputs.view(1, -1), labels, lengths,
                                         margin=args.margin, delta=args.delta)
                else:
                    loss = loss_func(outputs.view(1, -1), labels)
```

- **不需要**改 trainer.py:125-132 的 dispatch 块（marginRanking 走自己的分支，`loss_func` 在该 loss 下从不被读）。若要 belt-and-suspenders，加 `elif args.loss == 'marginRanking': loss_func = marginRanking` 到**第二条链**（line 132 之后），**绝不要**接在 line 125 的独立 `if listMLE` 后面，否则破坏 listMLE 选择。
- **eval 端无需改**: marginRanking 非 crossentropy → 命中 trainer.py:188-189 的 `else: outputs.squeeze().tolist()`，Tau 照常算。
- `__init__.py` 注册可选（losses/__init__.py 是 star-import registry），因 trainer 直接 import 符号，**非必需**。

> **Delta-filter 的诚实偏离**: PARS 原文在**建 pair 数据集时**全局过滤一次；这里 pair 是每个 batch（slate=batch_size）内动态生成，filter 也在 batch 内。语义等价、实现更简，但属于实现差异，deliverable 要披露。若要严格复刻全局过滤，需要离线 `build_pairs.py` 预处理 + 改 dataset/collate（更大改动，非最小路径）。

### (c) BERT backbone 配置（仅训练/离线打分路径——零 Python 改动）

`prefill_predictor.py:25` 的 `AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)` 直接给出 HF `BertForSequenceClassification`：[CLS]→BertPooler(Linear+tanh)→dropout→Linear(768→1)→`.logits`（forward 走 trainer.py:80 的 rank 分支返回 `.logits`），即 PARS 要的 "[CLS]-pooled → linear → scalar"。tokenizer 也由 line 27 自动取 BERT WordPiece。

**新文件** `/Users/alex/develop/vllm-ltr/train/configs/config_prefill_bert.txt`（**不要**编辑 OPT 配置；**不要**加 `tokenizer_name` 字段，schema 没有它）:
```json
{
  "model": {
    "pred_model": "bert-base-uncased",
    "num_labels": 1,
    "mtype": "rank",
    "activation": null,
    "max_length": 512,
    "max_batch_size": 1000
  }
}
```

- **唯一必改的正确性项**: `max_length` 必须 ≤ **512**（BERT `max_position_embeddings=512`；OPT 配置的 2048 会越界崩溃）。
- vLLM 原生 BERT serving（`llm_engine.py:228` 的 AUXLLM 走独立模型注册表，仓库只有 `OPTForSequenceClassification`，无 bert.py）**超出本路径范围**——训练 + 离线 Tau 用 `PredModel` / trainer eval loop 完全不需要它。

---

## ② Train PARS

```bash
cd /hy-tmp/vllm-ltr/train
LLAMA3_TOK=/hy-tmp/models/Meta-Llama-3-70B

python trainer.py \
  --config configs/config_prefill_bert.txt \
  --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
  --job-dir MODEL \
  --run-id bert-pars-llama3-70b-lmsys-margin1.0-delta0.2-b128-OURS \
  --batch-size 128 \
  --label-group-size 10 \
  --loss marginRanking \
  --margin 1.0 \
  --delta 0.2 \
  --tokenizer "$LLAMA3_TOK"
```

- `--batch-size 128` 对齐 PARS（且 batch=slate，越大 pair 越多 ~128×127）；BERT-base(110M)+seq512，4090 48GB 轻松装下。PARS 超参：epoch 5（默认）、Adam、lr 2e-5（默认）、margin 1.0、delta 0.2。
- 产物同样落 `MODEL/results/<run-id>/finetuned/model.safetensors` + `usage_config.json`。
- 预计 **5–10 min**（BERT-base 比 OPT-350M 小很多）。

> 想跑 backbone 消融（PARS Table III: BERT>OPT>T5）就把 `--config` 换成 `config_prefill_opt_350m.txt` 配 `--loss marginRanking` 再跑一遍——同 loss、不同 backbone，可选。

---

## ③ Test — the headline result (Table E1: generalization gap)

仓库**没有**独立 eval 脚本，也**没有** cross-distribution flag（trainer.py:117-118 硬编码同一文件 90/10 同分布切分）。需要自己写 ~50 行脚本，复用 trainer 的 eval loop，在**两个不同 trace 文件**上算 Tau。

**关键 footgun**: 用 `RankingDataset`（group-aware, trainer.py:50-52，除以 `label_group_size`）**不是** `RankingTestDataset`（trainer.py:72-74 不除），并传**和训练相同的** `--label-group-size 10`，否则 true label 空间和 predictor 学到的空间对不上。

**新文件** `/Users/alex/develop/vllm-ltr/train/eval_gap.py`:
```python
import argparse, json, math, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy.stats import kendalltau
from vllm.config_predictor import PrefillPredictorConfig
from vllm.model_executor.prefill_predictor import prefill_predictor_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from trainer import RankingDataset   # reuse group-aware dataset


def tau_on(predictor, cfg, file, llama_tok, group_size, label_max=8192, tail=None):
    data = [json.loads(l) for l in open(file)]
    if tail:                                   # in-dist: same 10% tail trainer evaluates
        data = data[int(0.9 * len(data)):]
    ds = RankingDataset(data, llama_tok, max_length=cfg.model.max_length,
                        label_max_length=label_max, label_group_size=group_size)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    true_labels, preds = [], []
    predictor.model.eval()
    with torch.no_grad():
        for prompt, labels, origin_len in dl:
            enc = predictor.tokenizer(list(prompt), max_length=cfg.model.max_length,
                                      padding=True, truncation=True, return_tensors="pt")
            ii = enc['input_ids'].to("cuda:0"); am = enc['attention_mask'].to("cuda:0")
            with torch.autocast(device_type="cuda"):
                out = predictor(ii, am)
            preds.extend(out.squeeze().tolist())
            true_labels.extend(labels.tolist())
    return kendalltau(true_labels, preds)[0]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--usage-config", required=True)   # MODEL/results/<run-id>/usage_config.json
    p.add_argument("--in-file", required=True)        # lmsys-70B-c20000:30000 (use --tail)
    p.add_argument("--cross-file", required=True)     # sharegpt-70B-c20000:30000
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--tokenizer", default="/hy-tmp/models/Meta-Llama-3-70B")
    a = p.parse_args()

    cfg = PrefillPredictorConfig.from_json(a.usage_config)
    if cfg.model.num_labels == -1:
        cfg.model.num_labels = math.ceil(8192 / a.group_size)
    llama_tok = AutoTokenizer.from_pretrained(a.tokenizer)
    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            predictor = prefill_predictor_model(
                pred_model=cfg.model.path, num_labels=cfg.model.num_labels,
                mtype=cfg.model.mtype, activation=cfg.model.activation,
                max_length=cfg.model.max_length, max_batch_size=cfg.model.max_batch_size)

    tau_in    = tau_on(predictor, cfg, a.in_file,    llama_tok, a.group_size, tail=True)
    tau_cross = tau_on(predictor, cfg, a.cross_file, llama_tok, a.group_size)
    print(f"Tau(in-dist lmsys held-out) = {tau_in:.3f}")
    print(f"Tau(cross-dist sharegpt)    = {tau_cross:.3f}")
    print(f"GAP = {tau_in - tau_cross:.3f}")
```

跑两个 predictor:
```bash
LMSYS=jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl
SHAREGPT=jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl

# listMLE 基线
python eval_gap.py --usage-config MODEL/results/opt-350m-llama3-70b-lmsys-listMLE-b32-OURS/usage_config.json \
  --in-file $LMSYS --cross-file $SHAREGPT --group-size 10

# PARS
python eval_gap.py --usage-config MODEL/results/bert-pars-llama3-70b-lmsys-margin1.0-delta0.2-b128-OURS/usage_config.json \
  --in-file $LMSYS --cross-file $SHAREGPT --group-size 10
```

**两者都只在 lmsys 上训练**，sharegpt 全程未见 = 真 cross-distribution。

**Table E1 (生成的 deliverable 表)**:

| Predictor (OURS) | Backbone | Loss | Tau (LMSYS in-dist) | Tau (ShareGPT cross) | Gap = in − cross |
|---|---|---|---|---|---|
| listMLE baseline | OPT-350M | listMLE | ~0.62 (复刻 README) | **待测** | **待测** |
| PARS | BERT-base | margin(1.0)+delta(0.2) | 待测 | 待测 | **应更小**（假设） |

- **headline**: PARS 的 Gap 应小于 listMLE 的 Gap（PARS 抗过拟合/更好泛化）。**这些 cross 数字仓库任何地方都没产出过，必须自己测，不得编造**（README 的 lmsys=0.62 / sharegpt=0.55 都是各自**同分布**训练得到的，不是 cross）。
- **ShareGPT held-out trace 文件名**（HF dataset `LLM-ltr/Llama3-Trace`）: `sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl`（70B，与 predictor 训练目标模型一致）。另有 8B 测试集 `llama3-8b-sharegpt-test-t1-s0-8192.jsonl`，仅当你改用 125M/8B 基线时才用。

---

## ⑤ Risks / order of operations

**GPU 在线时做（最贵资源，先占住）**:
1. **立刻跑 ①** listMLE 基线（~15–30 min），存好 `MODEL/results/<run-id>/finetuned/model.safetensors`。
2. ② 代码写完后立刻跑 PARS（~5–10 min）。
3. 跑 ③ `eval_gap.py`（分钟级）。

**离线/无 GPU 时做（趁 ① 在跑）**:
- 写 `marginRanking.py`、`config_prefill_bert.txt`、trainer.py 三处编辑、`eval_gap.py`。全部纯 CPU 可写可 import-check。

**还要下载的数据**:
- 确认 `jsonfiles/` 里有 lmsys-70B-c20000:30000 训练文件 + **sharegpt-70B-c20000:30000**（cross 测试，③ 必需）。当前 `train/jsonfiles/` **是空的**（已查），需先：
  ```bash
  cd /hy-tmp/vllm-ltr/train && huggingface-cli download LLM-ltr/Llama3-Trace --local-dir jsonfiles --repo-type dataset
  ```
- 确认本地有 Llama-3 tokenizer 目录（`--tokenizer`），否则默认 `meta-llama/Meta-Llama-3-70B` 是 gated，会下载失败。

**诚实的未知 / 风险**:
- **耗时是估算**，未实测 4090 单 step 时间；以实际为准。
- **cross-dist Tau 值未知**——这正是 headline 结果，必须实测，禁止预填。
- **Delta-filter 是 per-batch 动态过滤**，非 PARS 原文的离线全局过滤；语义等价但属实现差异，deliverable 要披露。严格复刻需离线 `build_pairs.py`（非最小路径）。
- **BERT max_length=512 截断**长 prompt；lmsys/sharegpt prompt 一般短，影响小，但要在 evaluation 里注明。
- **方向一致性**: margin loss 强制 high-label(短生成)→high-score，与 listMLE 同向，确保与 scheduler 兼容；这是最易错的符号项，已在 (a) 处理。
- **vLLM 原生 BERT serving 超出范围**: 真在 vLLM scheduler 里跑 BERT predictor 需新写 `BertForSequenceClassification` + 注册（`models/__init__.py:47` 旁），是另一个大任务；训练 + 离线 Tau 的 capstone 路径不需要。

**关键文件路径汇总**:
- 改: `/Users/alex/develop/vllm-ltr/train/trainer.py`（import+argparse+loss 调用点 152-157）
- 新建: `/Users/alex/develop/vllm-ltr/train/allrank/models/losses/marginRanking.py`
- 新建: `/Users/alex/develop/vllm-ltr/train/configs/config_prefill_bert.txt`
- 新建: `/Users/alex/develop/vllm-ltr/train/eval_gap.py`
- 产物: `/Users/alex/develop/vllm-ltr/train/MODEL/results/<run-id>/finetuned/model.safetensors` + `usage_config.json`