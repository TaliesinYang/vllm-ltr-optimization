# Generalization-gap eval (Table E1): Kendall's Tau in-distribution (held-out LMSYS) vs
# cross-distribution (ShareGPT, never seen in training). PARS should shrink the gap vs listMLE.
# Capstone CSCI 6806. NOTE: the cross-dist numbers are produced ONLY here — never fabricate them.
import argparse, json, math, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy.stats import kendalltau
from vllm.config_predictor import PrefillPredictorConfig
from vllm.model_executor.prefill_predictor import prefill_predictor_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from trainer import RankingDataset   # reuse the group-aware dataset (divides by label_group_size)


def tau_on(predictor, cfg, file, llama_tok, group_size, label_max=8192, held_out_tail=False):
    data = [json.loads(l) for l in open(file)]
    if held_out_tail:                              # in-dist: the same 10% tail trainer evaluates on
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
    p.add_argument("--in-file", required=True)         # lmsys-...-c20000:30000  (uses --held-out tail)
    p.add_argument("--cross-file", required=True)      # sharegpt-...-c20000:30000
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--tokenizer", default="/hy-tmp/models/Meta-Llama-3-8B-Instruct")
    a = p.parse_args()

    cfg = PrefillPredictorConfig.from_json(a.usage_config)
    if cfg.model.num_labels == -1:
        cfg.model.num_labels = math.ceil(8192 / a.group_size)
    llama_tok = AutoTokenizer.from_pretrained(a.tokenizer)
    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            # weights load from finetuned path; tokenizer from the BASE backbone (finetuned dir has no tokenizer)
            predictor = prefill_predictor_model(
                pred_model=cfg.model.path, num_labels=cfg.model.num_labels,
                mtype=cfg.model.mtype, activation=cfg.model.activation,
                max_length=cfg.model.max_length, max_batch_size=cfg.model.max_batch_size,
                tokenizer_name=cfg.model.pred_model)

    tau_in    = tau_on(predictor, cfg, a.in_file,    llama_tok, a.group_size, held_out_tail=True)
    tau_cross = tau_on(predictor, cfg, a.cross_file, llama_tok, a.group_size)
    print(f"Tau(in-dist  LMSYS held-out) = {tau_in:.3f}")
    print(f"Tau(cross    ShareGPT)       = {tau_cross:.3f}")
    print(f"GAP = Tau_in - Tau_cross     = {tau_in - tau_cross:.3f}")
