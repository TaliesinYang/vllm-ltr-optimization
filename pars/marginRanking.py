# PARS pairwise margin ranking loss (Tao et al. 2025, arXiv:2510.03243v2).
# Capstone CSCI 6806 — Dazhi Yang. Mirrors the pair construction of rankNet.py but uses a
# hinge (margin) objective and an optional PARS delta-filter on noisy pairs.
from itertools import product
import torch
from allrank.data.dataset_loading import PADDED_Y_VALUE


def marginRanking(y_pred, y_true, true_lengths=None, margin=1.0, delta=0.0,
                  padded_value_indicator=PADDED_Y_VALUE):
    """
    L = mean_{kept pairs} max(0, margin - (s_high - s_low)).

    HIGHER y_true (= shorter-generation bucket, per trainer.__len2label__) must get the HIGHER
    score, keeping the SAME ordering convention as listMLE (which sorts y_true descending). This
    keeps the trained predictor plug-compatible with the existing vLLM scheduler.

    :param y_pred:       predictor scores,           shape [1, slate_length]
    :param y_true:       length-bucket labels,       shape [1, slate_length]
    :param true_lengths: raw generated token lengths [1, slate_length] (for the delta filter)
    :param margin:       hinge margin (PARS fixes 1.0)
    :param delta:        min relative length diff to keep a pair (PARS: 0.2 Llama/GPT-4, 0.25 DS-R1)
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    pairs = list(product(range(y_true.shape[1]), repeat=2))     # mirrors rankNet.py
    pairs_true = y_true[:, pairs]
    pairs_pred = y_pred[:, pairs]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = pairs_pred[:, :, 0] - pairs_pred[:, :, 1]      # s_high - s_low

    keep = (true_diffs > 0) & (~torch.isinf(true_diffs))       # ordered, non-padded pairs

    # PARS delta-filter (Eq.1): keep a pair only if |L_A - L_B| / max(L_A, L_B) >= delta
    if delta > 0.0 and true_lengths is not None:
        lengths = true_lengths.to(y_pred.device).float()
        pl = lengths[:, pairs]
        l_hi, l_lo = pl[:, :, 0], pl[:, :, 1]
        rel_diff = (l_hi - l_lo).abs() / torch.clamp(torch.maximum(l_hi, l_lo), min=1.0)
        keep = keep & (rel_diff >= delta)

    pred_diffs = pred_diffs[keep]
    if pred_diffs.numel() == 0:
        return y_pred.new_zeros(1, requires_grad=True).squeeze()

    return torch.clamp(margin - pred_diffs, min=0.0).mean()    # = max(0, -y*(s_A - s_B) + margin), y=+1
