"""Stage 1 fusion: combine signal scores into inferred severity + mismatch label."""
from __future__ import annotations

import numpy as np

from .config import DELTA_THRESHOLD, FUSION_WEIGHTS, INT_TO_SEVERITY, SEVERITY_TO_INT


NEGATION_EMB_CAP = 0.75


def compute_scores(rule, emb, rt, df) -> dict:
    """Stage-1 scores with the negation-aware veto.

    LSA / bag-of-words embeddings are blind to negation ("there is NO fraud"
    still lands in the fraud cluster), so when the rule signal detects an
    explicit negation of urgency terms and itself scores low, the embedding
    signal is capped at rule + NEGATION_EMB_CAP.
    """
    from .data import signal_text
    texts = signal_text(df).tolist()
    r = rule.score(texts)
    e = emb.score(texts)
    neg = rule.negation_flags(texts)
    cap_mask = neg & (r < 1.0)
    e = np.where(cap_mask, np.minimum(e, r + NEGATION_EMB_CAP), e)
    t = rt.score(df["Resolution_Time_Hours"].to_numpy())
    return {"rule": r, "embedding": e.astype(np.float32), "resolution_time": t}


def fuse_scores(score_dict: dict[str, np.ndarray],
                weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted average of continuous severity scores. Missing signals are
    dropped and remaining weights renormalized."""
    weights = weights or FUSION_WEIGHTS
    active = {k: v for k, v in score_dict.items() if v is not None}
    total = sum(weights[k] for k in active)
    fused = np.zeros_like(next(iter(active.values())), dtype=np.float64)
    for k, v in active.items():
        fused += (weights[k] / total) * v.astype(np.float64)
    return fused


def severity_level(fused: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(fused), 0, 3).astype(int)


def mismatch_labels(inferred_level: np.ndarray, assigned_priority) -> dict:
    assigned = np.array([SEVERITY_TO_INT[p] for p in assigned_priority])
    delta = inferred_level - assigned
    is_mismatch = (np.abs(delta) >= DELTA_THRESHOLD).astype(int)
    mtype = np.where(delta > 0, "Hidden Crisis",
                     np.where(delta < 0, "False Alarm", "None"))
    mtype = np.where(is_mismatch == 1, mtype, "None")
    return {"assigned_int": assigned, "delta": delta,
            "label": is_mismatch, "mismatch_type": mtype,
            "inferred_severity": np.array([INT_TO_SEVERITY[i] for i in inferred_level])}


# ---------------------------------------------------------------------------
# Diagnostics: pairwise signal agreement + ablation support
# ---------------------------------------------------------------------------
def _discretize(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(scores), 0, 3).astype(int)


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum((np.mean(a == c)) * (np.mean(b == c)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        # average ties
        sx = x[order]
        i = 0
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def signal_agreement(score_dict: dict[str, np.ndarray]) -> dict:
    """Pairwise agreement between signals (the spec's 'Pseudo-Label Signal
    Agreement' metric): exact level agreement, Cohen's kappa, Spearman rho."""
    names = sorted(score_dict)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = _discretize(score_dict[a]), _discretize(score_dict[b])
            out[f"{a}|{b}"] = {
                "level_agreement": float(np.mean(da == db)),
                "within_one_level": float(np.mean(np.abs(da - db) <= 1)),
                "cohens_kappa": _cohens_kappa(da, db),
                "spearman_rho": _spearman(score_dict[a], score_dict[b]),
            }
    return out


def ablation_labels(score_dict: dict[str, np.ndarray], assigned_priority) -> dict:
    """Mismatch labels from every signal subset, for the README ablation table."""
    from itertools import combinations
    names = sorted(score_dict)
    variants = {}
    subsets = [(n,) for n in names] + list(combinations(names, 2)) + [tuple(names)]
    for sub in subsets:
        sd = {k: score_dict[k] for k in sub}
        lab = mismatch_labels(severity_level(fuse_scores(sd)), assigned_priority)
        variants["+".join(sub)] = lab["label"]
    return variants
