"""Classifier input features.

Two consumers:
  - transformer backend: serializes metadata into the text sequence
  - lite backend: LSA text embedding + one-hot structured metadata vector
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import domain_tier, full_text

CHANNELS = ["Chat", "Email", "Web Form"]
CATEGORIES = ["Account", "Billing", "Fraud", "General Inquiry", "Technical"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
TIERS = ["consumer", "business"]


def serialized_text(df: pd.DataFrame, rt_median: float) -> pd.Series:
    """Text+metadata serialization for the transformer backend."""
    rt = df["Resolution_Time_Hours"].fillna(rt_median).round(0).astype(int)
    tier = pd.Series(domain_tier(df["Customer_Email"]), index=df.index)
    return ("assigned priority: " + df["Priority_Level"].astype(str)
            + " | channel: " + df["Ticket_Channel"].astype(str)
            + " | customer tier: " + tier
            + " | category: " + df["Issue_Category"].astype(str)
            + " | resolution hours: " + rt.astype(str)
            + " | " + full_text(df))


def _onehot(values, vocab):
    M = np.zeros((len(values), len(vocab)), dtype=np.float32)
    lut = {v: i for i, v in enumerate(vocab)}
    for r, v in enumerate(values):
        j = lut.get(v)
        if j is not None:
            M[r, j] = 1.0
    return M


def metadata_matrix(df: pd.DataFrame, rt_signal) -> np.ndarray:
    """Structured metadata features for the lite backend.

    Includes the assigned priority (the quantity being audited), channel,
    customer domain tier, category, and resolution-time percentile.
    """
    tier = domain_tier(df["Customer_Email"])
    rt_pct = rt_signal.percentile(df["Resolution_Time_Hours"].to_numpy())
    parts = [
        _onehot(df["Priority_Level"].tolist(), PRIORITIES),
        _onehot(df["Ticket_Channel"].tolist(), CHANNELS),
        _onehot(list(tier), TIERS),
        _onehot(df["Issue_Category"].tolist(), CATEGORIES),
        rt_pct.reshape(-1, 1).astype(np.float32),
    ]
    return np.hstack(parts)


def lite_features(df: pd.DataFrame, rule, emb, rt_signal) -> np.ndarray:
    """LSA text embedding + structured metadata + the three Stage-1 signal
    scores. Including the signal scores distills the fusion teacher into the
    classifier and gives it out-of-vocabulary robustness (the rule signal
    handles negation/escalation that bag-of-words embeddings cannot)."""
    from .data import signal_text
    from .fusion import compute_scores, fuse_scores
    texts = signal_text(df).tolist()
    E = emb.encoder.encode(texts)
    M = metadata_matrix(df, rt_signal)
    sd = compute_scores(rule, emb, rt_signal, df)
    fused = fuse_scores(sd)
    assigned = _onehot(df["Priority_Level"].tolist(), PRIORITIES) @ \
        np.arange(4, dtype=np.float32)
    S = np.stack([sd["rule"], sd["embedding"], sd["resolution_time"],
                  fused.astype(np.float32),
                  (fused - assigned).astype(np.float32)], axis=1)
    return np.hstack([E.astype(np.float32), M, S.astype(np.float32)])
