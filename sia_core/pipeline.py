"""Stage-1 orchestration: fit signals, fuse, derive pseudo-labels + diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import signal_text
from .fusion import (ablation_labels, compute_scores, fuse_scores,
                     mismatch_labels, severity_level, signal_agreement)
from .signals import EmbeddingSignal, ResolutionTimeSignal, RuleSignal


class Stage1Pipeline:
    def __init__(self, encoder=None, k: int = 40):
        self.rule = RuleSignal()
        self.emb = EmbeddingSignal(encoder=encoder, k=k)
        self.rt = ResolutionTimeSignal()

    def fit(self, df: pd.DataFrame):
        texts = signal_text(df).tolist()
        self.emb.fit(texts)
        self.rt.fit(df["Resolution_Time_Hours"].to_numpy())
        return self

    def scores(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return compute_scores(self.rule, self.emb, self.rt, df)

    def pseudo_label(self, df: pd.DataFrame) -> dict:
        sd = self.scores(df)
        fused = fuse_scores(sd)
        level = severity_level(fused)
        lab = mismatch_labels(level, df["Priority_Level"].tolist())
        return {
            "scores": sd,
            "fused": fused,
            "inferred_level": level,
            **lab,
            "agreement": signal_agreement(sd),
        }

    def ablation(self, df: pd.DataFrame) -> dict:
        sd = self.scores(df)
        return ablation_labels(sd, df["Priority_Level"].tolist())
