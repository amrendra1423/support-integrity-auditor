"""Data loading, normalization, and stratified splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BUSINESS_DOMAINS, RANDOM_SEED, REQUIRED_COLS, SPLIT


def load_tickets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    if "Resolution_Time_Hours" not in df.columns:
        df["Resolution_Time_Hours"] = np.nan
    df["Resolution_Time_Hours"] = pd.to_numeric(
        df["Resolution_Time_Hours"], errors="coerce")
    df["Ticket_Subject"] = df["Ticket_Subject"].fillna("").astype(str)
    df["Ticket_Description"] = df["Ticket_Description"].fillna("").astype(str)
    return df


def full_text(df: pd.DataFrame) -> pd.Series:
    return (df["Ticket_Subject"].str.strip() + ". " +
            df["Ticket_Description"].str.strip())


def signal_text(df: pd.DataFrame) -> pd.Series:
    """Text scored by Stage-1 signals.

    Consistency analysis (see README) shows Ticket_Subject is statistically
    independent of Ticket_Description in this dataset (subjects are sampled
    noise), so severity signals score the description only. The subject is
    still given to the Stage-2 classifier, which can learn to discount it.
    """
    return df["Ticket_Description"].str.strip()


def domain_tier(email: pd.Series) -> pd.Series:
    dom = email.astype(str).str.split("@").str[-1].str.lower()
    return np.where(dom.isin(BUSINESS_DOMAINS), "business", "consumer")


def stratified_split(labels: np.ndarray, seed: int = RANDOM_SEED):
    """Return index arrays (train, val, test) stratified on the binary label."""
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(SPLIT["train"] * n))
        n_va = int(round(SPLIT["val"] * n))
        idx_tr.append(idx[:n_tr])
        idx_va.append(idx[n_tr:n_tr + n_va])
        idx_te.append(idx[n_tr + n_va:])
    return (np.sort(np.concatenate(idx_tr)),
            np.sort(np.concatenate(idx_va)),
            np.sort(np.concatenate(idx_te)))
