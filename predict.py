#!/usr/bin/env python3
"""SIA inference: read a ticket CSV, output predictions + Evidence Dossiers.

Usage
  python predict.py --input new_tickets.csv --artifacts artifacts --outdir out
  python predict.py --input tickets.csv --dossiers-for all   # dossier per row

Outputs
  out/predictions.csv   - per-ticket judgment, confidence, inferred severity
  out/dossiers.json     - Evidence Dossier for every flagged ticket (exact
                          schema from the problem statement)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sia_core.config import DELTA_THRESHOLD, INT_TO_SEVERITY, SEVERITY_TO_INT
from sia_core.data import load_tickets, signal_text
from sia_core.dossier import audit_dossier_grounding, build_dossier
from sia_core.features import lite_features, serialized_text
from sia_core.fusion import fuse_scores, severity_level
from sia_core.lite_model import LiteMLP
from sia_core.signals import load_signals


def load_backend(artifacts: str):
    cfg = json.load(open(os.path.join(artifacts, "run_config.json")))
    if cfg["backend"] == "transformer":
        from sia_core.hf_model import HfClassifier
        clf = HfClassifier.load(os.path.join(artifacts, "hf_model"))
    else:
        clf = LiteMLP.load(os.path.join(artifacts, "lite_model.json"))
    return cfg, clf


def predict_frame(df: pd.DataFrame, artifacts: str):
    """Return (predictions DataFrame, per-row explain context)."""
    cfg, clf = load_backend(artifacts)
    rule, emb, rt = load_signals(os.path.join(artifacts, "signals.json"))

    if cfg["backend"] == "transformer":
        proba = clf.predict_proba(serialized_text(df, cfg["rt_median"]).tolist())
    else:
        proba = clf.predict_proba(lite_features(df, rule, emb, rt))
    pred = proba.argmax(1)
    conf = proba.max(1)

    # Stage-1 signal context (for dossiers + inferred severity reporting)
    from sia_core.fusion import compute_scores
    texts = signal_text(df).tolist()
    scores = compute_scores(rule, emb, rt, df)
    fused = fuse_scores(scores)
    level = severity_level(fused)
    assigned = np.array([SEVERITY_TO_INT.get(p, 1)
                         for p in df["Priority_Level"]])
    delta = level - assigned

    out = df[["Ticket_ID", "Priority_Level"]].copy()
    out["judgment"] = np.where(pred == 1, "Mismatched", "Consistent")
    out["confidence"] = np.round(conf, 4)
    out["inferred_severity"] = [INT_TO_SEVERITY[i] for i in level]
    out["severity_delta"] = delta
    out["mismatch_type"] = np.where(
        pred == 0, "None", np.where(delta > 0, "Hidden Crisis", "False Alarm"))
    ctx = {"rule": rule, "emb": emb, "rt": rt, "texts": texts,
           "level": level, "delta": delta, "pred": pred, "conf": conf}
    return out, ctx


def dossiers_for(df: pd.DataFrame, out: pd.DataFrame, ctx, which="flagged"):
    rows = []
    for i in range(len(df)):
        if which == "flagged" and ctx["pred"][i] != 1:
            continue
        row = df.iloc[i]
        delta = int(ctx["delta"][i])
        mtype = out["mismatch_type"].iloc[i]
        if mtype == "None":  # classifier flagged but fusion delta is 0/1
            mtype = "Hidden Crisis" if delta >= 0 else "False Alarm"
        d = build_dossier(
            row,
            assigned_priority=str(row["Priority_Level"]),
            inferred_level=int(ctx["level"][i]),
            delta=delta,
            mismatch_type=mtype,
            confidence=float(ctx["conf"][i]),
            rule_explain=ctx["rule"].explain(ctx["texts"][i]),
            emb_explain=ctx["emb"].explain(ctx["texts"][i]),
            rt_explain=ctx["rt"].explain(
                row.get("Resolution_Time_Hours", float("nan"))),
        )
        violations = audit_dossier_grounding(d, row)
        d["grounding_audit"] = "PASS" if not violations else violations
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--dossiers-for", choices=["flagged", "all"],
                    default="flagged")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_tickets(args.input)
    out, ctx = predict_frame(df, args.artifacts)
    out.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    docs = dossiers_for(df, out, ctx, args.dossiers_for)
    with open(os.path.join(args.outdir, "dossiers.json"), "w") as f:
        json.dump(docs, f, indent=2)

    n_flag = int((out["judgment"] == "Mismatched").sum())
    n_fail = sum(1 for d in docs if d["grounding_audit"] != "PASS")
    print(f"{len(df)} tickets -> {n_flag} flagged "
          f"({(out['mismatch_type'] == 'Hidden Crisis').sum()} Hidden Crisis, "
          f"{(out['mismatch_type'] == 'False Alarm').sum()} False Alarm)")
    print(f"dossiers written: {len(docs)}  grounding audit failures: {n_fail}")
    print(f"-> {args.outdir}/predictions.csv, {args.outdir}/dossiers.json")


if __name__ == "__main__":
    main()
