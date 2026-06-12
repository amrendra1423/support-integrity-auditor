#!/usr/bin/env python3
"""SIA standalone training script.

Pipeline: Stage-1 pseudo-labeling (rule + embedding + resolution-time fusion)
-> diagnostics (signal agreement, ablation) -> Stage-2 classifier training
-> held-out evaluation against the verification thresholds.

Backends
  --backend transformer : fine-tuned DeBERTa-v3-small (submission grade;
                          needs torch/transformers, internet for weights)
  --backend lite        : pure-numpy LSA+MLP (offline verification / CI)

Examples
  python train_pipeline.py --data customer_support_tickets.csv
  python train_pipeline.py --data tickets.csv --backend lite --outdir artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sia_core.augment import augmented_tickets
from sia_core.config import FUSION_WEIGHTS, RANDOM_SEED
from sia_core.data import load_tickets, stratified_split
from sia_core.features import lite_features, serialized_text
from sia_core.fusion import fuse_scores, mismatch_labels, severity_level
from sia_core.lite_model import LiteMLP
from sia_core.metrics import binary_report, format_report
from sia_core.pipeline import Stage1Pipeline
from sia_core.signals import save_signals


def run_ablation(stage1, df, idx_tr, idx_te, outpath):
    """Train a light classifier on labels from every signal subset; evaluate
    against the full-fusion consensus on the held-out test split."""
    sd = stage1.scores(df)
    full = mismatch_labels(severity_level(fuse_scores(sd)),
                           df["Priority_Level"].tolist())["label"]
    X = lite_features(df, stage1.rule, stage1.emb, stage1.rt)
    from itertools import combinations
    names = sorted(sd)
    subsets = [(n,) for n in names] + list(combinations(names, 2)) \
        + [tuple(names)]
    table = {}
    for sub in subsets:
        labels = mismatch_labels(
            severity_level(fuse_scores({k: sd[k] for k in sub})),
            df["Priority_Level"].tolist())["label"]
        clf = LiteMLP(epochs=12)
        clf.fit(X[idx_tr], labels[idx_tr], verbose=False)
        pred = clf.predict(X[idx_te])
        rep = binary_report(full[idx_te], pred)
        table["+".join(sub)] = {
            "pseudo_mismatch_rate": float(labels.mean()),
            "label_agreement_with_fusion": float(np.mean(labels == full)),
            "test_macro_f1_vs_fusion": rep["macro_f1"],
            "test_recall_mismatched_vs_fusion":
                rep["per_class"]["Mismatched"]["recall"],
        }
    with open(outpath, "w") as f:
        json.dump(table, f, indent=2)
    print("\n=== Ablation (light classifier, evaluated vs full fusion) ===")
    for k, v in table.items():
        print(f"  {k:<42} rate={v['pseudo_mismatch_rate']:.3f} "
              f"agree={v['label_agreement_with_fusion']:.3f} "
              f"mF1={v['test_macro_f1_vs_fusion']:.3f}")
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--backend", choices=["transformer", "lite"],
                    default="transformer")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-small")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA adapter training instead of full fine-tune")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--k", type=int, default=40, help="KMeans clusters")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------- Stage 1
    print(f"[1/4] Loading {args.data}")
    df = load_tickets(args.data)
    print(f"      {len(df)} tickets")

    print("[2/4] Stage 1: fitting signals + pseudo-labeling")
    stage1 = Stage1Pipeline(k=args.k).fit(df)
    res = stage1.pseudo_label(df)
    y = res["label"]
    print(f"      mismatch rate={y.mean():.4f}  "
          f"types={pd.Series(res['mismatch_type']).value_counts().to_dict()}")
    print("      signal agreement:")
    for pair, m in res["agreement"].items():
        print(f"        {pair}: level={m['level_agreement']:.3f} "
              f"kappa={m['cohens_kappa']:.3f} rho={m['spearman_rho']:.3f}")
    save_signals(os.path.join(args.outdir, "signals.json"),
                 stage1.emb, stage1.rt)

    pseudo = df[["Ticket_ID"]].copy()
    pseudo["inferred_severity"] = res["inferred_severity"]
    pseudo["severity_delta"] = res["delta"]
    pseudo["mismatch_label"] = y
    pseudo["mismatch_type"] = res["mismatch_type"]
    for name, s in res["scores"].items():
        pseudo[f"score_{name}"] = s
    pseudo["fused_score"] = res["fused"]
    pseudo.to_csv(os.path.join(args.outdir, "pseudo_labels.csv"), index=False)

    idx_tr, idx_va, idx_te = stratified_split(y)
    json.dump({"train": idx_tr.tolist(), "val": idx_va.tolist(),
               "test": idx_te.tolist()},
              open(os.path.join(args.outdir, "splits.json"), "w"))

    # ---------------------------------------------------------- Ablation
    if not args.skip_ablation:
        run_ablation(stage1, df, idx_tr, idx_te,
                     os.path.join(args.outdir, "ablation.json"))

    # ---------------------------------------------------------- Stage 2
    print(f"[3/4] Stage 2: training classifier (backend={args.backend})")
    df_tr, y_tr = df.iloc[idx_tr], y[idx_tr]
    if not args.no_augment:
        aug = augmented_tickets()
        aug_res = stage1.pseudo_label(aug)
        df_tr = pd.concat([df_tr, aug], ignore_index=True)
        y_tr = np.concatenate([y_tr, aug_res["label"]])
        print(f"      +{len(aug)} augmented tickets "
              f"(aug mismatch rate={aug_res['label'].mean():.3f})")

    rt_median = float(np.nanmedian(df["Resolution_Time_Hours"]))
    json.dump({"backend": args.backend, "rt_median": rt_median,
               "fusion_weights": FUSION_WEIGHTS, "seed": RANDOM_SEED},
              open(os.path.join(args.outdir, "run_config.json"), "w"))

    if args.backend == "transformer":
        from sia_core.hf_model import HfClassifier
        clf = HfClassifier(model_name=args.model_name, lora=args.lora,
                           epochs=args.epochs or 3)
        tx_tr = serialized_text(df_tr, rt_median).tolist()
        tx_va = serialized_text(df.iloc[idx_va], rt_median).tolist()
        tx_te = serialized_text(df.iloc[idx_te], rt_median).tolist()
        clf.fit(tx_tr, y_tr, tx_va, y[idx_va])
        clf.save(os.path.join(args.outdir, "hf_model"))
        te_pred, te_proba = clf.predict(tx_te), clf.predict_proba(tx_te)
    else:
        clf = LiteMLP(epochs=args.epochs or 30)
        X_tr = lite_features(df_tr, stage1.rule, stage1.emb, stage1.rt)
        X_va = lite_features(df.iloc[idx_va], stage1.rule, stage1.emb, stage1.rt)
        X_te = lite_features(df.iloc[idx_te], stage1.rule, stage1.emb, stage1.rt)
        clf.fit(X_tr, y_tr, X_va, y[idx_va])
        clf.save(os.path.join(args.outdir, "lite_model.json"))
        te_pred, te_proba = clf.predict(X_te), clf.predict_proba(X_te)

    # ---------------------------------------------------------- Evaluation
    print("[4/4] Held-out evaluation (test split)")
    report = binary_report(y[idx_te], te_pred)
    report["backend"] = args.backend
    report["n_test"] = int(len(idx_te))
    report["signal_agreement"] = res["agreement"]
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(format_report(report))
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts -> {args.outdir}/")


if __name__ == "__main__":
    main()
