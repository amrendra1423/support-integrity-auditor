"""Evaluation metrics (numpy implementations; no sklearn dependency)."""
from __future__ import annotations

import numpy as np

from .config import THRESHOLDS

CLASS_NAMES = {0: "Consistent", 1: "Mismatched"}


def binary_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    out = {"accuracy": float(np.mean(y_true == y_pred)), "per_class": {}}
    f1s = []
    for c in (0, 1):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        out["per_class"][CLASS_NAMES[c]] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": int(np.sum(y_true == c)),
        }
    out["macro_f1"] = float(np.mean(f1s))
    out["confusion_matrix"] = {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }
    out["verification"] = verify(out)
    return out


def verify(report: dict) -> dict:
    checks = {
        f"accuracy >= {THRESHOLDS['accuracy']}":
            report["accuracy"] >= THRESHOLDS["accuracy"],
        f"macro_f1 >= {THRESHOLDS['macro_f1']}":
            report["macro_f1"] >= THRESHOLDS["macro_f1"],
        f"recall(Consistent) >= {THRESHOLDS['per_class_recall']}":
            report["per_class"]["Consistent"]["recall"]
            >= THRESHOLDS["per_class_recall"],
        f"recall(Mismatched) >= {THRESHOLDS['per_class_recall']}":
            report["per_class"]["Mismatched"]["recall"]
            >= THRESHOLDS["per_class_recall"],
    }
    return {"checks": {k: bool(v) for k, v in checks.items()},
            "passed": bool(all(checks.values()))}


def format_report(report: dict) -> str:
    lines = [f"accuracy : {report['accuracy']:.4f}",
             f"macro F1 : {report['macro_f1']:.4f}"]
    for name, m in report["per_class"].items():
        lines.append(f"{name:<11} P={m['precision']:.4f} R={m['recall']:.4f} "
                     f"F1={m['f1']:.4f} (n={m['support']})")
    cm = report["confusion_matrix"]
    lines.append(f"confusion  TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    lines.append("verification: " +
                 ("PASSED" if report["verification"]["passed"] else "FAILED"))
    for k, v in report["verification"]["checks"].items():
        lines.append(f"  [{'x' if v else ' '}] {k}")
    return "\n".join(lines)
