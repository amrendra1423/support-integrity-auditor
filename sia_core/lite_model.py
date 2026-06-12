"""Dependency-light classifier backend: a 2-layer MLP in pure numpy.

Used for sandboxed/CI verification of the full pipeline. The submission-grade
backend (hf_model.py) fine-tunes a pretrained transformer on the identical
pseudo-labels and inputs; both expose the same fit/predict_proba interface.

Class imbalance is addressed with inverse-frequency class weighting in the
cross-entropy loss (spec Stage-2 requirement).
"""
from __future__ import annotations

import json

import numpy as np

from .config import RANDOM_SEED


class LiteMLP:
    def __init__(self, hidden: int = 64, lr: float = 3e-3, epochs: int = 30,
                 batch_size: int = 256, seed: int = RANDOM_SEED,
                 weight_decay: float = 1e-5):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.batch_size, self.seed, self.weight_decay = batch_size, seed, weight_decay
        self.params = None
        self.mu = None
        self.sd = None

    # ----------------------------------------------------------------- utils
    def _init(self, d_in):
        rng = np.random.default_rng(self.seed)
        s1 = np.sqrt(2.0 / d_in)
        s2 = np.sqrt(2.0 / self.hidden)
        self.params = {
            "W1": rng.normal(0, s1, (d_in, self.hidden)).astype(np.float64),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, s2, (self.hidden, 2)).astype(np.float64),
            "b2": np.zeros(2),
        }

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def _forward(self, X):
        h_pre = X @ self.params["W1"] + self.params["b1"]
        h = np.maximum(h_pre, 0)
        logits = h @ self.params["W2"] + self.params["b2"]
        return h_pre, h, self._softmax(logits)

    # ------------------------------------------------------------------ API
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        X = (X - self.mu) / self.sd

        # Inverse-frequency class weights (imbalance handling).
        counts = np.bincount(y, minlength=2).astype(np.float64)
        cw = counts.sum() / (2.0 * counts)
        if verbose:
            print(f"class counts={counts.astype(int).tolist()} "
                  f"weights={np.round(cw, 3).tolist()}")

        self._init(X.shape[1])
        rng = np.random.default_rng(self.seed)
        n = len(X)
        # Adam state
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(vv) for k, vv in self.params.items()}
        b1m, b2m, eps, t = 0.9, 0.999, 1e-8, 0

        history = []
        for ep in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                xb, yb = X[idx], y[idx]
                wb = cw[yb]
                _, h, p = self._forward(xb)
                # weighted CE gradient
                g_logits = p.copy()
                g_logits[np.arange(len(yb)), yb] -= 1.0
                g_logits *= (wb / wb.sum())[:, None]
                grads = {
                    "W2": h.T @ g_logits + self.weight_decay * self.params["W2"],
                    "b2": g_logits.sum(0),
                }
                g_h = g_logits @ self.params["W2"].T
                g_h[h <= 0] = 0.0
                grads["W1"] = xb.T @ g_h + self.weight_decay * self.params["W1"]
                grads["b1"] = g_h.sum(0)
                t += 1
                for k in self.params:
                    m[k] = b1m * m[k] + (1 - b1m) * grads[k]
                    v[k] = b2m * v[k] + (1 - b2m) * grads[k] ** 2
                    mh = m[k] / (1 - b1m ** t)
                    vh = v[k] / (1 - b2m ** t)
                    self.params[k] -= self.lr * mh / (np.sqrt(vh) + eps)
            if verbose and (ep % 5 == 0 or ep == self.epochs - 1):
                msg = f"epoch {ep:>3}"
                if X_val is not None:
                    acc = float(np.mean(self.predict(X_val) == y_val))
                    msg += f"  val_acc={acc:.4f}"
                history.append(msg)
                print(msg)
        return self

    def predict_proba(self, X):
        X = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return self._forward(X)[2]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        blob = {k: v.tolist() for k, v in self.params.items()}
        blob.update({"mu": self.mu.tolist(), "sd": self.sd.tolist(),
                     "hidden": self.hidden})
        with open(path, "w") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            blob = json.load(f)
        model = cls(hidden=blob["hidden"])
        model.params = {k: np.array(blob[k]) for k in ("W1", "b1", "W2", "b2")}
        model.mu = np.array(blob["mu"])
        model.sd = np.array(blob["sd"])
        return model
